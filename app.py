import matplotlib
matplotlib.use('Agg')  # headless server - no display backend available
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import json
import io
import re
from scipy import stats
import sqlite3
import uuid
import time
from datetime import datetime, timezone
import warnings
from ostrivo_core import (
    load_data, clean_data, detect_anomalies, compute_stats, detect_date_column,
    generate_forecast, pick_forecast_metric, humanize_column_name,
    get_data_quality_scores, get_heuristic_recommendations,
    generate_pdf_report, generate_excel_report,
    is_excel_file, get_excel_sheet_names, load_excel_sheet, rank_excel_sheets,
    combine_dataframes,
    INDUSTRY_OPTIONS, suggest_category_and_metric_columns,
    top_performers_analysis, concentration_risk_analysis, control_chart_analysis,
    validate_password_strength, time_trend_analysis, industry_kpi_summary,
    segment_categories, estimate_time_to_limit, binary_outcome_risk_model,
)
from supabase_backend import (
    get_supabase_client, sign_up, sign_in, sign_out, restore_session,
    save_analysis, list_saved_analyses, load_analysis, delete_analysis,
    update_industry, verify_signup_code, resend_signup_code, delete_own_account,
)
from streamlit_cookies_controller import CookieController
warnings.filterwarnings('ignore')

# ── Industry-specific AI framing ─────────────────────────────────────────────
INDUSTRY_AI_CONTEXT = {
    'sales_retail': (
        "The user runs a sales/retail business. Frame findings around revenue, top-selling "
        "products or categories, regional performance, and customer/demand trends. "
        "Recommendations should be commercially actionable (pricing, inventory, promotion)."
    ),
    'finance_banking': (
        "The user works in finance/banking. Frame findings around risk exposure, concentration "
        "risk, portfolio diversification, and transaction/balance patterns. Treat anomalies as "
        "potential fraud or compliance signals worth reviewing, not just statistical outliers."
    ),
    'engineering_manufacturing': (
        "The user works in engineering/manufacturing. Frame findings around process stability, "
        "defect or failure rates, tolerances, and control limits. Treat anomalies as potential "
        "process control violations worth investigating, not just statistical outliers."
    ),
    'healthcare': (
        "The user works in healthcare (a hospital or medical establishment). Frame findings "
        "around patient volume, department/ward load, wait times or length of stay, and "
        "readmission or outcome trends. Treat anomalies as potential care-quality or capacity "
        "signals worth investigating, not just statistical outliers. Never offer clinical "
        "advice - stay focused on operational and data-quality insights."
    ),
}


def get_current_industry():
    """Return the currently active industry key (from the logged-in user's account metadata,
    or the session-only guest fallback when Supabase isn't configured), or None if unset."""
    if supabase_client and 'auth_user' in st.session_state:
        return (st.session_state['auth_user'].user_metadata or {}).get('industry')
    return st.session_state.get('guest_industry')


GUEST_ANALYSIS_LIMIT = 7


def get_guest_uses():
    """How many analyses this browser's guest trial has already used. session_state is
    the source of truth for the running session (updates instantly, so the limit is
    enforced right away); it's seeded once per session from the persistent cookie, which
    is the cookie component's own async round-trip and only needs to resolve once, on
    first read, rather than on every increment."""
    if 'guest_uses' not in st.session_state:
        cookie_val = 0
        if cookie_controller:
            try:
                cookie_val = int(cookie_controller.get('ostrivo_guest_uses') or 0)
            except (TypeError, ValueError):
                cookie_val = 0
        st.session_state['guest_uses'] = cookie_val
    return st.session_state['guest_uses']


def increment_guest_uses():
    """Record one more guest analysis used - updates session_state immediately (so the
    limit is enforced within the same session, not just after a reload) and persists to
    a cookie so the count survives a page reload too. No account or database involved,
    so all of it - the count included - simply disappears once the guest clears cookies
    or switches browsers."""
    new_count = get_guest_uses() + 1
    st.session_state['guest_uses'] = new_count
    if cookie_controller:
        cookie_controller.set('ostrivo_guest_uses', str(new_count))


def render_verify_screen(pending_email):
    """Show the enter-your-code screen for a pending signup and handle verify/resend/cancel.
    Called both directly after a successful signup (same script run, no rerun needed - avoids
    relying on a rerun round-trip to pick the state back up) and on later reruns for as long as
    pending_signup_email stays in session_state (e.g. after a page reload)."""
    st.subheader("Verify your email")
    st.write(f"We sent a code to **{pending_email}**. Enter it below to activate your account.")
    verify_code = st.text_input("Verification code", key="verify_code")
    vc1, vc2 = st.columns(2)
    with vc1:
        if st.button("Verify & Log In", key="verify_code_btn"):
            try:
                result = verify_signup_code(supabase_client, pending_email, verify_code)
                st.session_state['auth_user'] = result.user
                st.session_state['sb_access_token'] = result.session.access_token
                st.session_state['sb_refresh_token'] = result.session.refresh_token
                cookie_controller.set('ostrivo_access_token', result.session.access_token)
                cookie_controller.set('ostrivo_refresh_token', result.session.refresh_token)
                st.session_state.pop('pending_signup_email', None)
                # The cookie component needs a render cycle to flush the write to the
                # browser before we tear down the DOM with rerun() - an immediate rerun
                # can cut it off, so give it a brief moment first.
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Verification failed: {e}")
    with vc2:
        if st.button("Resend code", key="resend_code_btn"):
            try:
                resend_signup_code(supabase_client, pending_email)
                st.success("Code resent - check your email.")
            except Exception as e:
                st.error(f"Couldn't resend: {e}")
    if st.button("Use a different email", key="cancel_signup_btn"):
        st.session_state.pop('pending_signup_email', None)
        st.rerun()
    st.stop()

# ── Admin logging (SQLite) ───────────────────────────────────────────────────
# Note: on Streamlit Community Cloud the filesystem is ephemeral, so this data
# persists only while the app instance stays running, and resets on redeploy.
# A production version should swap this for an external database.
ADMIN_DB_PATH = "ostrivo_admin.db"


def init_admin_db():
    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            detail TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE events ADD COLUMN industry TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists from a prior run
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())[:8]
    return st.session_state['session_id']


def log_event(event_type, detail="", industry=None):
    if industry is None:
        try:
            industry = get_current_industry()
        except NameError:
            industry = None  # called before the industry helpers are defined (e.g. admin gate)
    try:
        conn = sqlite3.connect(ADMIN_DB_PATH)
        conn.execute(
            "INSERT INTO events (ts, session_id, event_type, detail, industry) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), get_session_id(), event_type, str(detail)[:500], industry)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def log_api_call(kind, model, response=None):
    try:
        input_tokens = getattr(response.usage, 'input_tokens', None) if response else None
        output_tokens = getattr(response.usage, 'output_tokens', None) if response else None
        conn = sqlite3.connect(ADMIN_DB_PATH)
        conn.execute(
            "INSERT INTO api_calls (ts, session_id, kind, model, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), get_session_id(), kind, model, input_tokens, output_tokens)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_usage_stats():
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
    by_type = dict(cur.fetchall())
    cur.execute("SELECT COUNT(DISTINCT session_id) FROM events")
    sessions = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM events WHERE event_type = 'error'")
    errors = cur.fetchone()[0] or 0
    cur.execute("SELECT industry, COUNT(*) FROM events WHERE industry IS NOT NULL GROUP BY industry")
    by_industry = dict(cur.fetchall())
    conn.close()
    return {'by_type': by_type, 'sessions': sessions, 'errors': errors, 'by_industry': by_industry}


def get_activity_over_time():
    """Daily event counts and distinct-session counts, for a traffic-over-time view."""
    conn = sqlite3.connect(ADMIN_DB_PATH)
    activity_df = pd.read_sql_query("""
        SELECT date(ts) AS day, COUNT(*) AS events, COUNT(DISTINCT session_id) AS sessions
        FROM events GROUP BY day ORDER BY day
    """, conn)
    conn.close()
    return activity_df


def get_api_cost_estimate():
    """Rough cost estimate from token counts. Verify actual pricing at anthropic.com/pricing
    and real spend at console.anthropic.com - this is not authoritative billing data."""
    default_price = {'input': 3.0, 'output': 15.0}  # USD per million tokens, update if pricing changes
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT model, SUM(input_tokens), SUM(output_tokens), COUNT(*) FROM api_calls GROUP BY model")
    rows = cur.fetchall()
    conn.close()

    breakdown = []
    total_cost = 0.0
    for model, in_tok, out_tok, count in rows:
        in_tok, out_tok = in_tok or 0, out_tok or 0
        price = default_price
        cost = (in_tok / 1_000_000 * price['input']) + (out_tok / 1_000_000 * price['output'])
        total_cost += cost
        breakdown.append({
            'model': model or 'unknown', 'calls': count,
            'input_tokens': in_tok, 'output_tokens': out_tok,
            'est_cost_usd': round(cost, 4)
        })
    return {'total_est_cost_usd': round(total_cost, 4), 'breakdown': breakdown}


def get_recent_events(limit=100):
    conn = sqlite3.connect(ADMIN_DB_PATH)
    events_df = pd.read_sql_query(
        "SELECT ts, session_id, event_type, detail, industry FROM events ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return events_df


def admin_chat_query(question, api_key):
    """Answer an admin's natural-language question about aggregated app activity (no customer data)."""
    context = {
        'usage_stats': get_usage_stats(),
        'activity_over_time': get_activity_over_time().to_dict('records'),
        'api_cost_estimate': get_api_cost_estimate(),
        'recent_events': get_recent_events(50).to_dict('records'),
    }
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are an operations assistant for Ostrivo, a business intelligence web app.
Answer the admin's question using only this aggregated app-activity data (no customer data is included here).

APP ACTIVITY DATA:
{json.dumps(context, indent=2, default=str)}

ADMIN QUESTION: {question}

Provide a clear, direct answer in 2-5 sentences using the numbers available. If the data can't answer the question, say so."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    log_api_call("admin_chat", "claude-sonnet-4-6", response)
    return response.content[0].text


init_admin_db()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ostrivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Admin console (hidden behind ?admin=1, password-gated) ──────────────────
if st.query_params.get("admin") == "1":
    st.title("🔐 Ostrivo Admin Console")
    st.caption("Aggregated app activity only - no customer-uploaded data is stored or shown here.")

    if not st.session_state.get('admin_authenticated'):
        admin_password_input = st.text_input("Admin password", type="password", key="admin_password_input")
        if st.button("Login", key="admin_login_btn"):
            try:
                correct_password = st.secrets.get("admin_password")
            except Exception:
                correct_password = None

            if not correct_password:
                st.error("No admin password is configured. Set 'admin_password' in Streamlit secrets.")
            elif admin_password_input != correct_password:
                st.error("Incorrect password.")
            else:
                st.session_state['admin_authenticated'] = True
                st.rerun()
        st.stop()

    if st.button("Log Out", key="admin_logout_btn"):
        st.session_state.pop('admin_authenticated', None)
        st.rerun()

    stats = get_usage_stats()
    cost = get_api_cost_estimate()

    c1, c2, c3 = st.columns(3)
    c1.metric("Distinct Sessions", stats['sessions'])
    c2.metric("Total Events", sum(stats['by_type'].values()))
    c3.metric("Errors Logged", stats['errors'])

    st.subheader("Traffic Over Time")
    activity_df = get_activity_over_time()
    if not activity_df.empty:
        st.line_chart(activity_df.set_index('day')[['events', 'sessions']])
    else:
        st.caption("No activity logged yet.")

    st.subheader("Events by Type")
    if stats['by_type']:
        st.bar_chart(stats['by_type'])
    else:
        st.caption("No activity logged yet.")

    st.subheader("Industry Breakdown")
    st.caption("Which industries the activity above is coming from (based on each user's industry setting).")
    if stats['by_industry']:
        industry_labels = {INDUSTRY_OPTIONS.get(k, k): v for k, v in stats['by_industry'].items()}
        st.bar_chart(industry_labels)
    else:
        st.caption("No industry-tagged activity yet.")

    st.subheader("Estimated AI API Cost")
    st.caption("Rough estimate from token counts - verify actual spend at console.anthropic.com.")
    st.metric("Estimated Total Cost", f"${cost['total_est_cost_usd']}")
    if cost['breakdown']:
        st.dataframe(pd.DataFrame(cost['breakdown']), use_container_width=True, key="table_admin_cost_breakdown")

    st.subheader("Recent Activity Log")
    recent_events_df = get_recent_events(100)
    if not recent_events_df.empty:
        st.dataframe(recent_events_df, use_container_width=True, height=300, key="table_admin_recent_events")
    else:
        st.caption("No events logged yet.")

    st.subheader("Ask the AI About This Activity")
    admin_api_key = st.text_input("AI API key (for admin chat)", type="password", key="admin_api_key")
    admin_question = st.text_input("Ask a question about app activity", key="admin_question_input")
    if admin_question and admin_api_key and st.button("Ask", key="admin_ask_btn"):
        with st.spinner("Thinking..."):
            try:
                admin_answer = admin_chat_query(admin_question, admin_api_key)
                st.markdown(f"**Answer:** {admin_answer}")
            except Exception as e:
                st.error(f"API error: {e}")

    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }

    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.9rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
    }

    .insight-box h4 {
        color: #0c4a6e;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }

    .insight-box p {
        color: #075985;
        font-size: 0.87rem;
        margin: 0;
        line-height: 1.6;
    }

    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
    }

    .warning-box h4 {
        color: #78350f;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }

    .warning-box p {
        color: #92400e;
        font-size: 0.87rem;
        margin: 0;
        line-height: 1.6;
    }

    .anomaly-badge {
        display: inline-block;
        background: #fee2e2;
        color: #dc2626;
        border: 1px solid #fca5a5;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .normal-badge {
        display: inline-block;
        background: #dcfce7;
        color: #16a34a;
        border: 1px solid #86efac;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
    }

    @media (prefers-color-scheme: dark) {
        .section-title {
            color: #f1f5f9;
        }
        .footer-note {
            color: #cbd5e1;
        }
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f4c81 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(2,132,199,0.3);
    }

    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.2s;
    }

    div[data-testid="stFileUploader"] {
        background: #f8fafc;
        border-radius: 12px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #64748b;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7;
        border-bottom-color: #0284c7;
    }

    .rec-card {
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin: 0.6rem 0;
        border-left: 4px solid;
    }

    .rec-card h4 {
        font-size: 0.92rem;
        font-weight: 600;
        margin: 0 0 0.35rem 0;
    }

    .rec-card p {
        font-size: 0.87rem;
        margin: 0;
        line-height: 1.55;
    }

    .rec-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #dc2626;
    }
    .rec-high h4 { color: #991b1b; }
    .rec-high p { color: #7f1d1d; }

    .rec-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }
    .rec-medium h4 { color: #78350f; }
    .rec-medium p { color: #92400e; }

    .rec-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left-color: #16a34a;
    }
    .rec-low h4 { color: #14532d; }
    .rec-low p { color: #166534; }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(128, 128, 128, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_ai_sheet_recommendation(sheets, api_key):
    """Ask the AI which sheet is most likely the primary data table to analyse.
    Returns a sheet name from `sheets`, or None if unavailable/inconclusive."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        previews = {}
        for name, sheet_df in sheets.items():
            previews[name] = {
                "shape": list(sheet_df.shape),
                "columns": [str(c) for c in sheet_df.columns[:10]],
                "sample_rows": sheet_df.head(3).astype(str).to_dict('records'),
            }

        prompt = f"""Multiple sheets were found in an uploaded Excel file. Identify which ONE sheet contains
the primary tabular business data to analyse (not instructions, notes, cover pages, or metadata).

SHEETS:
{json.dumps(previews, indent=2, default=str)}

Respond with ONLY the exact sheet name, nothing else - no punctuation, no explanation."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        log_api_call("sheet_classification", "claude-sonnet-4-6", response)
        suggested = response.content[0].text.strip()
        return suggested if suggested in sheets else None
    except Exception as e:
        log_event("error", f"get_ai_sheet_recommendation: {e}")
        return None


def get_column_labels(df, api_key):
    """Return a dict mapping each real column to a human-readable display label.
    Uses the AI model when a key is available, otherwise falls back to a heuristic."""
    cols = [c for c in df.columns if not str(c).startswith('_')]
    fallback = {c: humanize_column_name(c) for c in cols}

    if not api_key:
        return fallback

    try:
        client = anthropic.Anthropic(api_key=api_key)
        samples = {}
        for c in cols:
            vals = df[c].dropna().unique()[:3].tolist()
            samples[c] = [str(v) for v in vals]

        prompt = f"""You are relabelling spreadsheet column names for a business dashboard.
For each column below, provide a short, human-readable display label (2-4 words, Title Case, no abbreviations, no units unless essential).

COLUMNS AND SAMPLE VALUES:
{json.dumps(samples, indent=2)}

Respond with ONLY a JSON object mapping each original column name to its display label. No other text, no markdown code fences."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        log_api_call("column_labels", "claude-sonnet-4-6", response)
        text = response.content[0].text.strip()
        text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
        labels = json.loads(text)
        return {c: labels.get(c, fallback[c]) for c in cols}
    except Exception as e:
        log_event("error", f"get_column_labels: {e}")
        return fallback


def get_advisor_recommendations(df, clean_report, anomaly_summary, quality_scores, api_key, industry=None):
    """Return a list of {title, severity, category, recommendation} findings.
    Uses the AI model when a key is available, otherwise falls back to rule-based checks.
    If `industry` is given, the AI-generated findings are framed using that industry's context."""
    if not api_key:
        return get_heuristic_recommendations(clean_report, quality_scores)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]

        stats_dict = {}
        for col in num_cols[:6]:
            col_data = df[col].dropna()
            stats_dict[col] = {
                "mean": round(float(col_data.mean()), 2),
                "min": round(float(col_data.min()), 2),
                "max": round(float(col_data.max()), 2),
            }

        profile = {
            "rows": clean_report['cleaned_rows'],
            "columns": clean_report['original_cols'],
            "data_quality": quality_scores,
            "anomalies": anomaly_summary if anomaly_summary else {},
            "stats": stats_dict,
        }

        industry_line = f"\n{INDUSTRY_AI_CONTEXT[industry]}\n" if industry in INDUSTRY_AI_CONTEXT else ""

        prompt = f"""You are a business data advisor. Review this dataset profile and identify 3-6 specific, actionable findings a business owner should know about.
{industry_line}
DATA PROFILE:
{json.dumps(profile, indent=2)}

Assess each finding's severity based on how urgently it needs attention.

Respond with ONLY a JSON array, no other text, no markdown code fences, in this exact format:
[
  {{"title": "short title", "severity": "High", "category": "Data Quality", "recommendation": "one specific, actionable sentence"}}
]

Valid severity values: "High", "Medium", "Low". Valid category values: "Data Quality", "Business Insight", "Anomaly"."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        log_api_call("advisor", "claude-sonnet-4-6", response)
        text = response.content[0].text.strip()
        text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
        recs = json.loads(text)
        if isinstance(recs, list) and recs:
            return recs
        return get_heuristic_recommendations(clean_report, quality_scores)
    except Exception as e:
        log_event("error", f"get_advisor_recommendations: {e}")
        return get_heuristic_recommendations(clean_report, quality_scores)


def get_ai_summary(df, clean_report, anomaly_summary, api_key, goal=None, industry=None):
    """Call the AI model to generate a plain-English executive summary.
    If `goal` is given, the summary is steered toward that specific question/focus.
    If `industry` is given, the summary is framed using that industry's context."""
    client = anthropic.Anthropic(api_key=api_key)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'boolean', 'bool']).columns.tolist()

    # Build a compact data profile
    profile = {
        "rows": clean_report['cleaned_rows'],
        "columns": clean_report['original_cols'],
        "duplicates_removed": clean_report['duplicates_removed'],
        "missing_filled": clean_report['total_missing'],
        "numeric_columns": num_cols[:8],
        "categorical_columns": cat_cols[:6],
        "anomalies": anomaly_summary if anomaly_summary else {},
    }

    # Sample stats
    stats_dict = {}
    for col in num_cols[:6]:
        col_data = df[col].dropna()
        stats_dict[col] = {
            "mean": round(float(col_data.mean()), 2),
            "min": round(float(col_data.min()), 2),
            "max": round(float(col_data.max()), 2),
            "std": round(float(col_data.std()), 2)
        }

    goal_line = f"\nTHE USER'S SPECIFIC GOAL FOR THIS ANALYSIS: {goal}\nGive extra weight to this in your findings and recommendations.\n" if goal else ""
    industry_line = f"\nINDUSTRY CONTEXT: {INDUSTRY_AI_CONTEXT[industry]}\n" if industry in INDUSTRY_AI_CONTEXT else ""

    prompt = f"""You are Ostrivo, an AI business intelligence assistant. A user has uploaded a dataset.
Analyse the following data profile and provide a clear, concise executive summary.
{industry_line}{goal_line}
DATA PROFILE:
{json.dumps(profile, indent=2)}

KEY STATISTICS:
{json.dumps(stats_dict, indent=2)}

Write a structured executive summary with these sections:
1. **Overview** - What this dataset appears to contain (2-3 sentences)
2. **Key Findings** - 3-4 most important insights from the data
3. **Data Quality** - Comment on completeness and any issues found
4. **Anomalies** - What the anomaly detection found and what it might mean
5. **Recommended Actions** - 2-3 specific, actionable recommendations for the business

Be direct, specific, and avoid generic statements. Write as if advising a business owner who is not technical.
Keep the total response under 350 words."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    log_api_call("ai_summary", "claude-sonnet-4-6", response)
    return response.content[0].text


def ask_data_question(df, question, api_key, industry=None):
    """Answer a natural language question about the data.
    If `industry` is given, the answer is framed using that industry's context."""
    client = anthropic.Anthropic(api_key=api_key)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_stats = {}
    for col in num_cols[:8]:
        col_data = df[col].dropna()
        sample_stats[col] = {
            "mean": round(float(col_data.mean()), 2),
            "min": round(float(col_data.min()), 2),
            "max": round(float(col_data.max()), 2),
            "sum": round(float(col_data.sum()), 2),
            "count": int(col_data.count())
        }

    cat_summary = {}
    for col in df.select_dtypes(include=['object', 'boolean', 'bool']).columns[:4]:
        cat_summary[col] = df[col].value_counts().head(5).to_dict()

    context = {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_stats": sample_stats,
        "categorical_counts": cat_summary
    }

    industry_line = f"\n{INDUSTRY_AI_CONTEXT[industry]}\n" if industry in INDUSTRY_AI_CONTEXT else ""

    prompt = f"""You are Ostrivo, an AI data analyst. Answer the following question about a dataset.
{industry_line}
DATASET CONTEXT:
{json.dumps(context, indent=2)}

USER QUESTION: {question}

Provide a clear, direct answer in 2-4 sentences. Be specific with numbers where relevant. 
If the answer cannot be determined from the available data, say so clearly."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    log_api_call("chat_question", "claude-sonnet-4-6", response)
    return response.content[0].text


OSTRIVO_HELP_CONTEXT = """
Ostrivo is an AI-powered business intelligence web app. How it works:

- Upload a CSV or Excel file (.csv, .xlsx, .xls) via the sidebar uploader (max 200MB).
- The app auto-cleans the data: removes duplicate rows, fills missing values (median for numbers,
  mode for categories), and parses date columns.
- Tabs after upload: Auto-Pilot (one-click full analysis), Dashboard (charts), Anomalies (outlier
  detection via Isolation Forest), Advisor (data quality scorecard + recommendations), Forecast
  (trend projection for dated data), AI Summary (AI-written executive summary), Ask Your Data
  (chat about your dataset), Raw Data (download cleaned CSV/Excel/PDF report).
- AI features (summary, advisor, forecast narrative, chat) need an AI API key pasted into the
  sidebar. Get a free key at console.anthropic.com - it's never stored, only sent directly to the
  AI provider for that request.
- Nothing uploaded is stored on the server; it exists only for that browser session.
- Support contact: peterimoniose@live.com or +44 7425 406280.
"""


def get_help_answer(question, api_key):
    """Answer a question about how to use Ostrivo (not about the user's specific data)."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are the help assistant for Ostrivo, a business intelligence web app. Answer the
user's question about HOW TO USE the app, using only the reference info below. If they ask about
their specific uploaded data, tell them to use the "Ask Your Data" tab instead.

REFERENCE INFO:
{OSTRIVO_HELP_CONTEXT}

USER QUESTION: {question}

Answer in 2-4 concise, direct sentences."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    log_api_call("help_assistant", "claude-sonnet-4-6", response)
    return response.content[0].text


# ── Authentication (only active when Supabase secrets are configured) ───────
supabase_client = get_supabase_client(
    st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None,
    st.secrets.get("SUPABASE_ANON_KEY") if hasattr(st, "secrets") else None,
)
cookie_controller = CookieController() if supabase_client else None

if supabase_client:
    # A fresh, unauthenticated Client object is created above on every single script
    # rerun - it must be re-attached to the user's session every time (not just once
    # at login), or later calls like .table().insert() go out with no JWT and get
    # rejected by Row Level Security. Session tokens are cached in st.session_state
    # for this (fast, no cookie round-trip needed most reruns); the cookie is only
    # needed as a fallback on the very first run after a browser reload.
    access_token = st.session_state.get('sb_access_token')
    refresh_token = st.session_state.get('sb_refresh_token')
    if not (access_token and refresh_token):
        access_token = cookie_controller.get('ostrivo_access_token')
        refresh_token = cookie_controller.get('ostrivo_refresh_token')

    if access_token and refresh_token:
        try:
            session_result = restore_session(supabase_client, access_token, refresh_token)
            st.session_state['auth_user'] = session_result.user
            st.session_state['sb_access_token'] = session_result.session.access_token
            st.session_state['sb_refresh_token'] = session_result.session.refresh_token
        except Exception:
            st.session_state.pop('auth_user', None)
            st.session_state.pop('sb_access_token', None)
            st.session_state.pop('sb_refresh_token', None)
            cookie_controller.remove('ostrivo_access_token')
            cookie_controller.remove('ostrivo_refresh_token')

    guest_uses = get_guest_uses()
    guest_active = st.session_state.get('is_guest') and guest_uses < GUEST_ANALYSIS_LIMIT

    if 'auth_user' not in st.session_state and not guest_active:
        st.markdown("""
        <div class="main-header">
            <h1>📊 Ostrivo</h1>
            <p>Log in or create a free account to get started.</p>
        </div>
        """, unsafe_allow_html=True)

        if 'pending_signup_email' in st.session_state:
            render_verify_screen(st.session_state['pending_signup_email'])

        login_tab, signup_tab, guest_tab = st.tabs(["🔑 Log In", "✨ Sign Up", "🚀 Try as Guest"])

        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Log In")
            if login_submitted:
                try:
                    result = sign_in(supabase_client, login_email, login_password)
                    st.session_state['auth_user'] = result.user
                    st.session_state['sb_access_token'] = result.session.access_token
                    st.session_state['sb_refresh_token'] = result.session.refresh_token
                    cookie_controller.set('ostrivo_access_token', result.session.access_token)
                    cookie_controller.set('ostrivo_refresh_token', result.session.refresh_token)
                    # The cookie component needs a render cycle to flush the write to the
                    # browser before we tear down the DOM with rerun() - an immediate rerun
                    # can cut it off, so give it a brief moment first.
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")

        with signup_tab:
            with st.form("signup_form"):
                signup_name = st.text_input("Full Name", key="signup_name")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                st.caption("Must be at least 8 characters, with an uppercase letter, a lowercase letter, and a number.")
                signup_password_confirm = st.text_input(
                    "Re-enter Password", type="password", key="signup_password_confirm"
                )
                signup_industry = st.selectbox(
                    "Which industry best fits your work?", list(INDUSTRY_OPTIONS.keys()),
                    format_func=lambda k: INDUSTRY_OPTIONS[k], key="signup_industry",
                    help="Ostrivo tailors its analysis and recommendations to this. You can change it later."
                )
                signup_submitted = st.form_submit_button("Create Account")
            if signup_submitted:
                is_valid, password_message = validate_password_strength(signup_password)
                if not signup_name.strip():
                    st.error("Please enter your name.")
                elif not is_valid:
                    st.error(password_message)
                elif signup_password != signup_password_confirm:
                    st.error("Passwords don't match.")
                else:
                    try:
                        sign_up(
                            supabase_client, signup_email, signup_password,
                            industry=signup_industry, full_name=signup_name.strip()
                        )
                        st.session_state['pending_signup_email'] = signup_email
                        render_verify_screen(signup_email)
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")

        with guest_tab:
            remaining = GUEST_ANALYSIS_LIMIT - guest_uses
            if remaining > 0:
                st.write(
                    f"Try Ostrivo with your own data, no account needed - "
                    f"**{remaining} free {'analysis' if remaining == 1 else 'analyses'}** on this browser."
                )
                st.caption("Nothing you upload in guest mode is saved anywhere - there's no account "
                           "to save it to, so it's gone the moment you close the tab.")
                if st.button("🚀 Start as Guest", key="guest_start_btn"):
                    st.session_state['is_guest'] = True
                    st.rerun()
            else:
                st.write("You've used all your free guest analyses on this browser.")
                st.caption("Log in or sign up above to keep going - it's free.")

        st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if supabase_client and 'auth_user' in st.session_state:
        display_full_name = (st.session_state['auth_user'].user_metadata or {}).get('full_name')
        st.markdown(f"### 👤 {display_full_name or st.session_state['auth_user'].email}")
        if display_full_name:
            st.caption(st.session_state['auth_user'].email)
        if st.button("Log Out", key="logout_btn"):
            try:
                sign_out(supabase_client)
            except Exception:
                pass
            cookie_controller.remove('ostrivo_access_token')
            cookie_controller.remove('ostrivo_refresh_token')
            st.session_state.pop('auth_user', None)
            st.session_state.pop('sb_access_token', None)
            st.session_state.pop('sb_refresh_token', None)
            st.rerun()

        current_industry = (st.session_state['auth_user'].user_metadata or {}).get('industry')
        with st.expander(f"🏭 Industry: {INDUSTRY_OPTIONS.get(current_industry, 'Not set')}"):
            new_industry = st.selectbox(
                "Change industry focus", list(INDUSTRY_OPTIONS.keys()),
                index=list(INDUSTRY_OPTIONS.keys()).index(current_industry) if current_industry in INDUSTRY_OPTIONS else 0,
                format_func=lambda k: INDUSTRY_OPTIONS[k], key="change_industry_select"
            )
            if st.button("Update", key="update_industry_btn"):
                try:
                    result = update_industry(supabase_client, new_industry)
                    st.session_state['auth_user'] = result.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't update: {e}")

        with st.expander("⚠️ Delete Account"):
            st.caption("Permanently deletes your account and every saved analysis. This can't be undone.")
            if not st.session_state.get('confirming_delete'):
                if st.button("Delete My Account", key="delete_account_btn"):
                    st.session_state['confirming_delete'] = True
                    st.rerun()
            else:
                st.error("Are you sure? This permanently deletes your account and all saved analyses.")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Yes, delete everything", key="confirm_delete_btn"):
                        try:
                            delete_own_account(supabase_client)
                        except Exception as e:
                            st.error(f"Couldn't delete account: {e}")
                        else:
                            # The account is already gone at this point - local cookie
                            # cleanup is best-effort and must never make a successful
                            # deletion look like it failed.
                            try:
                                cookie_controller.remove('ostrivo_access_token')
                                cookie_controller.remove('ostrivo_refresh_token')
                            except Exception:
                                pass
                            st.session_state.clear()
                            st.rerun()
                with dc2:
                    if st.button("Cancel", key="cancel_delete_btn"):
                        st.session_state['confirming_delete'] = False
                        st.rerun()
        st.divider()
    elif supabase_client and st.session_state.get('is_guest'):
        remaining = max(GUEST_ANALYSIS_LIMIT - get_guest_uses(), 0)
        st.markdown("### 🚀 Guest Mode")
        st.caption(f"{remaining} free {'analysis' if remaining == 1 else 'analyses'} left on this browser. "
                   "Nothing you upload is saved.")
        if st.button("Sign Up / Log In", key="guest_to_login_btn"):
            st.session_state.pop('is_guest', None)
            st.rerun()
        st.divider()
        st.markdown("### 🏭 Industry Focus")
        st.session_state.setdefault('guest_industry', list(INDUSTRY_OPTIONS.keys())[0])
        st.session_state['guest_industry'] = st.selectbox(
            "Tailor analysis to an industry", list(INDUSTRY_OPTIONS.keys()),
            index=list(INDUSTRY_OPTIONS.keys()).index(st.session_state['guest_industry']),
            format_func=lambda k: INDUSTRY_OPTIONS[k], key="guest_industry_select"
        )
        st.divider()
    elif not supabase_client:
        st.markdown("### 🏭 Industry Focus")
        st.session_state.setdefault('guest_industry', list(INDUSTRY_OPTIONS.keys())[0])
        st.session_state['guest_industry'] = st.selectbox(
            "Tailor analysis to an industry", list(INDUSTRY_OPTIONS.keys()),
            index=list(INDUSTRY_OPTIONS.keys()).index(st.session_state['guest_industry']),
            format_func=lambda k: INDUSTRY_OPTIONS[k], key="guest_industry_select"
        )
        st.divider()

    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "AI API Key",
        type="password",
        placeholder="Enter your API key",
        help="Required to enable AI-powered summaries and Q&A"
    )
    st.caption("Your key is never stored or sent anywhere except directly to the AI provider.")

    st.divider()
    st.markdown("### 📂 Upload Data")
    uploaded_files = st.file_uploader(
        "CSV or Excel file(s)",
        type=["csv", "xlsx", "xls"],
        help="Max 200MB per file. Select multiple files (e.g. one per month) to analyse them together.",
        accept_multiple_files=True
    )

    if supabase_client and 'auth_user' in st.session_state:
        st.divider()
        st.markdown("### 📁 My Dashboards")
        try:
            saved_list = list_saved_analyses(supabase_client, st.session_state['auth_user'].id)
        except Exception as e:
            saved_list = []
            st.caption(f"Couldn't load saved dashboards: {e}")

        if not saved_list:
            st.caption("No saved dashboards yet. Analyse some data, then save it from the Raw Data tab.")
        for saved_item in saved_list:
            sd1, sd2, sd3 = st.columns([3, 1, 1])
            with sd1:
                st.caption(f"**{saved_item['name']}**  \n{saved_item['row_count']} rows")
            with sd2:
                if st.button("Load", key=f"load_{saved_item['id']}"):
                    st.session_state['loaded_analysis'] = load_analysis(supabase_client, saved_item['id'])
                    st.session_state.pop('logged_upload', None)
                    st.rerun()
            with sd3:
                if st.button("🗑️", key=f"delete_{saved_item['id']}"):
                    delete_analysis(supabase_client, saved_item['id'])
                    st.rerun()

    st.divider()
    st.markdown("### 📖 How it works")
    st.caption("""
1. Upload your data file
2. Ostrivo cleans and profiles it automatically
3. View interactive charts and anomaly detection
4. Get an AI-powered executive summary
5. Ask questions about your data
6. Download the full report
    """)

    st.divider()
    st.markdown("### 🆘 Help Assistant")
    help_question = st.text_input(
        "Ask how to use Ostrivo", key="help_question_input",
        placeholder="e.g. What file types are supported?"
    )
    if help_question and st.button("Ask", key="help_ask_btn"):
        if not api_key:
            st.caption("Add your AI API key above to use the help assistant.")
        else:
            with st.spinner("Thinking..."):
                try:
                    st.session_state['help_answer'] = get_help_answer(help_question, api_key)
                except Exception as e:
                    log_event("error", f"help_assistant: {e}")
                    st.error(f"API error: {e}")
    if 'help_answer' in st.session_state:
        st.caption(st.session_state['help_answer'])

    st.divider()
    st.markdown("### 💬 Need help?")
    st.caption("Questions, feedback, or support - reach out directly:")
    st.markdown("📧 [peterimoniose@live.com](mailto:peterimoniose@live.com)")
    st.markdown("📞 [+44 7425 406280](tel:+447425406280)")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Ostrivo</h1>
    <p>Upload your data. Get instant AI-powered insights, dashboards, and anomaly detection.</p>
</div>
""", unsafe_allow_html=True)

using_loaded_analysis = False
if uploaded_files:
    st.session_state.pop('loaded_analysis', None)  # a fresh upload overrides any previously loaded one
elif 'loaded_analysis' in st.session_state:
    using_loaded_analysis = True

if not uploaded_files and not using_loaded_analysis:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 1</div>
            <div class="metric-value" style="font-size:1.3rem">📤 Upload</div>
            <div class="metric-sub">CSV or Excel file from your business</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 2</div>
            <div class="metric-value" style="font-size:1.3rem">🔍 Analyse</div>
            <div class="metric-sub">Auto cleaning, dashboards & anomaly detection</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 3</div>
            <div class="metric-value" style="font-size:1.3rem">💡 Insights</div>
            <div class="metric-sub">AI executive summary & Q&A chat</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Upload a CSV or Excel file in the sidebar to get started. Select multiple files to "
            "analyse them together (e.g. one file per month)."
            + (" Or load a saved dashboard from the sidebar." if supabase_client and 'auth_user' in st.session_state else ""))
    st.stop()

# ── Load and process data ─────────────────────────────────────────────────────
if using_loaded_analysis:
    loaded = st.session_state['loaded_analysis']
    df = loaded['df']
    clean_report = loaded['clean_report']
    anomaly_summary = loaded['anomaly_summary']
    display_name = loaded['name']
    if loaded.get('ai_summary'):
        st.session_state['ai_summary'] = loaded['ai_summary']
else:
    display_name = uploaded_files[0].name if len(uploaded_files) == 1 else f"{len(uploaded_files)}_files_combined"

    try:
        named_dfs = []       # [(filename, df), ...]
        sheet_choice_notes = []  # info about auto-picked sheets, multi-file mode only

        if len(uploaded_files) == 1:
            f = uploaded_files[0]
            if is_excel_file(f.name):
                sheets_cache_key = f.name
                if st.session_state.get('excel_sheets_cache_key') != sheets_cache_key:
                    sheet_names = get_excel_sheet_names(f)
                    if len(sheet_names) > 1:
                        sheets = {name: load_excel_sheet(f, name) for name in sheet_names}
                        st.session_state['excel_sheets'] = sheets
                        st.session_state['excel_sheet_ranking'] = rank_excel_sheets(sheets)
                    else:
                        st.session_state['excel_sheets'] = None
                    st.session_state['excel_sheets_cache_key'] = sheets_cache_key
                    st.session_state.pop('excel_sheet_ai_suggestion', None)

                sheets = st.session_state.get('excel_sheets')

                if sheets:
                    ranked = st.session_state['excel_sheet_ranking']
                    sheet_options = [name for name, score in ranked]

                    if api_key and 'excel_sheet_ai_suggestion' not in st.session_state:
                        with st.spinner("Working out which sheet has your data..."):
                            st.session_state['excel_sheet_ai_suggestion'] = get_ai_sheet_recommendation(sheets, api_key)
                    ai_suggestion = st.session_state.get('excel_sheet_ai_suggestion')
                    default_sheet = ai_suggestion if ai_suggestion in sheet_options else sheet_options[0]

                    st.markdown('<p class="section-title">Multiple Sheets Detected</p>', unsafe_allow_html=True)
                    st.caption(f"This workbook has {len(sheet_options)} sheets. Ostrivo ranked them by how likely "
                               f"each is to be the real data table (vs. notes, instructions, or a cover page).")
                    selected_sheet = st.selectbox(
                        "Which sheet should Ostrivo analyse?", sheet_options,
                        index=sheet_options.index(default_sheet)
                    )
                    if ai_suggestion:
                        if ai_suggestion == selected_sheet:
                            st.caption(f"🤖 AI agrees: **{ai_suggestion}** looks like the data sheet.")
                        else:
                            st.caption(f"🤖 AI suggested **{ai_suggestion}**, but you've selected **{selected_sheet}**.")
                    named_dfs = [(f.name, sheets[selected_sheet])]
                else:
                    named_dfs = [(f.name, load_data(f))]
            else:
                named_dfs = [(f.name, load_data(f))]

        else:
            # Multiple files: auto-pick the best sheet per file (no per-file manual picker,
            # to keep the UI sane when there are many files) and combine them into one dataset.
            with st.spinner(f"Reading {len(uploaded_files)} files..."):
                for f in uploaded_files:
                    if is_excel_file(f.name):
                        sheet_names = get_excel_sheet_names(f)
                        if len(sheet_names) > 1:
                            sheets = {name: load_excel_sheet(f, name) for name in sheet_names}
                            ranked = rank_excel_sheets(sheets)
                            best_sheet = ranked[0][0]
                            if api_key:
                                ai_pick = get_ai_sheet_recommendation(sheets, api_key)
                                if ai_pick:
                                    best_sheet = ai_pick
                            named_dfs.append((f.name, sheets[best_sheet]))
                            sheet_choice_notes.append(f"{f.name}: used sheet '{best_sheet}'")
                        else:
                            named_dfs.append((f.name, load_data(f)))
                    else:
                        named_dfs.append((f.name, load_data(f)))

        if len(named_dfs) == 1:
            raw_df = named_dfs[0][1]
            combine_summary = None
        else:
            raw_df, combine_summary = combine_dataframes(named_dfs)

        if combine_summary:
            st.markdown('<p class="section-title">Multiple Files Combined</p>', unsafe_allow_html=True)
            st.caption(f"Combined {combine_summary['files_combined']} files into "
                       f"{combine_summary['total_rows']:,} rows total. A 'source_file' column was added "
                       f"so you can filter or break down charts by file.")
            if not combine_summary['columns_matched']:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Columns don't fully match across files</h4>
                    <p>Some files have different columns. Missing values were left blank rather than
                    causing an error - check the Raw Data tab if anything looks off.</p>
                </div>
                """, unsafe_allow_html=True)
            with st.expander("Per-file details"):
                for fname, count in combine_summary['file_row_counts'].items():
                    st.caption(f"- {fname}: {count:,} rows")
                for note in sheet_choice_notes:
                    st.caption(f"- {note}")

        # clean_data/detect_anomalies are expensive (e.g. two full date-parsing passes per
        # column) and re-running them on every widget interaction - not just on new uploads -
        # was spiking memory/CPU on large files. Cache the result and only redo the work when
        # the underlying data actually changes.
        upload_fingerprint = f"{display_name}|{raw_df.shape}|{'|'.join(map(str, raw_df.columns))}"
        cached_result = st.session_state.get('processed_result')
        if st.session_state.get('processed_upload_fingerprint') == upload_fingerprint and cached_result is not None:
            df = cached_result['df']
            clean_report = cached_result['clean_report']
            anomaly_summary = cached_result['anomaly_summary']
        else:
            with st.spinner("Cleaning your data..."):
                df, clean_report = clean_data(raw_df)
                df, anomaly_summary = detect_anomalies(df)
            st.session_state['processed_result'] = {
                'df': df, 'clean_report': clean_report, 'anomaly_summary': anomaly_summary,
            }
            st.session_state['processed_upload_fingerprint'] = upload_fingerprint

            # Deliberately no filename or data content logged here - only anonymous shape/counts.
            upload_detail = f"{clean_report['cleaned_rows']} rows x {clean_report['original_cols']} cols"
            if len(named_dfs) > 1:
                upload_detail += f" ({len(named_dfs)} files)"
            log_event("upload", upload_detail)
            st.session_state['logged_upload'] = display_name
            if st.session_state.get('is_guest'):
                increment_guest_uses()
    except Exception as e:
        log_event("error", "upload failed")
        st.error(f"Error loading file(s): {e}")
        st.stop()

if using_loaded_analysis:
    col_labels = loaded['col_labels']
    quality_scores = loaded['quality_scores']
else:
    # ── Column labels ────────────────────────────────────────────────────────
    labels_cache_key = f"{display_name}_{len(df.columns)}_{bool(api_key)}"
    if st.session_state.get('col_labels_cache_key') != labels_cache_key:
        with st.spinner("Labelling columns..."):
            st.session_state['col_labels'] = get_column_labels(df, api_key)
        st.session_state['col_labels_cache_key'] = labels_cache_key
    col_labels = st.session_state['col_labels']

    # ── Data quality scores ──────────────────────────────────────────────────
    quality_scores = get_data_quality_scores(clean_report, anomaly_summary)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Rows</div>
        <div class="metric-value">{clean_report['cleaned_rows']:,}</div>
        <div class="metric-sub">after cleaning</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Columns</div>
        <div class="metric-value">{clean_report['original_cols']}</div>
        <div class="metric-sub">features detected</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Missing Filled</div>
        <div class="metric-value">{clean_report['total_missing']:,}</div>
        <div class="metric-sub">values imputed</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Duplicates Removed</div>
        <div class="metric-value">{clean_report['duplicates_removed']:,}</div>
        <div class="metric-sub">rows dropped</div>
    </div>""", unsafe_allow_html=True)

with k5:
    anom_count = anomaly_summary.get('total_anomalies', 0) if anomaly_summary else 0
    anom_pct = anomaly_summary.get('anomaly_pct', 0) if anomaly_summary else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Anomalies Found</div>
        <div class="metric-value" style="color:#dc2626">{anom_count:,}</div>
        <div class="metric-sub">{anom_pct}% of rows flagged</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Manual column renaming ───────────────────────────────────────────────────
with st.expander("✏️ Rename columns"):
    st.caption("Override any auto-generated label below. Clear a field to go back to the default.")
    rename_cols = [c for c in df.columns if not str(c).startswith('_')]
    n_per_row = 3
    for i in range(0, len(rename_cols), n_per_row):
        row_cols = rename_cols[i:i + n_per_row]
        ui_cols = st.columns(n_per_row)
        for ui_col, col in zip(ui_cols, row_cols):
            with ui_col:
                st.text_input(
                    f"`{col}`",
                    value=col_labels.get(col, col),
                    key=f"rename_{display_name}_{col}"
                )

for _rc in df.columns:
    if not str(_rc).startswith('_'):
        _override = st.session_state.get(f"rename_{display_name}_{_rc}", "").strip()
        if _override:
            col_labels[_rc] = _override

st.markdown("<br>", unsafe_allow_html=True)

# Populated by render_chart() as each chart in this script run gets drawn - by the time the
# Dashboard Banner tab (declared last) runs, every earlier tab's charts are already in here,
# since Streamlit executes every tab's code every run regardless of which one is visible.
CHART_REGISTRY = {}


def render_chart(fig, key):
    """Render a Plotly chart and register it (keyed by its own title) so it can be picked
    for the Dashboard Banner."""
    label = fig.layout.title.text if fig.layout.title and fig.layout.title.text else key
    CHART_REGISTRY[key] = {'label': label, 'fig': fig}
    st.plotly_chart(fig, use_container_width=True, key=key)


def draw_plotly_as_matplotlib(fig, ax):
    """Redraw a simplified static version of a Plotly figure onto matplotlib axes - used only
    for the Dashboard Banner's downloadable image. Plotly's own charts are interactive and
    can't be reliably combined into one static image in this deployment (Kaleido, the usual
    tool for that, needs either a deprecated legacy version or a runtime Chrome download that
    isn't reliable on Streamlit Community Cloud's free tier), so this covers the handful of
    trace types actually used across the app (bar, scatter/line, pie, histogram, heatmap, box,
    violin) generically rather than needing a bespoke renderer per chart."""
    for trace in fig.data:
        ttype = trace.type
        if ttype == 'bar':
            x = [str(v) for v in trace.x] if trace.x is not None else []
            y = list(trace.y) if trace.y is not None else []
            if getattr(trace, 'orientation', None) == 'h':
                ax.barh(x, y, color='#3b82f6')
            else:
                ax.bar(x, y, color='#3b82f6')
                ax.tick_params(axis='x', rotation=45)
        elif ttype == 'scatter':
            x = list(trace.x) if trace.x is not None else []
            y = list(trace.y) if trace.y is not None else []
            mode = trace.mode or ''
            name = trace.name or ''
            color = '#dc2626' if 'ut of' in name.lower() else '#3b82f6'
            if 'markers' in mode and 'lines' not in mode:
                sizes = 20
                marker_sizes = getattr(trace.marker, 'size', None) if trace.marker else None
                if marker_sizes is not None and not isinstance(marker_sizes, (int, float)):
                    raw_sizes = np.array(marker_sizes, dtype=float)
                    if raw_sizes.size and raw_sizes.max() > 0:
                        sizes = 15 + (raw_sizes / raw_sizes.max()) * 60
                ax.scatter(x, y, color=color, s=sizes)
            else:
                ax.plot(x, y, color=color, linewidth=1.5)
        elif ttype == 'violin':
            y = list(trace.y) if trace.y is not None else []
            if y:
                ax.violinplot(y, showmedians=True)
                ax.set_xticks([])
        elif ttype == 'pie':
            values = list(trace.values) if trace.values is not None else []
            labels = [str(v) for v in trace.labels] if trace.labels is not None else None
            if values:
                ax.pie(values, labels=labels, autopct='%1.0f%%', textprops={'fontsize': 7})
        elif ttype == 'histogram':
            x = list(trace.x) if trace.x is not None else []
            if x:
                ax.hist(x, bins=25, color='#3b82f6')
        elif ttype == 'heatmap':
            if trace.z is not None:
                ax.imshow(trace.z, cmap='Blues', aspect='auto')
                if trace.x is not None:
                    ax.set_xticks(range(len(trace.x)))
                    ax.set_xticklabels(trace.x, rotation=45, ha='right', fontsize=6)
                if trace.y is not None:
                    ax.set_yticks(range(len(trace.y)))
                    ax.set_yticklabels(trace.y, fontsize=6)
        elif ttype == 'box':
            y = list(trace.y) if trace.y is not None else []
            if y:
                ax.boxplot(y)

    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    ax.set_title(title, fontsize=9, color='#e7ebf3')
    ax.tick_params(labelsize=6, colors='#94a3b8')
    ax.set_facecolor('#161d2e')
    for spine in ax.spines.values():
        spine.set_color('#232b3d')


def render_forecast_chart(df, date_col, metric_col, col_labels, chart_key, periods=30):
    """Shared trend + seasonality forecast chart, reused by the Forecast tab and any
    industry's Predictive Insights section that offers a metric forecast."""
    fc_df, fc_meta = generate_forecast(df, date_col, metric_col, periods=periods)
    if fc_df is None:
        st.warning("Not enough data points to build a reliable forecast (at least 5 dated rows are needed).")
        return

    metric_label = col_labels.get(metric_col, metric_col)
    actual = fc_df[fc_df['type'] == 'Actual']
    future = fc_df[fc_df['type'] == 'Forecast']

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=future[date_col], y=future['upper'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig_fc.add_trace(go.Scatter(
        x=future[date_col], y=future['lower'], mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(2,132,199,0.15)',
        name='Confidence range', hoverinfo='skip'
    ))
    fig_fc.add_trace(go.Scatter(
        x=actual[date_col], y=actual[metric_col], mode='lines',
        name='Actual', line=dict(color='#64748b', width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=future[date_col], y=future[metric_col], mode='lines',
        name='Forecast', line=dict(color='#0284c7', width=2, dash='dash')
    ))
    fig_fc.update_layout(
        title=f"{metric_label} Forecast - Next {periods} Periods",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94a3b8'),
        title_font=dict(size=14, color='#64748b'),
        legend=dict(orientation='h', y=-0.25)
    )
    render_chart(fig_fc, chart_key)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🚀 Auto-Pilot", "📊 Dashboard", "🔍 Anomalies", "🧭 Advisor", "🏭 Industry Insights",
    "📈 Forecast", "🤖 AI Summary", "💬 Ask Your Data", "📋 Raw Data", "🎨 Dashboard Banner"
])

# ── Tab 1: Auto-Pilot ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Auto-Pilot - One-Click Full Analysis</p>', unsafe_allow_html=True)
    st.caption("Tell it what you care about (optional), then run everything at once - "
               "summary, recommendations, and a forecast if your data has dates.")

    autopilot_goal = st.text_input(
        "What do you want to know? (optional)",
        key="autopilot_goal_input",
        placeholder="e.g. Analyse regional sales trends and flag anything concerning"
    )

    if st.button("🚀 Run Auto-Pilot", key="autopilot_run_btn"):
        log_event("autopilot_run")
        with st.spinner("Running full analysis..."):
            result = {'goal': autopilot_goal}

            result['advisor_recs'] = get_advisor_recommendations(
                df, clean_report, anomaly_summary, quality_scores, api_key, industry=get_current_industry()
            )

            if api_key:
                try:
                    result['ai_summary'] = get_ai_summary(
                        df, clean_report, anomaly_summary, api_key, goal=autopilot_goal or None,
                        industry=get_current_industry()
                    )
                except Exception as e:
                    log_event("error", f"autopilot ai_summary: {e}")
                    result['ai_summary'] = None
            else:
                result['ai_summary'] = None

            autopilot_date_col = detect_date_column(df)
            autopilot_num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
            if autopilot_date_col and autopilot_num_cols:
                metric = pick_forecast_metric(autopilot_num_cols, col_labels, autopilot_goal)
                fc_df, fc_meta = generate_forecast(df, autopilot_date_col, metric, periods=30)
                result['forecast'] = (metric, fc_df, fc_meta) if fc_df is not None else None
            else:
                result['forecast'] = None

            st.session_state['autopilot_result'] = result

    if 'autopilot_result' in st.session_state:
        res = st.session_state['autopilot_result']

        st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)
        ap1, ap2, ap3 = st.columns(3)
        ap1.metric("Completeness", f"{quality_scores['completeness']}%")
        ap2.metric("Duplicate Rate", f"{quality_scores['duplicate_rate']}%")
        ap3.metric("Anomaly Rate", f"{quality_scores['anomaly_rate']}%")

        if res['ai_summary']:
            st.markdown('<p class="section-title">Executive Summary</p>', unsafe_allow_html=True)
            st.markdown(res['ai_summary'])
        else:
            st.info("Add your AI API key in the sidebar to include an executive summary here.")

        st.markdown('<p class="section-title">Top Recommendations</p>', unsafe_allow_html=True)
        severity_class_ap = {'High': 'rec-high', 'Medium': 'rec-medium', 'Low': 'rec-low'}
        severity_icon_ap = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        for rec in res['advisor_recs'][:5]:
            sev = rec.get('severity', 'Low')
            st.markdown(f"""
            <div class="rec-card {severity_class_ap.get(sev, 'rec-low')}">
                <h4>{severity_icon_ap.get(sev, '🟢')} {rec.get('title', 'Finding')} - {rec.get('category', '')}</h4>
                <p>{rec.get('recommendation', '')}</p>
            </div>
            """, unsafe_allow_html=True)

        if res['forecast']:
            metric, fc_df, fc_meta = res['forecast']
            st.markdown('<p class="section-title">Forecast</p>', unsafe_allow_html=True)
            metric_label_ap = col_labels.get(metric, metric)
            actual_ap = fc_df[fc_df['type'] == 'Actual']
            future_ap = fc_df[fc_df['type'] == 'Forecast']
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=future_ap[detect_date_column(df)], y=future_ap['upper'],
                                         mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_ap.add_trace(go.Scatter(x=future_ap[detect_date_column(df)], y=future_ap['lower'],
                                         mode='lines', line=dict(width=0), fill='tonexty',
                                         fillcolor='rgba(2,132,199,0.15)', name='Confidence range', hoverinfo='skip'))
            fig_ap.add_trace(go.Scatter(x=actual_ap[detect_date_column(df)], y=actual_ap[metric],
                                         mode='lines', name='Actual', line=dict(color='#64748b', width=2)))
            fig_ap.add_trace(go.Scatter(x=future_ap[detect_date_column(df)], y=future_ap[metric],
                                         mode='lines', name='Forecast', line=dict(color='#0284c7', width=2, dash='dash')))
            fig_ap.update_layout(
                title=f"{metric_label_ap} Forecast - Next 30 Periods",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                legend=dict(orientation='h', y=-0.25)
            )
            render_chart(fig_ap, "chart_autopilot_forecast")
            st.caption(f"{metric_label_ap} shows {'an' if fc_meta['direction'] == 'increasing' else 'a'} "
                       f"{fc_meta['direction']} trend (~{abs(fc_meta['slope']):.2f} per period).")

        st.caption("Want more detail? Explore the Dashboard, Anomalies, Advisor, Forecast, and AI Summary "
                   "tabs individually.")


# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab2:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_clean = [c for c in num_cols if not c.startswith('_')]
    cat_cols = df.select_dtypes(include=['object', 'category', 'boolean', 'bool']).columns.tolist()
    dash_date_col = detect_date_column(df)

    if not num_cols_clean:
        st.warning("No numeric columns found for charting.")
    else:
        st.markdown('<p class="section-title">Distribution Analysis</p>', unsafe_allow_html=True)
        col_select = st.selectbox(
            "Select column to explore", num_cols_clean,
            format_func=lambda c: col_labels.get(c, c)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x=col_select, nbins=40,
                title=f"Distribution of {col_labels.get(col_select, col_select)}",
                labels=col_labels,
                color_discrete_sequence=["#0284c7"]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b'),
                showlegend=False
            )
            render_chart(fig, "chart_histogram")

        with col_b:
            dist_chart_type = st.selectbox("Chart type", ["Box Plot", "Violin Plot"], key="dist_chart_type")
            if dist_chart_type == "Box Plot":
                fig2 = px.box(
                    df, y=col_select,
                    title=f"Box Plot - {col_labels.get(col_select, col_select)}",
                    labels=col_labels,
                    color_discrete_sequence=["#0f4c81"]
                )
                dist_chart_key = "chart_boxplot"
            else:
                fig2 = px.violin(
                    df, y=col_select, box=True, points=False,
                    title=f"Violin Plot - {col_labels.get(col_select, col_select)}",
                    labels=col_labels,
                    color_discrete_sequence=["#0f4c81"]
                )
                dist_chart_key = "chart_violin"
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b')
            )
            render_chart(fig2, dist_chart_key)

        # Trend over time
        if dash_date_col:
            st.markdown('<p class="section-title">Trend Over Time</p>', unsafe_allow_html=True)
            trend_metric = st.selectbox(
                "Metric to trend", num_cols_clean,
                index=num_cols_clean.index(col_select) if col_select in num_cols_clean else 0,
                format_func=lambda c: col_labels.get(c, c), key="dash_trend_metric"
            )
            trend_df = time_trend_analysis(df, dash_date_col, trend_metric)
            fig_trend = px.line(
                trend_df, x='period', y='total',
                title=f"{col_labels.get(trend_metric, trend_metric)} Over Time",
                markers=True,
                color_discrete_sequence=["#0284c7"]
            )
            fig_trend.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b'),
                showlegend=False
            )
            render_chart(fig_trend, "chart_trend_over_time")

        # Correlation heatmap
        if len(num_cols_clean) >= 2:
            st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
            corr_cols = num_cols_clean[:12]
            corr = df[corr_cols].corr().round(2)
            corr = corr.rename(columns=col_labels, index=col_labels)
            fig3 = px.imshow(
                corr,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Feature Correlations",
                text_auto=True
            )
            fig3.update_layout(
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b'),
                height=500
            )
            render_chart(fig3, "chart_corr_heatmap")

        # Scatter plot
        if len(num_cols_clean) >= 2:
            st.markdown('<p class="section-title">Scatter Explorer</p>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_col = st.selectbox("X axis", num_cols_clean, index=0,
                                      format_func=lambda c: col_labels.get(c, c))
            with sc2:
                y_col = st.selectbox("Y axis", num_cols_clean, index=min(1, len(num_cols_clean)-1),
                                      format_func=lambda c: col_labels.get(c, c))
            with sc3:
                size_col = st.selectbox(
                    "Bubble size (optional)", ["None"] + num_cols_clean,
                    format_func=lambda c: "None" if c == "None" else col_labels.get(c, c)
                )

            color_col = None
            if cat_cols:
                color_col = cat_cols[0] if df[cat_cols[0]].nunique() <= 10 else None

            fig4 = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col,
                size=None if size_col == "None" else size_col,
                title=f"{col_labels.get(x_col, x_col)} vs {col_labels.get(y_col, y_col)}",
                labels=col_labels,
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b')
            )
            render_chart(fig4, "chart_scatter")

        # Categorical breakdown
        if cat_cols:
            st.markdown('<p class="section-title">Category Breakdown</p>', unsafe_allow_html=True)
            cb1, cb2 = st.columns(2)
            with cb1:
                cat_sel = st.selectbox("Select categorical column", cat_cols,
                                        format_func=lambda c: col_labels.get(c, c))
            with cb2:
                cat_chart_type = st.selectbox("Chart type", ["Bar Chart", "Pie Chart"], key="cat_chart_type")
            vc = df[cat_sel].value_counts().head(15).reset_index()
            vc.columns = [cat_sel, 'Count']
            if cat_chart_type == "Bar Chart":
                fig5 = px.bar(
                    vc, x=cat_sel, y='Count',
                    title=f"Top values - {col_labels.get(cat_sel, cat_sel)}",
                    labels=col_labels,
                    color='Count',
                    color_continuous_scale='Blues'
                )
                cat_chart_key = "chart_category_breakdown"
            else:
                fig5 = px.pie(
                    vc, names=cat_sel, values='Count', hole=0.45,
                    title=f"Share - {col_labels.get(cat_sel, cat_sel)}",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                cat_chart_key = "chart_category_pie"
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b'),
                showlegend=(cat_chart_type == "Pie Chart")
            )
            render_chart(fig5, cat_chart_key)

        # Descriptive stats table
        st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
        stats_df = compute_stats(df)
        if stats_df is not None:
            st.dataframe(stats_df.rename(columns=col_labels), use_container_width=True, key="table_descriptive_stats")


# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Anomaly Detection</p>', unsafe_allow_html=True)

    if not anomaly_summary:
        st.info("No numeric columns available for anomaly detection.")
    else:
        anom_count = anomaly_summary.get('total_anomalies', 0)
        anom_pct = anomaly_summary.get('anomaly_pct', 0)

        if anom_count > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ {anom_count:,} anomalous rows detected ({anom_pct}% of data)</h4>
                <p>These rows behave significantly differently from the rest of your data.
                They may represent data entry errors, fraud, exceptional events, or genuine outliers worth investigating.
                Columns used: {', '.join(anomaly_summary.get('cols_used', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <h4>✅ No significant anomalies detected</h4>
                <p>Your data appears consistent with no rows behaving significantly differently from the norm.</p>
            </div>
            """, unsafe_allow_html=True)

        num_cols_clean = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]

        if num_cols_clean and '_anomaly' in df.columns:
            # Anomaly scatter
            col_x = st.selectbox("X axis for anomaly view", num_cols_clean, index=0,
                                  format_func=lambda c: col_labels.get(c, c))
            col_y = st.selectbox("Y axis for anomaly view", num_cols_clean,
                                  index=min(1, len(num_cols_clean)-1),
                                  format_func=lambda c: col_labels.get(c, c))

            anomaly_scatter_labels = {**col_labels, '_anomaly': 'Anomaly'}
            fig_a = px.scatter(
                df, x=col_x, y=col_y,
                color='_anomaly',
                color_discrete_map={True: '#dc2626', False: '#0284c7'},
                title="Anomalous vs Normal rows",
                labels=anomaly_scatter_labels,
                opacity=0.7
            )
            fig_a.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b')
            )
            render_chart(fig_a, "chart_anomaly_scatter")

            # Show anomalous rows
            if anom_count > 0:
                st.markdown('<p class="section-title">Anomalous Rows</p>', unsafe_allow_html=True)
                anom_df = df[df['_anomaly'] == True].drop(
                    columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns]
                )
                st.dataframe(anom_df.head(50), use_container_width=True, key="table_anomalous_rows")
                st.caption(f"Showing up to 50 of {anom_count} anomalous rows.")


# ── Tab 4: Advisor ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Data Health &amp; Recommendations</p>', unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completeness</div>
            <div class="metric-value">{quality_scores['completeness']}%</div>
            <div class="metric-sub">of cells had original values</div>
        </div>""", unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duplicate Rate</div>
            <div class="metric-value">{quality_scores['duplicate_rate']}%</div>
            <div class="metric-sub">of rows were duplicates</div>
        </div>""", unsafe_allow_html=True)
    with q3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Anomaly Rate</div>
            <div class="metric-value">{quality_scores['anomaly_rate']}%</div>
            <div class="metric-sub">of rows flagged as unusual</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Recommendations</p>', unsafe_allow_html=True)

    if not api_key:
        st.caption("Showing rule-based checks. Add an AI API key in the sidebar for deeper, data-specific recommendations.")

    if st.button("🧭 Analyse & Recommend", key="gen_advisor"):
        with st.spinner("Reviewing your data..."):
            try:
                st.session_state['advisor_recs'] = get_advisor_recommendations(
                    df, clean_report, anomaly_summary, quality_scores, api_key,
                    industry=get_current_industry()
                )
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    severity_class = {'High': 'rec-high', 'Medium': 'rec-medium', 'Low': 'rec-low'}
    severity_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}

    if 'advisor_recs' in st.session_state:
        for rec in st.session_state['advisor_recs']:
            sev = rec.get('severity', 'Low')
            css_class = severity_class.get(sev, 'rec-low')
            icon = severity_icon.get(sev, '🟢')
            st.markdown(f"""
            <div class="rec-card {css_class}">
                <h4>{icon} {rec.get('title', 'Finding')} - {rec.get('category', '')}</h4>
                <p>{rec.get('recommendation', '')}</p>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 5: Industry Insights ──────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-title">Industry Insights</p>', unsafe_allow_html=True)
    current_industry = get_current_industry()

    if current_industry not in INDUSTRY_OPTIONS:
        st.info("Set your industry focus in the sidebar" +
                (" (or from 'My Dashboards' area after logging in)" if supabase_client else "") +
                " to see analysis tailored to it.")
    else:
        st.caption(f"Tailored for: **{INDUSTRY_OPTIONS[current_industry]}**")
        default_cat, default_num = suggest_category_and_metric_columns(df)
        industry_num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
        industry_cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'boolean', 'bool']).columns if not c.startswith('_')]

        industry_date_col = detect_date_column(df)

        if current_industry == 'sales_retail':
            if not industry_cat_cols or not industry_num_cols:
                st.warning("This analysis needs at least one category column and one numeric column.")
            else:
                ic1, ic2 = st.columns(2)
                with ic1:
                    tp_cat = st.selectbox("Category (e.g. product, region)", industry_cat_cols,
                                           index=industry_cat_cols.index(default_cat) if default_cat in industry_cat_cols else 0,
                                           format_func=lambda c: col_labels.get(c, c), key="tp_cat")
                with ic2:
                    tp_metric = st.selectbox("Metric (e.g. revenue, units sold)", industry_num_cols,
                                              index=industry_num_cols.index(default_num) if default_num in industry_num_cols else 0,
                                              format_func=lambda c: col_labels.get(c, c), key="tp_metric")

                sales_kpi = industry_kpi_summary(df, tp_cat, tp_metric)
                kq1, kq2, kq3, kq4 = st.columns(4)
                kq1.metric(f"Total {col_labels.get(tp_metric, tp_metric)}", f"{sales_kpi['total']:,.0f}")
                kq2.metric("Top Performer", str(sales_kpi['top_category']))
                kq3.metric("Top Share", f"{sales_kpi['top_category_share_pct']}%")
                kq4.metric("Categories", sales_kpi['category_count'])

                top_df = top_performers_analysis(df, tp_cat, tp_metric, top_n=10)
                sc1, sc2 = st.columns(2)
                with sc1:
                    fig_tp = px.bar(
                        top_df, x=tp_cat, y='total',
                        title=f"Top {col_labels.get(tp_cat, tp_cat)} by {col_labels.get(tp_metric, tp_metric)}",
                        labels={tp_cat: col_labels.get(tp_cat, tp_cat), 'total': col_labels.get(tp_metric, tp_metric)},
                        color='total', color_continuous_scale='Blues'
                    )
                    fig_tp.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                        showlegend=False
                    )
                    render_chart(fig_tp, "chart_top_performers")
                with sc2:
                    fig_share = px.pie(
                        top_df, names=tp_cat, values='total', hole=0.45,
                        title=f"Share of {col_labels.get(tp_metric, tp_metric)}"
                    )
                    fig_share.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                    )
                    render_chart(fig_share, "chart_sales_share")

                st.dataframe(top_df.rename(columns={tp_cat: col_labels.get(tp_cat, tp_cat), 'total': col_labels.get(tp_metric, tp_metric),
                                                     'share_pct': 'Share %'}), use_container_width=True)

                if industry_date_col:
                    sales_trend = time_trend_analysis(df, industry_date_col, tp_metric)
                    fig_sales_trend = px.line(
                        sales_trend, x='period', y='total', markers=True,
                        title=f"{col_labels.get(tp_metric, tp_metric)} Over Time",
                        labels={'period': col_labels.get(industry_date_col, industry_date_col), 'total': col_labels.get(tp_metric, tp_metric)}
                    )
                    fig_sales_trend.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                    )
                    render_chart(fig_sales_trend, "chart_sales_trend")

                st.markdown('<p class="section-title">Predictive Insights</p>', unsafe_allow_html=True)
                st.info(
                    "**Segmentation** needs a category column (e.g. product, customer) and at least "
                    "one numeric column (e.g. revenue, units sold) - two or more numeric columns give "
                    "richer, more distinct segments."
                )
                seg_numeric_choices = st.multiselect(
                    "Numeric columns to segment on", industry_num_cols,
                    default=industry_num_cols[:2] if len(industry_num_cols) >= 2 else industry_num_cols[:1],
                    format_func=lambda c: col_labels.get(c, c), key="sales_seg_numeric"
                )
                if df[tp_cat].nunique() < 2:
                    st.caption(f"Need at least 2 distinct values in {col_labels.get(tp_cat, tp_cat)} to segment - "
                               f"found {df[tp_cat].nunique()}.")
                elif not seg_numeric_choices:
                    st.caption("Pick at least one numeric column above to run segmentation.")
                else:
                    seg_result = segment_categories(df, tp_cat, seg_numeric_choices, n_clusters=3)
                    seg_profile = seg_result['profile']
                    fig_seg = px.bar(
                        seg_profile, x='segment', y='category_count',
                        title=f"Segment Sizes ({seg_result['n_clusters']} segments)",
                        labels={'segment': 'Segment', 'category_count': f"Number of {col_labels.get(tp_cat, tp_cat)}"},
                        color='segment', color_continuous_scale='Blues'
                    )
                    fig_seg.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                        showlegend=False
                    )
                    render_chart(fig_seg, "chart_sales_segments")
                    st.caption("Average values per segment - use these to tell segments apart "
                               "(e.g. 'high revenue, low volume').")
                    st.dataframe(
                        seg_profile.rename(columns={c: col_labels.get(c, c) for c in seg_numeric_choices}),
                        use_container_width=True
                    )

                st.info(
                    f"**{col_labels.get(tp_metric, tp_metric)} forecast** needs a date/time column plus "
                    "the numeric metric to project forward - both are already available here."
                )
                if industry_date_col:
                    render_forecast_chart(df, industry_date_col, tp_metric, col_labels, "chart_sales_forecast")
                else:
                    st.caption("No date column detected - add one to unlock a forecast.")

        elif current_industry == 'finance_banking':
            if not industry_cat_cols or not industry_num_cols:
                st.warning("This analysis needs at least one category column and one numeric (amount) column.")
            else:
                ic1, ic2 = st.columns(2)
                with ic1:
                    cr_cat = st.selectbox("Category (e.g. asset, account, holding)", industry_cat_cols,
                                           index=industry_cat_cols.index(default_cat) if default_cat in industry_cat_cols else 0,
                                           format_func=lambda c: col_labels.get(c, c), key="cr_cat")
                with ic2:
                    cr_amount = st.selectbox("Amount column", industry_num_cols,
                                              index=industry_num_cols.index(default_num) if default_num in industry_num_cols else 0,
                                              format_func=lambda c: col_labels.get(c, c), key="cr_amount")
                cr_result = concentration_risk_analysis(df, cr_cat, cr_amount)
                finance_kpi = industry_kpi_summary(df, cr_cat, cr_amount)
                risk_class = {'Low': 'insight-box', 'Moderate': 'warning-box', 'High': 'warning-box'}
                cq1, cq2, cq3, cq4 = st.columns(4)
                cq1.metric(f"Total {col_labels.get(cr_amount, cr_amount)}", f"{finance_kpi['total']:,.0f}")
                cq2.metric("Concentration Index (HHI)", cr_result['hhi'])
                cq3.metric("Risk Level", cr_result['risk_level'])
                cq4.metric("Holdings", finance_kpi['category_count'])
                st.markdown(f"""
                <div class="{risk_class.get(cr_result['risk_level'], 'insight-box')}">
                    <h4>{'⚠️' if cr_result['risk_level'] != 'Low' else '✅'} {cr_result['risk_level']} concentration risk</h4>
                    <p>'{cr_result['top_category']}' accounts for {cr_result['top_category_share_pct']}% of total
                    {col_labels.get(cr_amount, cr_amount)}. The Herfindahl-Hirschman Index (HHI) of {cr_result['hhi']}
                    {'suggests diversification is reasonable.' if cr_result['risk_level'] == 'Low' else 'suggests concentration risk worth reviewing.'}</p>
                </div>
                """, unsafe_allow_html=True)
                fc1, fc2 = st.columns(2)
                with fc1:
                    fig_cr = px.bar(
                        cr_result['breakdown'].head(15), x=cr_cat, y='total',
                        title=f"{col_labels.get(cr_amount, cr_amount)} by {col_labels.get(cr_cat, cr_cat)}",
                        labels={cr_cat: col_labels.get(cr_cat, cr_cat), 'total': col_labels.get(cr_amount, cr_amount)},
                        color='total', color_continuous_scale='Blues'
                    )
                    fig_cr.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                        showlegend=False
                    )
                    render_chart(fig_cr, "chart_concentration_risk")
                with fc2:
                    fig_cr_pie = px.pie(
                        cr_result['breakdown'].head(15), names=cr_cat, values='total', hole=0.45,
                        title=f"Breakdown of {col_labels.get(cr_amount, cr_amount)}"
                    )
                    fig_cr_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                    )
                    render_chart(fig_cr_pie, "chart_finance_share")

                if industry_date_col:
                    finance_trend = time_trend_analysis(df, industry_date_col, cr_amount)
                    fig_finance_trend = px.line(
                        finance_trend, x='period', y='total', markers=True,
                        title=f"{col_labels.get(cr_amount, cr_amount)} Over Time",
                        labels={'period': col_labels.get(industry_date_col, industry_date_col), 'total': col_labels.get(cr_amount, cr_amount)}
                    )
                    fig_finance_trend.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                    )
                    render_chart(fig_finance_trend, "chart_finance_trend")

                st.markdown('<p class="section-title">Predictive Insights</p>', unsafe_allow_html=True)
                st.info(
                    "**Fraud-risk flagging** reuses anomaly detection across all your numeric columns - "
                    "the more numeric columns describing each transaction, the more effective it is."
                )
                if '_anomaly' in df.columns:
                    finance_flagged = df[df['_anomaly']]
                    if finance_flagged.empty:
                        st.caption("No unusual transactions flagged in this dataset.")
                    else:
                        st.warning(
                            f"⚠️ {len(finance_flagged)} transaction(s) "
                            f"({len(finance_flagged) / len(df) * 100:.1f}%) flagged as statistically "
                            "unusual - worth reviewing as potential fraud or data-entry signals, not "
                            "confirmed fraud."
                        )
                        finance_display_cols = [c for c in [cr_cat, cr_amount] if c in finance_flagged.columns]
                        st.dataframe(
                            finance_flagged[finance_display_cols + ['_anomaly_score']].rename(
                                columns={**{c: col_labels.get(c, c) for c in finance_display_cols},
                                         '_anomaly_score': 'Anomaly Score'}
                            ).sort_values('Anomaly Score').head(20),
                            use_container_width=True
                        )
                else:
                    st.caption("Anomaly detection needs at least one numeric column - none available.")

        elif current_industry == 'engineering_manufacturing':
            if not industry_num_cols:
                st.warning("This analysis needs at least one numeric measurement column.")
            else:
                ic1, ic2 = st.columns(2)
                with ic1:
                    cc_metric = st.selectbox("Measurement to control-chart", industry_num_cols,
                                              index=industry_num_cols.index(default_num) if default_num in industry_num_cols else 0,
                                              format_func=lambda c: col_labels.get(c, c), key="cc_metric")
                with ic2:
                    cc_seq = industry_date_col
                    st.caption(f"Ordered by: {col_labels.get(cc_seq, cc_seq)}" if cc_seq else "Ordered by row order (no date column detected)")

                cc_result = control_chart_analysis(df, cc_metric, sequence_col=cc_seq)
                cq1, cq2, cq3, cq4 = st.columns(4)
                cq1.metric("Center Line (Mean)", cc_result['mean'])
                cq2.metric("Control Limits", f"{cc_result['lcl']:.2f} to {cc_result['ucl']:.2f}")
                cq3.metric("Out of Control Points", f"{cc_result['out_of_control_count']} ({cc_result['out_of_control_pct']}%)")
                cq4.metric("Std Deviation", cc_result['std'])

                points_df = cc_result['points']
                x_axis = points_df[cc_seq] if cc_seq else points_df.index

                fig_cc = go.Figure()
                fig_cc.add_trace(go.Scatter(x=x_axis, y=points_df[cc_metric], mode='lines+markers',
                                             name=col_labels.get(cc_metric, cc_metric), line=dict(color='#64748b')))
                fig_cc.add_hline(y=cc_result['mean'], line_dash='solid', line_color='#16a34a', annotation_text='Center')
                fig_cc.add_hline(y=cc_result['ucl'], line_dash='dash', line_color='#dc2626', annotation_text='UCL')
                fig_cc.add_hline(y=cc_result['lcl'], line_dash='dash', line_color='#dc2626', annotation_text='LCL')
                out_points = points_df[~points_df['in_control']]
                if not out_points.empty:
                    out_x = out_points[cc_seq] if cc_seq else out_points.index
                    fig_cc.add_trace(go.Scatter(x=out_x, y=out_points[cc_metric], mode='markers',
                                                 name='Out of control', marker=dict(color='#dc2626', size=10, symbol='x')))
                fig_cc.update_layout(
                    title=f"Process Control Chart - {col_labels.get(cc_metric, cc_metric)}",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                )
                render_chart(fig_cc, "chart_control_chart")
                st.caption("Control limits are mean +/- 3 standard deviations, standard SPC (X-chart) methodology. "
                           "Points outside the red lines may indicate a process control issue worth investigating.")

                fig_hist = px.histogram(
                    points_df, x=cc_metric, nbins=30,
                    title=f"Process Distribution - {col_labels.get(cc_metric, cc_metric)}",
                    labels={cc_metric: col_labels.get(cc_metric, cc_metric)}
                )
                fig_hist.add_vline(x=cc_result['mean'], line_dash='solid', line_color='#16a34a', annotation_text='Mean')
                fig_hist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                    showlegend=False
                )
                render_chart(fig_hist, "chart_measurement_distribution")

                st.markdown('<p class="section-title">Predictive Insights</p>', unsafe_allow_html=True)
                st.info(
                    "**Time-to-limit estimate** projects your control chart's trend forward to estimate "
                    "how many periods until it would breach a control limit, if the trend continues. "
                    "Needs at least 5 data points; works best with a date column so points are in true "
                    "chronological order. This is a lightweight trend estimate, not full predictive "
                    "maintenance - true Remaining Useful Life (RUL) prediction needs run-to-failure "
                    "sensor data most spreadsheets don't have."
                )
                try:
                    ttl_result = estimate_time_to_limit(points_df, cc_metric, cc_result['ucl'], cc_result['lcl'])
                    metric_label_ttl = col_labels.get(cc_metric, cc_metric)
                    if ttl_result['trend'] == 'stable':
                        st.success(f"✅ {metric_label_ttl} is stable - no clear trend toward either control limit.")
                    elif ttl_result['periods_to_breach'] is None:
                        st.info(f"📈 {metric_label_ttl} is {ttl_result['trend']}, but moving away from its "
                                "limits - no breach expected on the current trend.")
                    else:
                        st.warning(
                            f"⚠️ At the current trend, {metric_label_ttl} is projected to breach its "
                            f"{ttl_result['heading_toward']} in approximately "
                            f"**{ttl_result['periods_to_breach']} periods**."
                        )
                except ValueError as e:
                    st.caption(str(e))

        elif current_industry == 'healthcare':
            if not industry_cat_cols or not industry_num_cols:
                st.warning("This analysis needs at least one category column (e.g. department) and one numeric column "
                           "(e.g. patient count, wait time).")
            else:
                ic1, ic2 = st.columns(2)
                with ic1:
                    hc_cat = st.selectbox("Department / Ward / Condition", industry_cat_cols,
                                           index=industry_cat_cols.index(default_cat) if default_cat in industry_cat_cols else 0,
                                           format_func=lambda c: col_labels.get(c, c), key="hc_cat")
                with ic2:
                    hc_metric = st.selectbox("Metric (e.g. patient count, wait time, length of stay)", industry_num_cols,
                                              index=industry_num_cols.index(default_num) if default_num in industry_num_cols else 0,
                                              format_func=lambda c: col_labels.get(c, c), key="hc_metric")

                hc_kpi = industry_kpi_summary(df, hc_cat, hc_metric)
                hq1, hq2, hq3, hq4 = st.columns(4)
                hq1.metric(f"Total {col_labels.get(hc_metric, hc_metric)}", f"{hc_kpi['total']:,.0f}")
                hq2.metric("Busiest", str(hc_kpi['top_category']))
                hq3.metric("Its Share", f"{hc_kpi['top_category_share_pct']}%")
                hq4.metric("Departments", hc_kpi['category_count'])

                hc_df = top_performers_analysis(df, hc_cat, hc_metric, top_n=10)
                fig_hc = px.bar(
                    hc_df, x=hc_cat, y='total',
                    title=f"{col_labels.get(hc_metric, hc_metric)} by {col_labels.get(hc_cat, hc_cat)}",
                    labels={hc_cat: col_labels.get(hc_cat, hc_cat), 'total': col_labels.get(hc_metric, hc_metric)},
                    color='total', color_continuous_scale='Blues'
                )
                fig_hc.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                    showlegend=False
                )
                render_chart(fig_hc, "chart_healthcare_volume")

                if industry_date_col:
                    hc_trend = time_trend_analysis(df, industry_date_col, hc_metric)
                    fig_hc_trend = px.line(
                        hc_trend, x='period', y='total', markers=True,
                        title=f"{col_labels.get(hc_metric, hc_metric)} Over Time",
                        labels={'period': col_labels.get(industry_date_col, industry_date_col), 'total': col_labels.get(hc_metric, hc_metric)}
                    )
                    fig_hc_trend.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                    )
                    render_chart(fig_hc_trend, "chart_healthcare_trend")

                st.markdown('<p class="section-title">Process Monitoring</p>', unsafe_allow_html=True)
                st.caption("Track a continuous metric like wait time or length of stay for out-of-range signals - "
                           "the same control-chart technique used in healthcare quality improvement.")
                hc_cc_metric = st.selectbox("Metric to monitor", industry_num_cols,
                                             index=industry_num_cols.index(hc_metric) if hc_metric in industry_num_cols else 0,
                                             format_func=lambda c: col_labels.get(c, c), key="hc_cc_metric")
                hc_cc_result = control_chart_analysis(df, hc_cc_metric, sequence_col=industry_date_col)
                hcq1, hcq2, hcq3 = st.columns(3)
                hcq1.metric("Center Line (Mean)", hc_cc_result['mean'])
                hcq2.metric("Control Limits", f"{hc_cc_result['lcl']:.2f} to {hc_cc_result['ucl']:.2f}")
                hcq3.metric("Out of Range Points", f"{hc_cc_result['out_of_control_count']} ({hc_cc_result['out_of_control_pct']}%)")

                hc_points_df = hc_cc_result['points']
                hc_x_axis = hc_points_df[industry_date_col] if industry_date_col else hc_points_df.index

                fig_hc_cc = go.Figure()
                fig_hc_cc.add_trace(go.Scatter(x=hc_x_axis, y=hc_points_df[hc_cc_metric], mode='lines+markers',
                                                name=col_labels.get(hc_cc_metric, hc_cc_metric), line=dict(color='#64748b')))
                fig_hc_cc.add_hline(y=hc_cc_result['mean'], line_dash='solid', line_color='#16a34a', annotation_text='Center')
                fig_hc_cc.add_hline(y=hc_cc_result['ucl'], line_dash='dash', line_color='#dc2626', annotation_text='UCL')
                fig_hc_cc.add_hline(y=hc_cc_result['lcl'], line_dash='dash', line_color='#dc2626', annotation_text='LCL')
                hc_out_points = hc_points_df[~hc_points_df['in_control']]
                if not hc_out_points.empty:
                    hc_out_x = hc_out_points[industry_date_col] if industry_date_col else hc_out_points.index
                    fig_hc_cc.add_trace(go.Scatter(x=hc_out_x, y=hc_out_points[hc_cc_metric], mode='markers',
                                                    name='Out of range', marker=dict(color='#dc2626', size=10, symbol='x')))
                fig_hc_cc.update_layout(
                    title=f"{col_labels.get(hc_cc_metric, hc_cc_metric)} - Control Chart",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b')
                )
                render_chart(fig_hc_cc, "chart_healthcare_control")
                st.caption("Control limits are mean +/- 3 standard deviations. Points outside the red lines may "
                           "indicate a capacity or care-quality issue worth investigating - not a clinical diagnosis.")

                st.markdown('<p class="section-title">Predictive Insights</p>', unsafe_allow_html=True)

                st.info(
                    f"**{col_labels.get(hc_metric, hc_metric)} forecast** needs a date/time column plus "
                    "the numeric metric to project forward - both are already available here."
                )
                if industry_date_col:
                    render_forecast_chart(df, industry_date_col, hc_metric, col_labels, "chart_healthcare_forecast")
                else:
                    st.caption("No date column detected - add one to unlock a patient volume forecast.")

                st.info(
                    "**Readmission/outcome risk** needs a column with exactly two outcome values "
                    "(e.g. a 'readmitted' column with yes/no) plus at least one numeric column to "
                    "learn from, and 20+ rows with complete data. This is an operational risk signal "
                    "only - never a clinical diagnosis."
                )
                hc_binary_cat_cols = [c for c in industry_cat_cols if df[c].nunique() == 2]
                if not hc_binary_cat_cols:
                    st.caption("No two-outcome column detected in this dataset (e.g. a yes/no readmission column).")
                else:
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        hc_outcome_col = st.selectbox("Outcome column", hc_binary_cat_cols,
                                                       format_func=lambda c: col_labels.get(c, c), key="hc_outcome_col")
                    with rc2:
                        hc_risk_features = st.multiselect(
                            "Numeric predictors", industry_num_cols,
                            default=industry_num_cols[:3],
                            format_func=lambda c: col_labels.get(c, c), key="hc_risk_features"
                        )
                    if not hc_risk_features:
                        st.caption("Pick at least one numeric predictor above to train the model.")
                    else:
                        try:
                            hc_risk_result = binary_outcome_risk_model(df, hc_outcome_col, hc_risk_features)
                            rq1, rq2 = st.columns(2)
                            rq1.metric("Model Accuracy", f"{hc_risk_result['accuracy'] * 100:.1f}%")
                            rq2.metric("Predicting", str(hc_risk_result['positive_class']))
                            st.caption("Accuracy measured on a held-out 25% test split - a directional "
                                       "signal, not a guarantee.")

                            hc_importance_df = pd.DataFrame(hc_risk_result['feature_importances'])
                            hc_importance_df['feature'] = hc_importance_df['feature'].map(lambda c: col_labels.get(c, c))
                            fig_hc_importance = px.bar(
                                hc_importance_df, x='weight', y='feature', orientation='h',
                                title="What Drives the Risk Score",
                                color='weight', color_continuous_scale='RdBu'
                            )
                            fig_hc_importance.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter', color='#94a3b8'), title_font=dict(size=14, color='#64748b'),
                                showlegend=False
                            )
                            render_chart(fig_hc_importance, "chart_healthcare_risk_importance")

                            hc_scored_rename = {c: col_labels.get(c, c) for c in hc_risk_features}
                            hc_scored_rename[hc_outcome_col] = col_labels.get(hc_outcome_col, hc_outcome_col)
                            hc_scored_rename['risk_score'] = 'Risk Score'
                            st.dataframe(
                                hc_risk_result['scored_data'].head(20).rename(columns=hc_scored_rename),
                                use_container_width=True
                            )
                        except ValueError as e:
                            st.caption(str(e))


# ── Tab 6: Forecast ───────────────────────────────────────────────────────────
with tab6:
    st.markdown('<p class="section-title">Forecast</p>', unsafe_allow_html=True)

    date_col = detect_date_column(df)
    num_cols_fc = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]

    if not date_col:
        st.info("No date column detected in this dataset. Forecasting needs at least one column of dates.")
    elif not num_cols_fc:
        st.info("No numeric columns available to forecast.")
    else:
        fc1, fc2 = st.columns(2)
        with fc1:
            metric_col = st.selectbox(
                "Metric to forecast", num_cols_fc,
                format_func=lambda c: col_labels.get(c, c)
            )
        with fc2:
            periods = st.slider("Periods to forecast ahead", min_value=7, max_value=90, value=30, step=1)

        forecast_df, forecast_meta = generate_forecast(df, date_col, metric_col, periods)

        if forecast_df is None:
            st.warning("Not enough data points to build a reliable forecast (at least 5 dated rows are needed).")
        else:
            metric_label = col_labels.get(metric_col, metric_col)
            actual = forecast_df[forecast_df['type'] == 'Actual']
            future = forecast_df[forecast_df['type'] == 'Forecast']

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=future[date_col], y=future['upper'], mode='lines',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=future[date_col], y=future['lower'], mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor='rgba(2,132,199,0.15)',
                name='Confidence range', hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=actual[date_col], y=actual[metric_col], mode='lines',
                name='Actual', line=dict(color='#64748b', width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=future[date_col], y=future[metric_col], mode='lines',
                name='Forecast', line=dict(color='#0284c7', width=2, dash='dash')
            ))
            fig_fc.update_layout(
                title=f"{metric_label} Forecast - Next {periods} Periods",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                title_font=dict(size=14, color='#64748b'),
                legend=dict(orientation='h', y=-0.25)
            )
            render_chart(fig_fc, "chart_forecast")

            st.markdown(f"""
            <div class="insight-box">
                <h4>📈 Trend</h4>
                <p>{metric_label} shows {'an' if forecast_meta['direction'] == 'increasing' else 'a'} {forecast_meta['direction']} trend, changing by
                approximately {abs(forecast_meta['slope']):.2f} per period.
                The shaded band is an approximate 95% confidence range based on historical variation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption("This is a simple trend + day-of-week seasonality projection, not a guarantee - "
                       "treat it as a directional estimate, not a precise prediction.")


# ── Tab 7: AI Summary ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<p class="section-title">AI-Powered Executive Summary</p>', unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="warning-box">
            <h4>🔑 API Key Required</h4>
            <p>Enter your AI API key in the sidebar to generate an AI-powered executive summary.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🤖 Generate Executive Summary", key="gen_summary"):
            with st.spinner("AI is analysing your data..."):
                try:
                    summary = get_ai_summary(df, clean_report, anomaly_summary, api_key, industry=get_current_industry())
                    st.session_state['ai_summary'] = summary
                except Exception as e:
                    log_event("error", f"get_ai_summary: {e}")
                    st.error(f"API error: {e}")

        if 'ai_summary' in st.session_state:
            st.markdown(st.session_state['ai_summary'])


# ── Tab 8: Chat ───────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<p class="section-title">Ask Questions About Your Data</p>', unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="warning-box">
            <h4>🔑 API Key Required</h4>
            <p>Enter your AI API key in the sidebar to chat with your data.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Suggested questions
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What is the average value of each numeric column?",
            "Which category appears most frequently?",
            "What is the total sum of the main numeric column?",
            "Are there any patterns in the data worth noting?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sugg_{s[:20]}"):
                st.session_state['chat_question'] = s

        question = st.text_input(
            "Ask anything about your data",
            value=st.session_state.get('chat_question', ''),
            placeholder="e.g. What is the average sales value?"
        )

        if question and st.button("Ask", key="ask_btn"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_data_question(df, question, api_key, industry=get_current_industry())
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>💡 Answer</h4>
                        <p>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API error: {e}")


# ── Tab 9: Raw Data ───────────────────────────────────────────────────────────
with tab9:
    st.markdown('<p class="section-title">Cleaned Dataset</p>', unsafe_allow_html=True)
    display_df = df.drop(columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns])
    st.dataframe(display_df, use_container_width=True, height=500, key="table_cleaned_dataset")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        if st.download_button(
            label="⬇️ Download Cleaned Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"ostrivo_cleaned_{display_name.split('.')[0]}.csv",
            mime="text/csv"
        ):
            log_event("csv_export")

    with dl2:
        pdf_advisor_recs = st.session_state.get('advisor_recs') or get_heuristic_recommendations(clean_report, quality_scores)
        pdf_bytes = generate_pdf_report(
            filename=display_name,
            clean_report=clean_report,
            quality_scores=quality_scores,
            anomaly_summary=anomaly_summary,
            advisor_recs=pdf_advisor_recs,
            ai_summary=st.session_state.get('ai_summary'),
        )
        if st.download_button(
            label="⬇️ Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"ostrivo_report_{display_name.split('.')[0]}.pdf",
            mime="application/pdf"
        ):
            log_event("pdf_export")

    with dl3:
        excel_stats_df = compute_stats(df)
        excel_anom_df = None
        if '_anomaly' in df.columns:
            excel_anom_df = df[df['_anomaly'] == True].drop(
                columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns]
            )
        excel_bytes = generate_excel_report(display_df, excel_stats_df, excel_anom_df, col_labels)
        if st.download_button(
            label="⬇️ Download Excel (Power BI-ready)",
            data=excel_bytes,
            file_name=f"ostrivo_{display_name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            log_event("excel_export")
        st.caption("Import via Power BI's Get Data → Excel Workbook, or Get Data → Text/CSV for the CSV export.")

    if supabase_client and 'auth_user' in st.session_state:
        st.markdown('<p class="section-title">Save This Analysis</p>', unsafe_allow_html=True)
        save_name = st.text_input("Name this analysis", value=display_name.split('.')[0], key="save_analysis_name")
        if st.button("💾 Save to My Dashboards", key="save_analysis_btn"):
            try:
                save_analysis(
                    supabase_client, st.session_state['auth_user'].id, save_name, df,
                    clean_report, quality_scores, anomaly_summary,
                    st.session_state.get('ai_summary'), col_labels, display_name,
                )
                st.success(f"Saved as '{save_name}'! Find it in 'My Dashboards' in the sidebar.")
            except Exception as e:
                st.error(f"Couldn't save: {e}")

# ── Tab 10: Dashboard Banner ────────────────────────────────────────────────────
with tab10:
    st.markdown('<p class="section-title">Dashboard Banner</p>', unsafe_allow_html=True)
    st.caption("Pick charts from anywhere in this analysis to build a custom summary view - "
               "preview it here, or download it as one shareable image.")

    if not CHART_REGISTRY:
        st.info("No charts are available yet - upload data and explore the other tabs first, "
                 "then come back here.")
    else:
        banner_keys = list(CHART_REGISTRY.keys())
        banner_selected = st.multiselect(
            "Charts to include",
            banner_keys,
            format_func=lambda k: CHART_REGISTRY[k]['label'],
            key="banner_chart_selection"
        )

        if not banner_selected:
            st.caption("Select one or more charts above to build your banner.")
        else:
            st.markdown('<p class="section-title">Preview</p>', unsafe_allow_html=True)
            banner_cols = st.columns(2)
            for i, bkey in enumerate(banner_selected):
                with banner_cols[i % 2]:
                    st.plotly_chart(CHART_REGISTRY[bkey]['fig'], use_container_width=True, key=f"banner_preview_{bkey}")

            if st.button("🖼️ Generate Downloadable Banner", key="generate_banner_btn"):
                with st.spinner("Building your banner..."):
                    n = len(banner_selected)
                    ncols = 2 if n > 1 else 1
                    nrows = (n + ncols - 1) // ncols
                    fig_mpl, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows))
                    fig_mpl.patch.set_facecolor('#0b0f19')
                    axes_flat = np.atleast_1d(axes).flatten()

                    for i, bkey in enumerate(banner_selected):
                        draw_plotly_as_matplotlib(CHART_REGISTRY[bkey]['fig'], axes_flat[i])
                    for j in range(n, len(axes_flat)):
                        axes_flat[j].axis('off')

                    fig_mpl.suptitle(f"Ostrivo Dashboard - {display_name}", fontsize=14, color='white', y=1.0)
                    fig_mpl.tight_layout()

                    banner_buffer = io.BytesIO()
                    fig_mpl.savefig(banner_buffer, format='png', dpi=150, bbox_inches='tight',
                                     facecolor=fig_mpl.get_facecolor())
                    plt.close(fig_mpl)
                    banner_buffer.seek(0)
                    st.session_state['banner_image_bytes'] = banner_buffer.getvalue()

            if 'banner_image_bytes' in st.session_state:
                st.image(st.session_state['banner_image_bytes'], use_container_width=True)
                st.download_button(
                    "⬇️ Download Banner (PNG)",
                    data=st.session_state['banner_image_bytes'],
                    file_name=f"ostrivo_banner_{display_name.split('.')[0]}.png",
                    mime="image/png",
                    key="download_banner_btn"
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Ostrivo - AI-Powered Business Intelligence &nbsp;·&nbsp;
    Upload your data, unlock your insights
</div>
""", unsafe_allow_html=True)
