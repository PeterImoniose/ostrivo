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
from datetime import datetime, timezone
import warnings
from ostrivo_core import (
    load_data, clean_data, detect_anomalies, compute_stats, detect_date_column,
    generate_forecast, pick_forecast_metric, humanize_column_name,
    get_data_quality_scores, get_heuristic_recommendations,
    generate_pdf_report, generate_excel_report,
)
warnings.filterwarnings('ignore')

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


def log_event(event_type, detail=""):
    try:
        conn = sqlite3.connect(ADMIN_DB_PATH)
        conn.execute(
            "INSERT INTO events (ts, session_id, event_type, detail) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), get_session_id(), event_type, str(detail)[:500])
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
    conn.close()
    return {'by_type': by_type, 'sessions': sessions, 'errors': errors}


def get_api_cost_estimate():
    """Rough cost estimate from token counts. Verify actual pricing at anthropic.com/pricing
    and real spend at console.anthropic.com — this is not authoritative billing data."""
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
        "SELECT ts, session_id, event_type, detail FROM events ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return events_df


def admin_chat_query(question, api_key):
    """Answer an admin's natural-language question about aggregated app activity (no customer data)."""
    context = {
        'usage_stats': get_usage_stats(),
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
    st.caption("Aggregated app activity only — no customer-uploaded data is stored or shown here.")

    admin_password_input = st.text_input("Admin password", type="password", key="admin_password_input")

    if not admin_password_input:
        st.info("Enter the admin password to continue.")
        st.stop()

    try:
        correct_password = st.secrets.get("admin_password")
    except Exception:
        correct_password = None

    if not correct_password:
        st.error("No admin password is configured. Set 'admin_password' in Streamlit secrets.")
        st.stop()

    if admin_password_input != correct_password:
        st.error("Incorrect password.")
        st.stop()

    stats = get_usage_stats()
    cost = get_api_cost_estimate()

    c1, c2, c3 = st.columns(3)
    c1.metric("Distinct Sessions", stats['sessions'])
    c2.metric("Total Events", sum(stats['by_type'].values()))
    c3.metric("Errors Logged", stats['errors'])

    st.subheader("Events by Type")
    if stats['by_type']:
        st.bar_chart(stats['by_type'], key="chart_admin_events_by_type")
    else:
        st.caption("No activity logged yet.")

    st.subheader("Estimated AI API Cost")
    st.caption("Rough estimate from token counts — verify actual spend at console.anthropic.com.")
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
        border-bottom: 2px solid #e2e8f0;
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
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

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


def get_advisor_recommendations(df, clean_report, anomaly_summary, quality_scores, api_key):
    """Return a list of {title, severity, category, recommendation} findings.
    Uses the AI model when a key is available, otherwise falls back to rule-based checks."""
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

        prompt = f"""You are a business data advisor. Review this dataset profile and identify 3-6 specific, actionable findings a business owner should know about.

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


def get_ai_summary(df, clean_report, anomaly_summary, api_key, goal=None):
    """Call the AI model to generate a plain-English executive summary.
    If `goal` is given, the summary is steered toward that specific question/focus."""
    client = anthropic.Anthropic(api_key=api_key)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

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

    prompt = f"""You are Ostrivo, an AI business intelligence assistant. A user has uploaded a dataset.
Analyse the following data profile and provide a clear, concise executive summary.
{goal_line}
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


def ask_data_question(df, question, api_key):
    """Answer a natural language question about the data."""
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
    for col in df.select_dtypes(include=['object']).columns[:4]:
        cat_summary[col] = df[col].value_counts().head(5).to_dict()

    context = {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_stats": sample_stats,
        "categorical_counts": cat_summary
    }

    prompt = f"""You are Ostrivo, an AI data analyst. Answer the following question about a dataset.

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
  sidebar. Get a free key at console.anthropic.com — it's never stored, only sent directly to the
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
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
    uploaded_file = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Max 200MB"
    )

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
    st.caption("Questions, feedback, or support — reach out directly:")
    st.markdown("📧 [peterimoniose@live.com](mailto:peterimoniose@live.com)")
    st.markdown("📞 [+44 7425 406280](tel:+447425406280)")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Ostrivo</h1>
    <p>Upload your data. Get instant AI-powered insights, dashboards, and anomaly detection.</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
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

    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

# ── Load and process data ─────────────────────────────────────────────────────
with st.spinner("Loading and cleaning your data..."):
    try:
        raw_df = load_data(uploaded_file)
        df, clean_report = clean_data(raw_df)
        df, anomaly_summary = detect_anomalies(df)
        if st.session_state.get('logged_upload') != uploaded_file.name:
            # Deliberately no filename or data content logged here — only anonymous shape/counts.
            log_event("upload", f"{clean_report['cleaned_rows']} rows x {clean_report['original_cols']} cols")
            st.session_state['logged_upload'] = uploaded_file.name
    except Exception as e:
        log_event("error", "upload failed")
        st.error(f"Error loading file: {e}")
        st.stop()

# ── Column labels ────────────────────────────────────────────────────────────
labels_cache_key = f"{uploaded_file.name}_{len(df.columns)}_{bool(api_key)}"
if st.session_state.get('col_labels_cache_key') != labels_cache_key:
    with st.spinner("Labelling columns..."):
        st.session_state['col_labels'] = get_column_labels(df, api_key)
    st.session_state['col_labels_cache_key'] = labels_cache_key
col_labels = st.session_state['col_labels']

# ── Data quality scores ──────────────────────────────────────────────────────
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🚀 Auto-Pilot", "📊 Dashboard", "🔍 Anomalies", "🧭 Advisor", "📈 Forecast",
    "🤖 AI Summary", "💬 Ask Your Data", "📋 Raw Data"
])

# ── Tab 1: Auto-Pilot ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Auto-Pilot — One-Click Full Analysis</p>', unsafe_allow_html=True)
    st.caption("Tell it what you care about (optional), then run everything at once — "
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
                df, clean_report, anomaly_summary, quality_scores, api_key
            )

            if api_key:
                try:
                    result['ai_summary'] = get_ai_summary(
                        df, clean_report, anomaly_summary, api_key, goal=autopilot_goal or None
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
                <h4>{severity_icon_ap.get(sev, '🟢')} {rec.get('title', 'Finding')} — {rec.get('category', '')}</h4>
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
                                         mode='lines', name='Actual', line=dict(color='#0f172a', width=2)))
            fig_ap.add_trace(go.Scatter(x=future_ap[detect_date_column(df)], y=future_ap[metric],
                                         mode='lines', name='Forecast', line=dict(color='#0284c7', width=2, dash='dash')))
            fig_ap.update_layout(
                title=f"{metric_label_ap} Forecast — Next 30 Periods",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#374151'), title_font=dict(size=14, color='#0f172a'),
                legend=dict(orientation='h', y=-0.25)
            )
            st.plotly_chart(fig_ap, use_container_width=True, key="chart_autopilot_forecast")
            st.caption(f"{metric_label_ap} shows {'an' if fc_meta['direction'] == 'increasing' else 'a'} "
                       f"{fc_meta['direction']} trend (~{abs(fc_meta['slope']):.2f} per period).")

        st.caption("Want more detail? Explore the Dashboard, Anomalies, Advisor, Forecast, and AI Summary "
                   "tabs individually.")


# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab2:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_clean = [c for c in num_cols if not c.startswith('_')]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

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
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_histogram")

        with col_b:
            fig2 = px.box(
                df, y=col_select,
                title=f"Box Plot — {col_labels.get(col_select, col_select)}",
                labels=col_labels,
                color_discrete_sequence=["#0f4c81"]
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a')
            )
            st.plotly_chart(fig2, use_container_width=True, key="chart_boxplot")

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
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a'),
                height=500
            )
            st.plotly_chart(fig3, use_container_width=True, key="chart_corr_heatmap")

        # Scatter plot
        if len(num_cols_clean) >= 2:
            st.markdown('<p class="section-title">Scatter Explorer</p>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                x_col = st.selectbox("X axis", num_cols_clean, index=0,
                                      format_func=lambda c: col_labels.get(c, c))
            with sc2:
                y_col = st.selectbox("Y axis", num_cols_clean, index=min(1, len(num_cols_clean)-1),
                                      format_func=lambda c: col_labels.get(c, c))

            color_col = None
            if cat_cols:
                color_col = cat_cols[0] if df[cat_cols[0]].nunique() <= 10 else None

            fig4 = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col,
                title=f"{col_labels.get(x_col, x_col)} vs {col_labels.get(y_col, y_col)}",
                labels=col_labels,
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a')
            )
            st.plotly_chart(fig4, use_container_width=True, key="chart_scatter")

        # Categorical breakdown
        if cat_cols:
            st.markdown('<p class="section-title">Category Breakdown</p>', unsafe_allow_html=True)
            cat_sel = st.selectbox("Select categorical column", cat_cols,
                                    format_func=lambda c: col_labels.get(c, c))
            vc = df[cat_sel].value_counts().head(15).reset_index()
            vc.columns = [cat_sel, 'Count']
            fig5 = px.bar(
                vc, x=cat_sel, y='Count',
                title=f"Top values — {col_labels.get(cat_sel, cat_sel)}",
                labels=col_labels,
                color='Count',
                color_continuous_scale='Blues'
            )
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a'),
                showlegend=False
            )
            st.plotly_chart(fig5, use_container_width=True, key="chart_category_breakdown")

        # Descriptive stats table
        st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
        stats_df = compute_stats(df)
        if stats_df is not None:
            st.dataframe(stats_df.rename(columns=col_labels), use_container_width=True, key="table_descriptive_stats")


# ── Tab 2: Anomalies ──────────────────────────────────────────────────────────
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
            col_x = st.selectbox("X axis for anomaly view", num_cols_clean, index=0, key='ax',
                                  format_func=lambda c: col_labels.get(c, c))
            col_y = st.selectbox("Y axis for anomaly view", num_cols_clean,
                                  index=min(1, len(num_cols_clean)-1), key='ay',
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
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a')
            )
            st.plotly_chart(fig_a, use_container_width=True, key="chart_anomaly_scatter")

            # Show anomalous rows
            if anom_count > 0:
                st.markdown('<p class="section-title">Anomalous Rows</p>', unsafe_allow_html=True)
                anom_df = df[df['_anomaly'] == True].drop(
                    columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns]
                )
                st.dataframe(anom_df.head(50), use_container_width=True, key="table_anomalous_rows")
                st.caption(f"Showing up to 50 of {anom_count} anomalous rows.")


# ── Tab 3: Advisor ────────────────────────────────────────────────────────────
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
                    df, clean_report, anomaly_summary, quality_scores, api_key
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
                <h4>{icon} {rec.get('title', 'Finding')} — {rec.get('category', '')}</h4>
                <p>{rec.get('recommendation', '')}</p>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 4: Forecast ───────────────────────────────────────────────────────────
with tab5:
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
                name='Actual', line=dict(color='#0f172a', width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=future[date_col], y=future[metric_col], mode='lines',
                name='Forecast', line=dict(color='#0284c7', width=2, dash='dash')
            ))
            fig_fc.update_layout(
                title=f"{metric_label} Forecast — Next {periods} Periods",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#374151'),
                title_font=dict(size=14, color='#0f172a'),
                legend=dict(orientation='h', y=-0.25)
            )
            st.plotly_chart(fig_fc, use_container_width=True, key="chart_forecast")

            st.markdown(f"""
            <div class="insight-box">
                <h4>📈 Trend</h4>
                <p>{metric_label} shows {'an' if forecast_meta['direction'] == 'increasing' else 'a'} {forecast_meta['direction']} trend, changing by
                approximately {abs(forecast_meta['slope']):.2f} per period.
                The shaded band is an approximate 95% confidence range based on historical variation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption("This is a simple trend + day-of-week seasonality projection, not a guarantee — "
                       "treat it as a directional estimate, not a precise prediction.")


# ── Tab 5: AI Summary ─────────────────────────────────────────────────────────
with tab6:
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
                    summary = get_ai_summary(df, clean_report, anomaly_summary, api_key)
                    st.session_state['ai_summary'] = summary
                except Exception as e:
                    log_event("error", f"get_ai_summary: {e}")
                    st.error(f"API error: {e}")

        if 'ai_summary' in st.session_state:
            st.markdown(st.session_state['ai_summary'])


# ── Tab 6: Chat ───────────────────────────────────────────────────────────────
with tab7:
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
                    answer = ask_data_question(df, question, api_key)
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>💡 Answer</h4>
                        <p>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API error: {e}")


# ── Tab 7: Raw Data ───────────────────────────────────────────────────────────
with tab8:
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
            file_name=f"ostrivo_cleaned_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        ):
            log_event("csv_export")

    with dl2:
        pdf_advisor_recs = st.session_state.get('advisor_recs') or get_heuristic_recommendations(clean_report, quality_scores)
        pdf_bytes = generate_pdf_report(
            filename=uploaded_file.name,
            clean_report=clean_report,
            quality_scores=quality_scores,
            anomaly_summary=anomaly_summary,
            advisor_recs=pdf_advisor_recs,
            ai_summary=st.session_state.get('ai_summary'),
        )
        if st.download_button(
            label="⬇️ Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"ostrivo_report_{uploaded_file.name.split('.')[0]}.pdf",
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
            file_name=f"ostrivo_{uploaded_file.name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            log_event("excel_export")
        st.caption("Import via Power BI's Get Data → Excel Workbook, or Get Data → Text/CSV for the CSV export.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Ostrivo — AI-Powered Business Intelligence &nbsp;·&nbsp;
    Upload your data, unlock your insights
</div>
""", unsafe_allow_html=True)
