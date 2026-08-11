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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ostrivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def load_data(uploaded_file):
    """Load CSV or Excel file into DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")
    return df


def clean_data(df):
    """Clean and profile a DataFrame."""
    original_shape = df.shape
    report = {}

    # Basic info
    report['original_rows'] = original_shape[0]
    report['original_cols'] = original_shape[1]

    # Duplicates
    dupes = df.duplicated().sum()
    report['duplicates_removed'] = int(dupes)
    df = df.drop_duplicates()

    # Missing values
    missing = df.isnull().sum()
    report['missing_by_col'] = missing[missing > 0].to_dict()
    report['total_missing'] = int(missing.sum())

    # Fill numeric missing with median, categorical with mode
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

    # Try to parse date columns
    for col in cat_cols:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
            except Exception:
                pass

    report['cleaned_rows'] = len(df)
    report['numeric_cols'] = list(num_cols)
    report['categorical_cols'] = [c for c in cat_cols if c in df.columns]

    return df, report


def detect_anomalies(df):
    """Run Isolation Forest on numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 1:
        return df, []

    use_cols = num_cols[:10]  # limit to 10 cols for speed
    X = df[use_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    preds = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)

    df = df.copy()
    df['_anomaly'] = (preds == -1)
    df['_anomaly_score'] = scores

    anomaly_summary = {
        'total_anomalies': int((preds == -1).sum()),
        'anomaly_pct': round(float((preds == -1).mean()) * 100, 1),
        'cols_used': use_cols
    }
    return df, anomaly_summary


def compute_stats(df):
    """Compute descriptive stats for numeric columns."""
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ['_anomaly_score'] if c in df.columns], errors='ignore'
    )
    if num_df.empty:
        return None
    return num_df.describe().round(3)


def detect_date_column(df):
    """Return the first datetime-typed column, or None if none exists."""
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return date_cols[0] if date_cols else None


def generate_forecast(df, date_col, value_col, periods=30):
    """Fit a simple linear trend + day-of-week seasonality model and project it forward.
    Returns (combined_df, meta) where combined_df has Actual and Forecast rows, or (None, None)
    if there isn't enough data to fit a model."""
    ts = df[[date_col, value_col]].dropna().copy()
    ts = ts.groupby(date_col, as_index=False)[value_col].sum()
    ts = ts.sort_values(date_col).reset_index(drop=True)

    if len(ts) < 5:
        return None, None

    ts['t'] = np.arange(len(ts))
    slope, intercept = np.polyfit(ts['t'], ts[value_col], 1)
    trend = slope * ts['t'] + intercept

    dow = ts[date_col].dt.dayofweek
    residual = ts[value_col] - trend
    seasonal_by_dow = residual.groupby(dow).mean()
    resid_std = residual.std() if len(residual) > 1 else 0.0

    last_date = ts[date_col].max()
    freq = pd.infer_freq(ts[date_col]) or 'D'
    future_dates = pd.date_range(last_date, periods=periods + 1, freq=freq)[1:]
    future_t = np.arange(len(ts), len(ts) + len(future_dates))
    future_trend = slope * future_t + intercept
    future_seasonal = np.array([seasonal_by_dow.get(d, 0.0) for d in future_dates.dayofweek])
    future_values = future_trend + future_seasonal

    forecast_df = pd.DataFrame({
        date_col: future_dates,
        value_col: future_values,
        'type': 'Forecast',
        'lower': future_values - 1.96 * resid_std,
        'upper': future_values + 1.96 * resid_std,
    })
    history_df = ts[[date_col, value_col]].copy()
    history_df['type'] = 'Actual'
    history_df['lower'] = history_df[value_col]
    history_df['upper'] = history_df[value_col]

    combined = pd.concat([history_df, forecast_df], ignore_index=True)
    direction = 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'flat')
    meta = {'slope': float(slope), 'direction': direction}
    return combined, meta


def humanize_column_name(col):
    """Fallback heuristic: turn a raw column name into a readable label without AI."""
    name = re.sub(r'[_\-]+', ' ', str(col)).strip()
    name = re.sub(r'(?<!^)(?<![A-Z0-9 ])([A-Z])', r' \1', name)
    return name.title()


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
        text = response.content[0].text.strip()
        text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
        labels = json.loads(text)
        return {c: labels.get(c, fallback[c]) for c in cols}
    except Exception:
        return fallback


def get_data_quality_scores(clean_report, anomaly_summary):
    """Compute simple 0-100 data quality scores from the clean and anomaly reports."""
    total_cells = clean_report['original_rows'] * clean_report['original_cols']
    completeness = 100.0
    if total_cells > 0:
        completeness = round(100 - (clean_report['total_missing'] / total_cells * 100), 1)

    dup_rate = 0.0
    if clean_report['original_rows'] > 0:
        dup_rate = round(clean_report['duplicates_removed'] / clean_report['original_rows'] * 100, 1)

    anomaly_rate = anomaly_summary.get('anomaly_pct', 0) if anomaly_summary else 0

    return {
        'completeness': completeness,
        'duplicate_rate': dup_rate,
        'anomaly_rate': anomaly_rate,
    }


def get_heuristic_recommendations(clean_report, quality_scores):
    """Rule-based recommendations, used when no AI API key is available."""
    recs = []

    if quality_scores['completeness'] < 95:
        recs.append({
            'title': 'Missing data detected',
            'severity': 'High' if quality_scores['completeness'] < 80 else 'Medium',
            'category': 'Data Quality',
            'recommendation': f"{clean_report['total_missing']} values were missing and auto-filled. "
                               f"Review data collection to reduce future gaps."
        })

    if quality_scores['duplicate_rate'] > 0:
        recs.append({
            'title': 'Duplicate rows found',
            'severity': 'High' if quality_scores['duplicate_rate'] > 10 else 'Medium',
            'category': 'Data Quality',
            'recommendation': f"{clean_report['duplicates_removed']} duplicate rows were removed "
                               f"({quality_scores['duplicate_rate']}% of the dataset)."
        })

    if quality_scores['anomaly_rate'] > 0:
        recs.append({
            'title': 'Unusual rows flagged',
            'severity': 'Medium',
            'category': 'Anomaly',
            'recommendation': f"{quality_scores['anomaly_rate']}% of rows were flagged as statistical "
                               f"outliers. Check the Anomalies tab to investigate them."
        })

    if not recs:
        recs.append({
            'title': 'Data looks healthy',
            'severity': 'Low',
            'category': 'Data Quality',
            'recommendation': 'No major data quality issues were detected in this dataset.'
        })

    return recs


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
        text = response.content[0].text.strip()
        text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
        recs = json.loads(text)
        if isinstance(recs, list) and recs:
            return recs
        return get_heuristic_recommendations(clean_report, quality_scores)
    except Exception:
        return get_heuristic_recommendations(clean_report, quality_scores)


def _pdf_safe(text):
    """Strip characters the PDF's core font can't encode (emoji, smart quotes, em-dashes)."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    replacements = {'—': '-', '–': '-', '‘': "'", '’': "'", '“': '"', '”': '"', '•': '-'}
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text.encode('latin-1', 'ignore').decode('latin-1')


def generate_pdf_report(filename, clean_report, quality_scores, anomaly_summary, advisor_recs, ai_summary):
    """Build a downloadable PDF summarising the dataset, quality scores, anomalies, and AI findings."""
    pdf = FPDF()
    pdf.add_page()
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    def line(text, size=10, bold=False, color=(50, 50, 50), gap_after=0):
        pdf.set_x(pdf.l_margin)
        pdf.set_font('Helvetica', 'B' if bold else '', size)
        pdf.set_text_color(*color)
        pdf.multi_cell(page_width, 7, _pdf_safe(text))
        if gap_after:
            pdf.ln(gap_after)

    def section_title(text):
        line(text, size=13, bold=True, color=(15, 23, 42))

    line("Ostrivo Data Report", size=20, bold=True, color=(15, 23, 42))
    line(f"Source file: {filename}", size=10, color=(100, 116, 139), gap_after=4)

    section_title("Dataset Overview")
    line(f"Rows: {clean_report['cleaned_rows']:,}")
    line(f"Columns: {clean_report['original_cols']}")
    line(f"Missing values filled: {clean_report['total_missing']:,}")
    line(f"Duplicate rows removed: {clean_report['duplicates_removed']:,}", gap_after=3)

    section_title("Data Quality Scores")
    line(f"Completeness: {quality_scores['completeness']}%")
    line(f"Duplicate rate: {quality_scores['duplicate_rate']}%")
    line(f"Anomaly rate: {quality_scores['anomaly_rate']}%", gap_after=3)

    if anomaly_summary:
        section_title("Anomaly Detection")
        line(f"Anomalous rows: {anomaly_summary.get('total_anomalies', 0)} "
             f"({anomaly_summary.get('anomaly_pct', 0)}% of data)")
        cols_used = ', '.join(anomaly_summary.get('cols_used', []))
        if cols_used:
            line(f"Columns used: {cols_used}")
        pdf.ln(3)

    if advisor_recs:
        section_title("Recommendations")
        for rec in advisor_recs:
            line(f"[{rec.get('severity', '')}] {rec.get('title', '')}", bold=True)
            line(rec.get('recommendation', ''), gap_after=1)
        pdf.ln(2)

    if ai_summary:
        section_title("AI Executive Summary")
        line(ai_summary)

    return bytes(pdf.output())


def get_ai_summary(df, clean_report, anomaly_summary, api_key):
    """Call the AI model to generate a plain-English executive summary."""
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

    prompt = f"""You are Ostrivo, an AI business intelligence assistant. A user has uploaded a dataset.
Analyse the following data profile and provide a clear, concise executive summary.

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
    except Exception as e:
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard", "🔍 Anomalies", "🧭 Advisor", "📈 Forecast",
    "🤖 AI Summary", "💬 Ask Your Data", "📋 Raw Data"
])

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig2, use_container_width=True)

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
            st.plotly_chart(fig3, use_container_width=True)

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
            st.plotly_chart(fig4, use_container_width=True)

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
            st.plotly_chart(fig5, use_container_width=True)

        # Descriptive stats table
        st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
        stats_df = compute_stats(df)
        if stats_df is not None:
            st.dataframe(stats_df.rename(columns=col_labels), use_container_width=True)


# ── Tab 2: Anomalies ──────────────────────────────────────────────────────────
with tab2:
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
            st.plotly_chart(fig_a, use_container_width=True)

            # Show anomalous rows
            if anom_count > 0:
                st.markdown('<p class="section-title">Anomalous Rows</p>', unsafe_allow_html=True)
                anom_df = df[df['_anomaly'] == True].drop(
                    columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns]
                )
                st.dataframe(anom_df.head(50), use_container_width=True)
                st.caption(f"Showing up to 50 of {anom_count} anomalous rows.")


# ── Tab 3: Advisor ────────────────────────────────────────────────────────────
with tab3:
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
with tab4:
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
            st.plotly_chart(fig_fc, use_container_width=True)

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
with tab5:
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
                    st.error(f"API error: {e}")

        if 'ai_summary' in st.session_state:
            st.markdown(st.session_state['ai_summary'])


# ── Tab 6: Chat ───────────────────────────────────────────────────────────────
with tab6:
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
with tab7:
    st.markdown('<p class="section-title">Cleaned Dataset</p>', unsafe_allow_html=True)
    display_df = df.drop(columns=[c for c in ['_anomaly', '_anomaly_score'] if c in df.columns])
    st.dataframe(display_df, use_container_width=True, height=500)

    dl1, dl2 = st.columns(2)

    with dl1:
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"ostrivo_cleaned_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )

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
        st.download_button(
            label="⬇️ Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"ostrivo_report_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Ostrivo — AI-Powered Business Intelligence &nbsp;·&nbsp;
    Upload your data, unlock your insights
</div>
""", unsafe_allow_html=True)
