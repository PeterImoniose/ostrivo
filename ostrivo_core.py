"""Pure data-processing logic for Ostrivo, kept free of Streamlit/network dependencies
so it can be unit tested directly (see tests/test_ostrivo_core.py)."""

import io
import re

import numpy as np
import pandas as pd
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
                parsed = pd.to_datetime(df[col])
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


def pick_forecast_metric(num_cols, col_labels, goal_text):
    """Pick the numeric column whose label best matches the goal text, or the first column."""
    if goal_text:
        goal_lower = goal_text.lower()
        for col in num_cols:
            label_words = re.findall(r'\w+', col_labels.get(col, col).lower())
            if any(word in goal_lower for word in label_words if len(word) > 2):
                return col
    return num_cols[0] if num_cols else None


def humanize_column_name(col):
    """Fallback heuristic: turn a raw column name into a readable label without AI."""
    name = re.sub(r'[_\-]+', ' ', str(col)).strip()
    name = re.sub(r'(?<!^)(?<![A-Z0-9 ])([A-Z])', r' \1', name)
    return name.title()


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


def generate_excel_report(display_df, stats_df, anom_df, col_labels):
    """Build a multi-sheet Excel workbook (cleaned data, stats, anomalies) that imports
    cleanly into Power BI via Get Data -> Excel Workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        display_df.rename(columns=col_labels).to_excel(writer, sheet_name='Cleaned Data', index=False)
        if stats_df is not None:
            stats_df.rename(columns=col_labels).to_excel(writer, sheet_name='Descriptive Stats')
        if anom_df is not None and not anom_df.empty:
            anom_df.rename(columns=col_labels).to_excel(writer, sheet_name='Anomalies', index=False)
    return buffer.getvalue()
