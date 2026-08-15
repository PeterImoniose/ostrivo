"""Pure data-processing logic for Ostrivo, kept free of Streamlit/network dependencies
so it can be unit tested directly (see tests/test_ostrivo_core.py)."""

import io
import re

import numpy as np
import pandas as pd
from fpdf import FPDF
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def json_safe(obj):
    """Recursively convert numpy/pandas scalar types (and NaN) into plain JSON-serializable
    Python types. Needed before sending report dicts (clean_report, quality_scores, etc.) to
    an external JSON API like Supabase, since numpy int64/float64/bool_ aren't natively
    JSON-serializable and NaN isn't valid JSON."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj


def combine_dataframes(named_dfs):
    """Combine multiple (filename, DataFrame) pairs into one DataFrame for analysing together
    (e.g. one file per month). Adds a 'source_file' column and uses an outer-join concat so
    files with slightly different columns don't error out - missing values become NaN.
    Returns (combined_df, summary) where summary reports per-file row counts and whether the
    column sets matched across all files."""
    if not named_dfs:
        raise ValueError("No files to combine")

    all_columns = [frozenset(df.columns) for _, df in named_dfs]
    columns_matched = len(set(all_columns)) == 1

    parts = []
    file_row_counts = {}
    for name, df in named_dfs:
        part = df.copy()
        part['source_file'] = name
        parts.append(part)
        file_row_counts[name] = len(df)

    combined = pd.concat(parts, ignore_index=True, sort=False)

    summary = {
        'files_combined': len(named_dfs),
        'file_row_counts': file_row_counts,
        'columns_matched': columns_matched,
        'total_rows': len(combined),
    }
    return combined, summary


def load_data(uploaded_file):
    """Load CSV or Excel file into DataFrame. For multi-sheet Excel files, loads the first sheet only -
    use get_excel_sheet_names/load_excel_sheet for sheet-aware loading."""
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")
    return df


def is_excel_file(filename):
    """Return True if the filename has an Excel extension."""
    return filename.lower().endswith(('.xlsx', '.xls'))


def get_excel_sheet_names(uploaded_file):
    """Return the list of sheet names in an Excel file without loading all the data."""
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    return xls.sheet_names


def load_excel_sheet(uploaded_file, sheet_name):
    """Load a single named sheet from an Excel file."""
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def score_sheet_as_data(df):
    """Heuristic score (higher = more likely a real tabular data sheet, not notes/instructions/cover pages).
    Rewards larger, well-filled sheets with short, distinct column headers and at least one numeric column."""
    if df is None or df.empty or df.shape[1] == 0:
        return 0.0

    rows, cols = df.shape
    size_score = min(rows, 1000) * min(cols, 20)

    col_names = [str(c) for c in df.columns]
    avg_len = sum(len(c) for c in col_names) / max(len(col_names), 1)
    header_score = 1.0 if avg_len < 40 else 0.4
    unnamed_ratio = sum(1 for c in col_names if c.lower().startswith('unnamed')) / max(len(col_names), 1)
    header_score *= (1 - unnamed_ratio * 0.5)

    total_cells = rows * cols
    fill_ratio = 1 - (df.isnull().sum().sum() / total_cells) if total_cells > 0 else 0

    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    numeric_score = 1.0 if numeric_cols > 0 else 0.6

    return round(size_score * header_score * max(fill_ratio, 0.1) * numeric_score, 2)


def rank_excel_sheets(sheets):
    """Given {sheet_name: DataFrame}, return [(sheet_name, score), ...] ranked most to least
    likely to be the primary data table."""
    scored = [(name, score_sheet_as_data(sheet_df)) for name, sheet_df in sheets.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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

    use_cols = num_cols
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


# ── Industry-specific analysis ────────────────────────────────────────────────

INDUSTRY_OPTIONS = {
    'sales_retail': 'Sales & Retail Business',
    'finance_banking': 'Finance & Banking',
    'engineering_manufacturing': 'Engineering & Manufacturing',
    'healthcare': 'Healthcare',
}


def suggest_category_and_metric_columns(df):
    """Suggest a default categorical column and numeric column for an industry analysis -
    e.g. the categorical column with a reasonable number of unique values, and the first
    numeric column. Returns (category_col, metric_col), either of which may be None."""
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if not str(c).startswith('_')]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).startswith('_')]

    best_cat = None
    for c in cat_cols:
        nunique = df[c].nunique()
        if 2 <= nunique <= 50:
            best_cat = c
            break
    if best_cat is None and cat_cols:
        best_cat = cat_cols[0]

    best_num = num_cols[0] if num_cols else None
    return best_cat, best_num


def top_performers_analysis(df, category_col, metric_col, top_n=10):
    """Sales & Retail: rank a category (product, region, ...) by the total of a numeric metric
    (revenue, units sold, ...). Returns a DataFrame [category_col, 'total', 'share_pct']."""
    if category_col not in df.columns or metric_col not in df.columns:
        raise ValueError("category_col and metric_col must both exist in the DataFrame")

    grouped = df.groupby(category_col)[metric_col].sum().reset_index()
    grouped.columns = [category_col, 'total']
    grouped = grouped.sort_values('total', ascending=False)

    total_sum = grouped['total'].sum()
    grouped['share_pct'] = round(grouped['total'] / total_sum * 100, 1) if total_sum else 0.0

    return grouped.head(top_n).reset_index(drop=True)


def concentration_risk_analysis(df, category_col, amount_col):
    """Finance & Banking: Herfindahl-Hirschman Index of amount_col concentrated across
    category_col - a standard portfolio/risk concentration measure (0-1 scale; <0.15 = low,
    0.15-0.25 = moderate, >0.25 = high concentration risk). Returns a dict with the HHI, a
    risk level label, the top category's share, and a per-category breakdown DataFrame."""
    if category_col not in df.columns or amount_col not in df.columns:
        raise ValueError("category_col and amount_col must both exist in the DataFrame")

    grouped = df.groupby(category_col)[amount_col].sum().reset_index()
    grouped.columns = [category_col, 'total']
    total_sum = grouped['total'].sum()
    grouped['share'] = grouped['total'] / total_sum if total_sum else 0.0

    hhi = float((grouped['share'] ** 2).sum())
    if hhi < 0.15:
        risk_level = 'Low'
    elif hhi < 0.25:
        risk_level = 'Moderate'
    else:
        risk_level = 'High'

    grouped = grouped.sort_values('total', ascending=False).reset_index(drop=True)
    top_share_pct = round(float(grouped['share'].iloc[0]) * 100, 1) if len(grouped) else 0.0

    return {
        'hhi': round(hhi, 4),
        'risk_level': risk_level,
        'top_category': grouped[category_col].iloc[0] if len(grouped) else None,
        'top_category_share_pct': top_share_pct,
        'breakdown': grouped,
    }


def control_chart_analysis(df, metric_col, sequence_col=None):
    """Engineering & Manufacturing: a statistical process control (individuals/X) chart for
    metric_col - center line (mean), upper/lower control limits (mean +/- 3 std), and which
    points fall outside those limits. If sequence_col is given (e.g. a date or index column),
    points are sorted by it first. Returns a dict with the control limits and a DataFrame of
    points with an 'in_control' flag."""
    if metric_col not in df.columns:
        raise ValueError("metric_col must exist in the DataFrame")

    cols = [metric_col] + ([sequence_col] if sequence_col and sequence_col in df.columns else [])
    data = df[cols].dropna().reset_index(drop=True)
    if sequence_col and sequence_col in data.columns:
        data = data.sort_values(sequence_col).reset_index(drop=True)

    mean = float(data[metric_col].mean()) if len(data) else 0.0
    std = float(data[metric_col].std()) if len(data) > 1 else 0.0
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    data = data.copy()
    data['in_control'] = data[metric_col].between(lcl, ucl)
    out_of_control_count = int((~data['in_control']).sum())

    return {
        'mean': round(mean, 4),
        'std': round(std, 4),
        'ucl': round(ucl, 4),
        'lcl': round(lcl, 4),
        'out_of_control_count': out_of_control_count,
        'out_of_control_pct': round(out_of_control_count / len(data) * 100, 1) if len(data) else 0.0,
        'points': data,
    }


def time_trend_analysis(df, date_col, metric_col, freq='D'):
    """Aggregate metric_col over time (summed per period) for a trend chart - reused across
    every industry's Insights tab wherever a date column is available. freq follows pandas
    offset aliases ('D' daily, 'W' weekly, 'M' monthly). Returns a DataFrame [period, total]."""
    if date_col not in df.columns or metric_col not in df.columns:
        raise ValueError("date_col and metric_col must both exist in the DataFrame")

    data = df[[date_col, metric_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])

    trend = data.set_index(date_col).resample(freq)[metric_col].sum().reset_index()
    trend.columns = ['period', 'total']
    return trend


def industry_kpi_summary(df, category_col, metric_col):
    """Headline KPIs for a category+metric industry view (total, category count, top
    category and its share, average per category) - reused across every industry that ranks
    a category by a numeric metric (Sales & Retail, Finance & Banking, Healthcare)."""
    if category_col not in df.columns or metric_col not in df.columns:
        raise ValueError("category_col and metric_col must both exist in the DataFrame")

    grouped = df.groupby(category_col)[metric_col].sum().reset_index()
    grouped.columns = [category_col, 'total']
    total_sum = float(grouped['total'].sum())
    category_count = len(grouped)

    if category_count:
        top_row = grouped.sort_values('total', ascending=False).iloc[0]
        top_category = top_row[category_col]
        top_category_share_pct = round(float(top_row['total']) / total_sum * 100, 1) if total_sum else 0.0
    else:
        top_category = None
        top_category_share_pct = 0.0

    return {
        'total': round(total_sum, 2),
        'category_count': category_count,
        'top_category': top_category,
        'top_category_share_pct': top_category_share_pct,
        'avg_per_category': round(total_sum / category_count, 2) if category_count else 0.0,
    }


def segment_categories(df, category_col, numeric_cols, n_clusters=3):
    """Group category_col entities (products, customers, departments, ...) into behavioural
    segments using KMeans on their aggregated numeric_cols - e.g. 'high revenue, low volume'
    vs 'high volume, low revenue' product groups. Needs at least 2 distinct categories; uses
    fewer than n_clusters if there aren't enough categories to support it. Returns a dict with
    the per-category segment assignments and a per-segment profile (average values) so
    segments can be labelled in plain English."""
    if category_col not in df.columns:
        raise ValueError("category_col must exist in the DataFrame")
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"numeric_cols not found in DataFrame: {missing}")
    if not numeric_cols:
        raise ValueError("At least one numeric column is required")

    grouped = df.groupby(category_col)[numeric_cols].sum().reset_index()
    if len(grouped) < 2:
        raise ValueError("Need at least 2 distinct categories to segment")

    effective_k = min(n_clusters, len(grouped))
    scaled = StandardScaler().fit_transform(grouped[numeric_cols].values)

    km = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    grouped = grouped.copy()
    grouped['segment'] = km.fit_predict(scaled)

    profile = grouped.groupby('segment')[numeric_cols].mean().reset_index()
    profile['category_count'] = grouped.groupby('segment')[category_col].count().values

    return {
        'segments': grouped,
        'profile': profile,
        'n_clusters': effective_k,
    }


def estimate_time_to_limit(points_df, metric_col, ucl, lcl):
    """Fit a simple linear trend to metric_col over its row sequence and estimate how many
    periods until it would breach the nearest control limit, if the trend continues - a
    lightweight, honest form of predictive maintenance from trend extrapolation. This is not
    a substitute for true Remaining Useful Life (RUL) prediction, which needs run-to-failure
    sensor data most uploads won't have. Needs at least 5 data points. Returns a dict
    describing the trend direction and, only if genuinely heading toward a limit, the
    estimated number of periods until breach (None otherwise)."""
    if metric_col not in points_df.columns:
        raise ValueError("metric_col must exist in the DataFrame")

    values = points_df[metric_col].dropna().reset_index(drop=True)
    if len(values) < 5:
        raise ValueError("Need at least 5 data points to estimate a trend")

    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    current_value = float(values.iloc[-1])

    if abs(slope) < 1e-9:
        return {
            'slope': 0.0,
            'trend': 'stable',
            'heading_toward': None,
            'periods_to_breach': None,
            'current_value': round(current_value, 4),
        }

    if slope > 0:
        target_limit, heading_toward = ucl, 'upper control limit'
    else:
        target_limit, heading_toward = lcl, 'lower control limit'

    periods_to_breach = (target_limit - current_value) / slope
    if periods_to_breach < 0:
        periods_to_breach = None

    return {
        'slope': round(float(slope), 6),
        'trend': 'increasing' if slope > 0 else 'decreasing',
        'heading_toward': heading_toward,
        'periods_to_breach': round(float(periods_to_breach), 1) if periods_to_breach is not None else None,
        'current_value': round(current_value, 4),
    }


def binary_outcome_risk_model(df, outcome_col, feature_cols):
    """Trains a simple logistic regression to estimate the probability of a binary outcome
    (e.g. readmitted yes/no, churned yes/no) from numeric feature columns - a directional risk
    signal, not a clinical or otherwise authoritative prediction. Needs outcome_col to have
    exactly 2 distinct values and at least 20 complete rows. Returns accuracy on a held-out
    test split, feature importances ranked by influence, and a copy of the data with a
    'risk_score' column (0-1 probability of the outcome) added, sorted highest-risk first."""
    if outcome_col not in df.columns:
        raise ValueError("outcome_col must exist in the DataFrame")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature_cols not found in DataFrame: {missing}")
    if not feature_cols:
        raise ValueError("At least one feature column is required")

    data = df[[outcome_col] + feature_cols].dropna().reset_index(drop=True)
    outcome_values = data[outcome_col].unique()
    if len(outcome_values) != 2:
        raise ValueError("outcome_col must have exactly 2 distinct values")
    if len(data) < 20:
        raise ValueError("Need at least 20 complete rows to train a meaningful model")

    positive_class = sorted(outcome_values, key=str)[-1]
    y = (data[outcome_col] == positive_class).astype(int)
    X_scaled = StandardScaler().fit_transform(data[feature_cols].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    importances = sorted(
        zip(feature_cols, model.coef_[0]), key=lambda pair: abs(pair[1]), reverse=True
    )

    data = data.copy()
    data['risk_score'] = model.predict_proba(X_scaled)[:, 1]

    return {
        'accuracy': round(accuracy, 3),
        'positive_class': positive_class,
        'feature_importances': [{'feature': f, 'weight': round(float(w), 4)} for f, w in importances],
        'scored_data': data.sort_values('risk_score', ascending=False).reset_index(drop=True),
    }


def validate_password_strength(password):
    """Enforce signup password rules: at least 8 characters, one uppercase letter, one
    lowercase letter, and one digit. Returns (is_valid, message) - message is empty on success,
    otherwise the first rule that failed."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r'[A-Z]', password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r'\d', password):
        return False, "Password must include at least one number."
    return True, ""
