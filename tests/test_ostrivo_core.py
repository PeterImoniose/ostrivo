"""Unit tests for ostrivo_core.py - the Streamlit-independent data processing logic."""

import io
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ostrivo_core import (
    load_data, clean_data, detect_anomalies, compute_stats, detect_date_column,
    generate_forecast, pick_forecast_metric, humanize_column_name,
    get_data_quality_scores, get_heuristic_recommendations,
    _pdf_safe, generate_pdf_report, generate_excel_report,
    is_excel_file, get_excel_sheet_names, load_excel_sheet,
    score_sheet_as_data, rank_excel_sheets, combine_dataframes,
)


class FakeUploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile: a BytesIO with a .name attribute."""
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


# ── humanize_column_name ──────────────────────────────────────────────────────

def test_humanize_column_name_underscores():
    assert humanize_column_name("qty_sld") == "Qty Sld"


def test_humanize_column_name_camel_case():
    assert humanize_column_name("revenueGBP") == "Revenue Gbp"


def test_humanize_column_name_already_readable():
    assert humanize_column_name("Region") == "Region"


# ── load_data ──────────────────────────────────────────────────────────────────

def test_load_data_csv():
    f = FakeUploadedFile(b"a,b\n1,2\n3,4\n", "test.csv")
    df = load_data(f)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_data_xlsx():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(buf, index=False)
    f = FakeUploadedFile(buf.getvalue(), "test.xlsx")
    df = load_data(f)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_data_unsupported_type():
    f = FakeUploadedFile(b"whatever", "test.txt")
    with pytest.raises(ValueError):
        load_data(f)


# ── multi-sheet Excel detection ─────────────────────────────────────────────

def _make_multi_sheet_excel():
    buf = io.BytesIO()
    data_df = pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 5,
        "revenue": [100 + i * 3 for i in range(20)],
        "cost": [50 + i for i in range(20)],
    })
    notes_df = pd.DataFrame({
        "Please read before using this spreadsheet": [
            "This workbook contains confidential sales data.",
            "Contact finance@example.com with questions.",
            "Data last refreshed on 2026-01-01.",
        ]
    })
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        notes_df.to_excel(writer, sheet_name='Read Me First', index=False)
        data_df.to_excel(writer, sheet_name='Sales Data', index=False)
    buf.seek(0)
    return FakeUploadedFile(buf.getvalue(), "multi_sheet.xlsx")


def test_is_excel_file():
    assert is_excel_file("report.xlsx") is True
    assert is_excel_file("report.XLS") is True
    assert is_excel_file("report.csv") is False


def test_get_excel_sheet_names():
    f = _make_multi_sheet_excel()
    names = get_excel_sheet_names(f)
    assert names == ['Read Me First', 'Sales Data']


def test_load_excel_sheet():
    f = _make_multi_sheet_excel()
    df = load_excel_sheet(f, 'Sales Data')
    assert list(df.columns) == ['region', 'revenue', 'cost']
    assert len(df) == 20


def test_score_sheet_as_data_empty_df():
    assert score_sheet_as_data(pd.DataFrame()) == 0.0


def test_score_sheet_as_data_prefers_real_data_over_notes():
    data_df = pd.DataFrame({
        "region": ["North", "South"] * 10,
        "revenue": list(range(20)),
    })
    notes_df = pd.DataFrame({
        "Please read before using this spreadsheet": ["some long instructional text here"] * 2
    })
    assert score_sheet_as_data(data_df) > score_sheet_as_data(notes_df)


def test_rank_excel_sheets_puts_data_sheet_first():
    f = _make_multi_sheet_excel()
    sheets = {name: load_excel_sheet(f, name) for name in get_excel_sheet_names(f)}
    ranked = rank_excel_sheets(sheets)
    assert ranked[0][0] == 'Sales Data'


# ── combine_dataframes (multi-file analysis) ─────────────────────────────────

def test_combine_dataframes_matching_columns():
    jan = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 150]})
    feb = pd.DataFrame({"region": ["North", "South"], "revenue": [110, 160]})
    combined, summary = combine_dataframes([("jan.csv", jan), ("feb.csv", feb)])

    assert len(combined) == 4
    assert summary['files_combined'] == 2
    assert summary['total_rows'] == 4
    assert summary['columns_matched'] is True
    assert summary['file_row_counts'] == {"jan.csv": 2, "feb.csv": 2}
    assert 'source_file' in combined.columns
    assert set(combined['source_file']) == {"jan.csv", "feb.csv"}


def test_combine_dataframes_mismatched_columns_outer_joins():
    jan = pd.DataFrame({"region": ["North"], "revenue": [100]})
    feb = pd.DataFrame({"region": ["South"], "revenue": [150], "cost": [70]})
    combined, summary = combine_dataframes([("jan.csv", jan), ("feb.csv", feb)])

    assert summary['columns_matched'] is False
    assert len(combined) == 2
    assert 'cost' in combined.columns
    # jan's row has no cost value -> NaN, not an error
    assert combined.loc[combined['source_file'] == 'jan.csv', 'cost'].isna().all()


def test_combine_dataframes_single_file():
    jan = pd.DataFrame({"region": ["North"], "revenue": [100]})
    combined, summary = combine_dataframes([("jan.csv", jan)])
    assert len(combined) == 1
    assert summary['files_combined'] == 1
    assert summary['columns_matched'] is True


def test_combine_dataframes_empty_list_raises():
    with pytest.raises(ValueError):
        combine_dataframes([])


# ── clean_data ─────────────────────────────────────────────────────────────────

def test_clean_data_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, report = clean_data(df)
    assert report['duplicates_removed'] == 1
    assert report['cleaned_rows'] == 2


def test_clean_data_fills_missing_values():
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", None, "a"]})
    cleaned, report = clean_data(df)
    assert cleaned["num"].isnull().sum() == 0
    assert cleaned["cat"].isnull().sum() == 0
    assert report['total_missing'] == 2
    # numeric filled with median of [1.0, 3.0] = 2.0
    assert cleaned["num"].iloc[1] == 2.0
    # categorical filled with mode 'a'
    assert cleaned["cat"].iloc[1] == "a"


def test_clean_data_parses_date_column():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"], "val": [1, 2, 3]})
    cleaned, report = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


# ── detect_anomalies ─────────────────────────────────────────────────────────

def test_detect_anomalies_flags_outlier():
    values = [10, 11, 9, 10, 12, 9, 11, 10, 500]  # 500 is a clear outlier
    df = pd.DataFrame({"val": values})
    result_df, summary = detect_anomalies(df)
    assert summary['total_anomalies'] >= 1
    assert result_df.loc[result_df['val'] == 500, '_anomaly'].iloc[0] == True  # noqa: E712


def test_detect_anomalies_no_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    result_df, summary = detect_anomalies(df)
    assert summary == []


# ── compute_stats ────────────────────────────────────────────────────────────

def test_compute_stats_returns_describe_table():
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    stats_df = compute_stats(df)
    assert stats_df is not None
    assert 'mean' in stats_df.index


def test_compute_stats_no_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b"]})
    assert compute_stats(df) is None


# ── detect_date_column ───────────────────────────────────────────────────────

def test_detect_date_column_found():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "val": [1, 2]})
    assert detect_date_column(df) == "date"


def test_detect_date_column_none():
    df = pd.DataFrame({"val": [1, 2]})
    assert detect_date_column(df) is None


# ── generate_forecast ────────────────────────────────────────────────────────

def test_generate_forecast_increasing_trend():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    values = [100 + i * 5 for i in range(20)]
    df = pd.DataFrame({"date": dates, "value": values})
    combined, meta = generate_forecast(df, "date", "value", periods=10)
    assert combined is not None
    assert meta['direction'] == 'increasing'
    assert meta['slope'] == pytest.approx(5, abs=0.5)
    assert (combined['type'] == 'Forecast').sum() == 10
    assert (combined['type'] == 'Actual').sum() == 20


def test_generate_forecast_insufficient_data():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"date": dates, "value": [1, 2, 3]})
    combined, meta = generate_forecast(df, "date", "value")
    assert combined is None
    assert meta is None


# ── pick_forecast_metric ─────────────────────────────────────────────────────

def test_pick_forecast_metric_matches_goal():
    num_cols = ["revenue", "cost"]
    col_labels = {"revenue": "Revenue", "cost": "Cost"}
    assert pick_forecast_metric(num_cols, col_labels, "analyse cost trends") == "cost"


def test_pick_forecast_metric_falls_back_to_first():
    num_cols = ["revenue", "cost"]
    col_labels = {"revenue": "Revenue", "cost": "Cost"}
    assert pick_forecast_metric(num_cols, col_labels, "") == "revenue"
    assert pick_forecast_metric(num_cols, col_labels, None) == "revenue"


# ── get_data_quality_scores ──────────────────────────────────────────────────

def test_get_data_quality_scores_math():
    clean_report = {
        'original_rows': 100, 'original_cols': 5,
        'total_missing': 10, 'duplicates_removed': 5,
    }
    anomaly_summary = {'anomaly_pct': 3.0}
    scores = get_data_quality_scores(clean_report, anomaly_summary)
    assert scores['completeness'] == 98.0  # 100 - (10/500*100)
    assert scores['duplicate_rate'] == 5.0  # 5/100*100
    assert scores['anomaly_rate'] == 3.0


def test_get_data_quality_scores_no_anomaly_summary():
    clean_report = {'original_rows': 10, 'original_cols': 2, 'total_missing': 0, 'duplicates_removed': 0}
    scores = get_data_quality_scores(clean_report, None)
    assert scores['anomaly_rate'] == 0


# ── get_heuristic_recommendations ────────────────────────────────────────────

def test_get_heuristic_recommendations_healthy_data():
    clean_report = {'total_missing': 0, 'duplicates_removed': 0}
    quality_scores = {'completeness': 100.0, 'duplicate_rate': 0.0, 'anomaly_rate': 0.0}
    recs = get_heuristic_recommendations(clean_report, quality_scores)
    assert len(recs) == 1
    assert recs[0]['title'] == 'Data looks healthy'


def test_get_heuristic_recommendations_flags_issues():
    clean_report = {'total_missing': 50, 'duplicates_removed': 20}
    quality_scores = {'completeness': 70.0, 'duplicate_rate': 15.0, 'anomaly_rate': 5.0}
    recs = get_heuristic_recommendations(clean_report, quality_scores)
    titles = [r['title'] for r in recs]
    assert 'Missing data detected' in titles
    assert 'Duplicate rows found' in titles
    assert 'Unusual rows flagged' in titles
    missing_rec = next(r for r in recs if r['title'] == 'Missing data detected')
    assert missing_rec['severity'] == 'High'  # completeness < 80


# ── _pdf_safe ────────────────────────────────────────────────────────────────

def test_pdf_safe_strips_markdown_bold():
    assert _pdf_safe("**Overview**") == "Overview"


def test_pdf_safe_strips_heading():
    assert _pdf_safe("# Title\nBody") == "Title\nBody"


def test_pdf_safe_replaces_smart_punctuation():
    result = _pdf_safe("It’s an em—dash and “quotes”")
    assert "’" not in result
    assert "—" not in result
    assert "'s" in result


def test_pdf_safe_handles_none():
    assert _pdf_safe(None) == ""


# ── generate_pdf_report ──────────────────────────────────────────────────────

def test_generate_pdf_report_returns_valid_pdf_bytes():
    clean_report = {'cleaned_rows': 10, 'original_cols': 3, 'total_missing': 0, 'duplicates_removed': 0}
    quality_scores = {'completeness': 100.0, 'duplicate_rate': 0.0, 'anomaly_rate': 0.0}
    pdf_bytes = generate_pdf_report(
        filename="test.csv", clean_report=clean_report, quality_scores=quality_scores,
        anomaly_summary=None, advisor_recs=None, ai_summary=None,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b'%PDF')


def test_generate_pdf_report_with_all_sections():
    clean_report = {'cleaned_rows': 10, 'original_cols': 3, 'total_missing': 2, 'duplicates_removed': 1}
    quality_scores = {'completeness': 95.0, 'duplicate_rate': 10.0, 'anomaly_rate': 5.0}
    anomaly_summary = {'total_anomalies': 1, 'anomaly_pct': 5.0, 'cols_used': ['revenue']}
    advisor_recs = [{'severity': 'High', 'title': 'Test finding', 'recommendation': 'Do something.'}]
    pdf_bytes = generate_pdf_report(
        filename="test.csv", clean_report=clean_report, quality_scores=quality_scores,
        anomaly_summary=anomaly_summary, advisor_recs=advisor_recs, ai_summary="**Summary** text.",
    )
    assert pdf_bytes.startswith(b'%PDF')


# ── generate_excel_report ────────────────────────────────────────────────────

def test_generate_excel_report_returns_valid_xlsx_bytes():
    display_df = pd.DataFrame({"revenue": [100, 200], "region": ["North", "South"]})
    stats_df = compute_stats(display_df)
    excel_bytes = generate_excel_report(display_df, stats_df, None, col_labels={})
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0
    assert excel_bytes.startswith(b'PK')  # xlsx files are zip archives


def test_generate_excel_report_with_anomalies_sheet():
    display_df = pd.DataFrame({"revenue": [100, 200, 9000]})
    anom_df = display_df[display_df['revenue'] > 1000]
    excel_bytes = generate_excel_report(display_df, None, anom_df, col_labels={})
    assert excel_bytes.startswith(b'PK')
