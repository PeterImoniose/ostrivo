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
    _pdf_safe, generate_pdf_report, generate_excel_report, _write_df_to_excel,
    is_excel_file, get_excel_sheet_names, load_excel_sheet,
    score_sheet_as_data, rank_excel_sheets, combine_dataframes, json_safe,
    suggest_category_and_metric_columns, top_performers_analysis,
    concentration_risk_analysis, control_chart_analysis,
    validate_password_strength, time_trend_analysis, industry_kpi_summary,
    segment_categories, estimate_time_to_limit, binary_outcome_risk_model,
    convert_column_types, detect_and_convert_numeric_column, detect_and_convert_boolean_column,
    detect_industry, is_rate_like_metric, sample_for_chart, sample_for_chart_preserving_flag,
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


def test_load_data_csv_cp1252_currency_symbol():
    # Byte 0xa3 is the GBP sign in cp1252/latin-1 but is invalid as a UTF-8 start byte -
    # this is what real UK datasets (e.g. online retail invoices) trip on.
    content = "item,price\nWidget,\xa310.50\n".encode("cp1252")
    f = FakeUploadedFile(content, "test.csv")
    df = load_data(f)
    assert list(df.columns) == ["item", "price"]
    assert df.loc[0, "price"] == "\xa310.50"


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


# ── json_safe ────────────────────────────────────────────────────────────────

def test_json_safe_converts_numpy_int():
    assert json_safe(np.int64(5)) == 5
    assert isinstance(json_safe(np.int64(5)), int)


def test_json_safe_converts_numpy_float():
    assert json_safe(np.float64(2.5)) == 2.5
    assert isinstance(json_safe(np.float64(2.5)), float)


def test_json_safe_converts_numpy_nan_to_none():
    assert json_safe(np.float64('nan')) is None
    assert json_safe(float('nan')) is None


def test_json_safe_converts_numpy_bool():
    assert json_safe(np.bool_(True)) is True
    assert isinstance(json_safe(np.bool_(True)), bool)


def test_json_safe_recurses_into_dict_and_list():
    original = {
        'a': np.int64(3),
        'b': [np.float64(1.5), np.int64(2)],
        'c': {'nested': np.bool_(False)},
    }
    result = json_safe(original)
    assert result == {'a': 3, 'b': [1.5, 2], 'c': {'nested': False}}
    assert isinstance(result['a'], int)
    assert isinstance(result['c']['nested'], bool)


def test_json_safe_real_clean_report_round_trips_through_json():
    import json
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]})
    _, report = clean_data(df)
    # Would raise TypeError before json_safe if numpy types leaked through
    json.dumps(json_safe(report))


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


def test_clean_data_parses_dayfirst_date_column():
    # "19-02-2020" only makes sense day-first (there's no 19th month) - month-first
    # parsing fails on it entirely, so day-first has to be tried and preferred.
    df = pd.DataFrame({
        "date": ["05-02-2020", "12-02-2020", "19-02-2020", "26-02-2020"],
        "val": [1, 2, 3, 4],
    })
    cleaned, report = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["date"].iloc[0] == pd.Timestamp("2020-02-05")
    assert cleaned["date"].iloc[2] == pd.Timestamp("2020-02-19")


def test_clean_data_date_column_tolerates_partial_garbage():
    # 8/10 valid dates (80%, above the 70% threshold) with 2 genuinely unparseable
    # values - should still convert, with NaT for the bad ones, rather than a single
    # bad value silently blocking conversion for the whole column.
    dates = [f"2026-01-{i:02d}" for i in range(1, 9)] + ["not a date", "also not a date"]
    df = pd.DataFrame({"date": dates, "val": range(10)})
    cleaned, report = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["date"].isna().sum() == 2


# ── detect_and_convert_numeric_column ────────────────────────────────────────

def test_detect_and_convert_numeric_column_currency_symbol():
    series = pd.Series(["$1,200.50", "$980", "$1,050.25"])
    result, currency, is_percent = detect_and_convert_numeric_column(series)
    assert result.tolist() == [1200.50, 980.0, 1050.25]
    assert currency == 'USD'
    assert is_percent is False


def test_detect_and_convert_numeric_column_currency_word():
    series = pd.Series(["45,000 Naira", "12,000 Naira", "8,500 Naira"])
    result, currency, is_percent = detect_and_convert_numeric_column(series)
    assert result.tolist() == [45000.0, 12000.0, 8500.0]
    assert currency == 'NGN'


def test_detect_and_convert_numeric_column_percent():
    series = pd.Series(["45%", "12%", "99%"])
    result, currency, is_percent = detect_and_convert_numeric_column(series)
    assert result.tolist() == [45.0, 12.0, 99.0]
    assert is_percent is True
    assert currency is None


def test_detect_and_convert_numeric_column_accounting_negative():
    series = pd.Series(["1,200", "(500)", "300"])
    result, currency, is_percent = detect_and_convert_numeric_column(series)
    assert result.tolist() == [1200.0, -500.0, 300.0]


def test_detect_and_convert_numeric_column_rejects_free_text():
    series = pd.Series(["north", "south", "east"])
    result, currency, is_percent = detect_and_convert_numeric_column(series)
    assert result is None
    assert currency is None


# ── detect_and_convert_boolean_column ────────────────────────────────────────

def test_detect_and_convert_boolean_column_yes_no():
    series = pd.Series(["Yes", "No", "Yes", "No"])
    result = detect_and_convert_boolean_column(series)
    assert result.tolist() == [True, False, True, False]
    assert str(result.dtype) == 'boolean'


def test_detect_and_convert_boolean_column_rejects_non_boolean_text():
    series = pd.Series(["north", "south", "north"])
    assert detect_and_convert_boolean_column(series) is None


def test_detect_and_convert_boolean_column_rejects_single_value():
    series = pd.Series(["Yes", "Yes", "Yes"])
    assert detect_and_convert_boolean_column(series) is None


# ── convert_column_types / clean_data integration ────────────────────────────

def test_convert_column_types_renames_currency_column():
    df = pd.DataFrame({"revenue": ["$1,200", "$980", "$1,050"]})
    converted, report = convert_column_types(df)
    assert "revenue (USD)" in converted.columns
    assert pd.api.types.is_numeric_dtype(converted["revenue (USD)"])
    assert report['currency_columns_converted'] == {'revenue': 'revenue (USD)'}


def test_convert_column_types_converts_boolean_column():
    df = pd.DataFrame({"in_stock": ["Yes", "No", "Yes"]})
    converted, report = convert_column_types(df)
    assert str(converted["in_stock"].dtype) == 'boolean'
    assert report['boolean_columns_converted'] == ['in_stock']


def test_convert_column_types_leaves_mostly_numeric_id_column_as_text():
    # "Invoice" is a real-world example: mostly digits, but a "C" prefix marks cancelled
    # orders - converting to numeric would silently turn that signal into NaN.
    df = pd.DataFrame({"Invoice": ["536365", "536366", "536367", "C536379"]})
    converted, report = convert_column_types(df)
    assert converted["Invoice"].dtype == object
    assert list(converted["Invoice"]) == ["536365", "536366", "536367", "C536379"]


def test_convert_column_types_leaves_camel_case_id_column_as_text():
    # "StockCode" has no separator between "Stock" and "Code" - the ID-name check must
    # split camelCase, not just punctuation, to catch it.
    df = pd.DataFrame({"StockCode": ["21654", "21703", "90123B", "85199S"]})
    converted, report = convert_column_types(df)
    assert converted["StockCode"].dtype == object


def test_clean_data_converts_currency_and_boolean_columns():
    df = pd.DataFrame({
        "price": ["$10.50", "$20.00", None, "$15.25"],
        "readmitted": ["Yes", "No", "No", "Yes"],
    })
    cleaned, report = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned["price (USD)"])
    assert cleaned["price (USD)"].isnull().sum() == 0
    assert str(cleaned["readmitted"].dtype) == 'boolean'
    assert report['currency_columns_converted'] == {'price': 'price (USD)'}
    assert report['boolean_columns_converted'] == ['readmitted']


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


def test_write_df_to_excel_splits_across_sheets_when_over_row_limit():
    # Excel caps a sheet at 1,048,576 rows including the header - real datasets can exceed
    # that (a 1M+-row guest-mode upload triggered exactly this in production). No row should
    # be dropped; the data should just spill onto additional sheets instead.
    df = pd.DataFrame({"value": range(10)})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        _write_df_to_excel(writer, df, 'Data', index=False, max_rows=4)
    buffer.seek(0)
    sheets = pd.read_excel(buffer, sheet_name=None)
    assert list(sheets.keys()) == ['Data', 'Data 2', 'Data 3']
    assert sorted(pd.concat(sheets.values())['value'].tolist()) == list(range(10))


# ── suggest_category_and_metric_columns ──────────────────────────────────────

def test_suggest_category_and_metric_columns_typical():
    df = pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 3,
        "revenue": range(12),
    })
    cat, num = suggest_category_and_metric_columns(df)
    assert cat == "region"
    assert num == "revenue"


def test_suggest_category_and_metric_columns_no_categorical():
    df = pd.DataFrame({"revenue": range(5)})
    cat, num = suggest_category_and_metric_columns(df)
    assert cat is None
    assert num == "revenue"


def test_suggest_category_and_metric_columns_no_numeric():
    df = pd.DataFrame({"region": ["North", "South"]})
    cat, num = suggest_category_and_metric_columns(df)
    assert cat == "region"
    assert num is None


def test_suggest_category_and_metric_columns_prefers_measure_name_over_id():
    # "invoice_id" comes first and is near-unique (a real ID), but "revenue" is the
    # column actually worth summing/charting.
    df = pd.DataFrame({
        "invoice_id": range(1000, 1020),
        "revenue": [50.0] * 20,
    })
    cat, num = suggest_category_and_metric_columns(df)
    assert num == "revenue"


def test_suggest_category_and_metric_columns_prefers_richer_category():
    # "is_holiday" (2 values) comes first, but "store_type" (3 values) is more
    # informative for a category breakdown - it should be preferred.
    df = pd.DataFrame({
        "is_holiday": [True, False] * 10,
        "store_type": (["A", "B", "C"] * 7)[:20],
        "sales": range(20),
    })
    cat, num = suggest_category_and_metric_columns(df)
    assert cat == "store_type"


# ── top_performers_analysis (Sales & Retail) ─────────────────────────────────

def test_top_performers_analysis_ranks_correctly():
    df = pd.DataFrame({
        "product": ["A", "A", "B", "B", "C"],
        "revenue": [100, 100, 50, 50, 10],
    })
    result = top_performers_analysis(df, "product", "revenue")
    assert list(result["product"]) == ["A", "B", "C"]
    assert list(result["total"]) == [200, 100, 10]
    assert result["share_pct"].sum() == pytest.approx(100.0, abs=0.1)


def test_top_performers_analysis_respects_top_n():
    df = pd.DataFrame({
        "product": ["A", "B", "C", "D"],
        "revenue": [40, 30, 20, 10],
    })
    result = top_performers_analysis(df, "product", "revenue", top_n=2)
    assert len(result) == 2
    assert list(result["product"]) == ["A", "B"]


def test_top_performers_analysis_missing_column_raises():
    df = pd.DataFrame({"product": ["A"], "revenue": [1]})
    with pytest.raises(ValueError):
        top_performers_analysis(df, "product", "nonexistent")


# ── concentration_risk_analysis (Finance & Banking) ──────────────────────────

def test_concentration_risk_fully_concentrated_is_high():
    df = pd.DataFrame({
        "account": ["A", "B", "C"],
        "balance": [1000, 0, 0],
    })
    result = concentration_risk_analysis(df, "account", "balance")
    assert result["hhi"] == pytest.approx(1.0)
    assert result["risk_level"] == "High"
    assert result["top_category"] == "A"
    assert result["top_category_share_pct"] == pytest.approx(100.0)


def test_concentration_risk_evenly_spread_is_low():
    df = pd.DataFrame({
        "account": [f"acct_{i}" for i in range(10)],
        "balance": [100] * 10,
    })
    result = concentration_risk_analysis(df, "account", "balance")
    assert result["hhi"] == pytest.approx(0.1)
    assert result["risk_level"] == "Low"


def test_concentration_risk_breakdown_shares_sum_to_one():
    df = pd.DataFrame({"account": ["A", "B", "C"], "balance": [500, 300, 200]})
    result = concentration_risk_analysis(df, "account", "balance")
    assert result["breakdown"]["share"].sum() == pytest.approx(1.0)


def test_concentration_risk_missing_column_raises():
    df = pd.DataFrame({"account": ["A"], "balance": [1]})
    with pytest.raises(ValueError):
        concentration_risk_analysis(df, "account", "nonexistent")


# ── control_chart_analysis (Engineering & Manufacturing) ─────────────────────

def test_control_chart_flags_clear_outlier():
    # A realistic-sized sample: with too few points, one extreme outlier can inflate
    # mean +/- 3*std enough to widen the limits and mask itself - a known limitation of
    # small-sample control charts generally, not specific to this implementation. Real
    # SPC practice uses a reasonably sized baseline for exactly this reason.
    df = pd.DataFrame({"measurement": [10.0] * 20 + [50.0]})
    result = control_chart_analysis(df, "measurement")
    assert result["out_of_control_count"] == 1
    outlier_row = result["points"][result["points"]["measurement"] == 50.0]
    assert outlier_row["in_control"].iloc[0] == False  # noqa: E712
    normal_row = result["points"][result["points"]["measurement"] == 10.0].iloc[0]
    assert normal_row["in_control"] == True  # noqa: E712


def test_control_chart_no_outliers_in_tight_data():
    df = pd.DataFrame({"measurement": [10.0, 10.1, 9.9, 10.0, 9.95, 10.05]})
    result = control_chart_analysis(df, "measurement")
    assert result["out_of_control_count"] == 0


def test_control_chart_sorts_by_sequence_col():
    df = pd.DataFrame({
        "measurement": [3, 1, 2],
        "run_order": [3, 1, 2],
    })
    result = control_chart_analysis(df, "measurement", sequence_col="run_order")
    assert list(result["points"]["measurement"]) == [1, 2, 3]


def test_control_chart_missing_column_raises():
    df = pd.DataFrame({"measurement": [1, 2, 3]})
    with pytest.raises(ValueError):
        control_chart_analysis(df, "nonexistent")


# ── validate_password_strength ───────────────────────────────────────────────

def test_validate_password_strength_accepts_valid_password():
    is_valid, message = validate_password_strength("Secure1Pass")
    assert is_valid is True
    assert message == ""


def test_validate_password_strength_rejects_too_short():
    is_valid, message = validate_password_strength("Ab1defg")
    assert is_valid is False
    assert "8 characters" in message


def test_validate_password_strength_rejects_no_uppercase():
    is_valid, message = validate_password_strength("lowercase1")
    assert is_valid is False
    assert "uppercase" in message


def test_validate_password_strength_rejects_no_lowercase():
    is_valid, message = validate_password_strength("UPPERCASE1")
    assert is_valid is False
    assert "lowercase" in message


def test_validate_password_strength_rejects_no_digit():
    is_valid, message = validate_password_strength("NoDigitsHere")
    assert is_valid is False
    assert "number" in message


# ── time_trend_analysis ──────────────────────────────────────────────────────

def test_time_trend_analysis_aggregates_by_day():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "revenue": [100, 50, 200],
    })
    result = time_trend_analysis(df, "date", "revenue")
    assert list(result["total"]) == [150, 200]
    assert len(result) == 2


def test_time_trend_analysis_ignores_unparseable_dates():
    df = pd.DataFrame({
        "date": ["2026-01-01", "not-a-date", "2026-01-02"],
        "revenue": [100, 999, 200],
    })
    result = time_trend_analysis(df, "date", "revenue")
    assert result["total"].sum() == 300


def test_time_trend_analysis_missing_column_raises():
    df = pd.DataFrame({"date": ["2026-01-01"], "revenue": [100]})
    with pytest.raises(ValueError):
        time_trend_analysis(df, "date", "nonexistent")


# ── industry_kpi_summary ─────────────────────────────────────────────────────

def test_industry_kpi_summary_typical():
    df = pd.DataFrame({
        "department": ["ER", "ER", "Cardiology", "Radiology"],
        "patients": [40, 10, 30, 20],
    })
    result = industry_kpi_summary(df, "department", "patients")
    assert result["total"] == 100.0
    assert result["category_count"] == 3
    assert result["top_category"] == "ER"
    assert result["top_category_share_pct"] == 50.0
    assert result["avg_per_category"] == pytest.approx(33.33, abs=0.01)


def test_industry_kpi_summary_empty_dataframe():
    df = pd.DataFrame({"department": [], "patients": []})
    result = industry_kpi_summary(df, "department", "patients")
    assert result["category_count"] == 0
    assert result["top_category"] is None
    assert result["avg_per_category"] == 0.0


def test_industry_kpi_summary_missing_column_raises():
    df = pd.DataFrame({"department": ["ER"], "patients": [1]})
    with pytest.raises(ValueError):
        industry_kpi_summary(df, "department", "nonexistent")


# ── segment_categories ───────────────────────────────────────────────────────

def test_segment_categories_assigns_every_category():
    df = pd.DataFrame({
        "product": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "revenue": [1000, 1000, 900, 900, 50, 50, 40, 40],
        "units": [10, 10, 9, 9, 200, 200, 190, 190],
    })
    result = segment_categories(df, "product", ["revenue", "units"], n_clusters=2)
    assert set(result["segments"]["product"]) == {"A", "B", "C", "D"}
    assert result["n_clusters"] == 2
    assert len(result["profile"]) == 2
    assert result["profile"]["category_count"].sum() == 4


def test_segment_categories_reduces_clusters_when_too_few_categories():
    df = pd.DataFrame({"product": ["A", "B"], "revenue": [100, 50]})
    result = segment_categories(df, "product", ["revenue"], n_clusters=5)
    assert result["n_clusters"] == 2


def test_segment_categories_too_few_categories_raises():
    df = pd.DataFrame({"product": ["A"], "revenue": [100]})
    with pytest.raises(ValueError):
        segment_categories(df, "product", ["revenue"])


def test_segment_categories_missing_numeric_col_raises():
    df = pd.DataFrame({"product": ["A", "B"], "revenue": [100, 50]})
    with pytest.raises(ValueError):
        segment_categories(df, "product", ["nonexistent"])


# ── estimate_time_to_limit ───────────────────────────────────────────────────

def test_estimate_time_to_limit_detects_increasing_trend():
    points_df = pd.DataFrame({"measurement": [10, 12, 14, 16, 18, 20]})
    result = estimate_time_to_limit(points_df, "measurement", ucl=30, lcl=0)
    assert result["trend"] == "increasing"
    assert result["heading_toward"] == "upper control limit"
    assert result["periods_to_breach"] > 0


def test_estimate_time_to_limit_stable_data_has_no_breach_estimate():
    points_df = pd.DataFrame({"measurement": [10, 10, 10, 10, 10, 10]})
    result = estimate_time_to_limit(points_df, "measurement", ucl=30, lcl=0)
    assert result["trend"] == "stable"
    assert result["periods_to_breach"] is None


def test_estimate_time_to_limit_too_few_points_raises():
    points_df = pd.DataFrame({"measurement": [10, 12, 14]})
    with pytest.raises(ValueError):
        estimate_time_to_limit(points_df, "measurement", ucl=30, lcl=0)


def test_estimate_time_to_limit_missing_column_raises():
    points_df = pd.DataFrame({"measurement": [10, 12, 14, 16, 18]})
    with pytest.raises(ValueError):
        estimate_time_to_limit(points_df, "nonexistent", ucl=30, lcl=0)


# ── binary_outcome_risk_model ────────────────────────────────────────────────

def test_binary_outcome_risk_model_typical():
    np.random.seed(42)
    n = 60
    risk_feature = np.concatenate([np.random.normal(20, 3, n // 2), np.random.normal(60, 3, n // 2)])
    outcome = ["no"] * (n // 2) + ["yes"] * (n // 2)
    df = pd.DataFrame({"risk_feature": risk_feature, "readmitted": outcome})
    result = binary_outcome_risk_model(df, "readmitted", ["risk_feature"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["positive_class"] == "yes"
    assert "risk_score" in result["scored_data"].columns
    assert len(result["feature_importances"]) == 1
    scores = result["scored_data"]["risk_score"]
    assert list(scores) == sorted(scores, reverse=True)


def test_binary_outcome_risk_model_too_few_rows_raises():
    df = pd.DataFrame({"outcome": ["yes", "no"] * 5, "feature": range(10)})
    with pytest.raises(ValueError):
        binary_outcome_risk_model(df, "outcome", ["feature"])


def test_binary_outcome_risk_model_wrong_class_count_raises():
    df = pd.DataFrame({"outcome": ["a", "b", "c"] * 10, "feature": range(30)})
    with pytest.raises(ValueError):
        binary_outcome_risk_model(df, "outcome", ["feature"])


def test_binary_outcome_risk_model_missing_feature_raises():
    df = pd.DataFrame({"outcome": ["yes", "no"] * 15, "feature": range(30)})
    with pytest.raises(ValueError):
        binary_outcome_risk_model(df, "outcome", ["nonexistent"])


# ── sample_for_chart / sample_for_chart_preserving_flag ──────────────────────

def test_sample_for_chart_no_op_when_under_limit():
    df = pd.DataFrame({"x": range(100)})
    result, was_sampled = sample_for_chart(df, max_points=5000)
    assert was_sampled is False
    assert len(result) == 100


def test_sample_for_chart_samples_when_over_limit():
    df = pd.DataFrame({"x": range(10000)})
    result, was_sampled = sample_for_chart(df, max_points=2000)
    assert was_sampled is True
    assert len(result) == 2000


def test_sample_for_chart_deterministic_across_calls():
    df = pd.DataFrame({"x": range(10000)})
    result1, _ = sample_for_chart(df, max_points=500)
    result2, _ = sample_for_chart(df, max_points=500)
    assert list(result1["x"]) == list(result2["x"])


def test_sample_for_chart_preserving_flag_no_op_when_under_limit():
    df = pd.DataFrame({"x": range(100), "_anomaly": [False] * 95 + [True] * 5})
    result, was_sampled = sample_for_chart_preserving_flag(df, "_anomaly", max_points=5000)
    assert was_sampled is False
    assert len(result) == 100


def test_sample_for_chart_preserving_flag_keeps_all_flagged_rows():
    df = pd.DataFrame({
        "x": range(10000),
        "_anomaly": [i % 500 == 0 for i in range(10000)],  # 20 flagged rows
    })
    result, was_sampled = sample_for_chart_preserving_flag(df, "_anomaly", max_points=2000)
    assert was_sampled is True
    assert result["_anomaly"].sum() == 20
    assert len(result) == 2000


def test_sample_for_chart_preserving_flag_missing_column_returns_unchanged():
    df = pd.DataFrame({"x": range(10000)})
    result, was_sampled = sample_for_chart_preserving_flag(df, "_anomaly", max_points=2000)
    assert was_sampled is False
    assert len(result) == 10000


# ── is_rate_like_metric ──────────────────────────────────────────────────────

def test_is_rate_like_metric_true_for_price_and_rate_names():
    assert is_rate_like_metric("Fuel_Price") is True
    assert is_rate_like_metric("Interest Rate") is True
    assert is_rate_like_metric("CPI") is True
    assert is_rate_like_metric("Unit Cost") is True


def test_is_rate_like_metric_false_for_additive_names():
    assert is_rate_like_metric("Revenue") is False
    assert is_rate_like_metric("Weekly_Sales") is False
    assert is_rate_like_metric("Units Sold") is False


# ── top_performers_analysis / time_trend_analysis / industry_kpi_summary agg ──

def test_top_performers_analysis_averages_rate_like_metric():
    df = pd.DataFrame({
        "store": ["A", "A", "B", "B"],
        "fuel_price": [3.50, 3.70, 4.00, 4.20],
    })
    result = top_performers_analysis(df, "store", "fuel_price")
    assert dict(zip(result["store"], result["total"])) == {"A": pytest.approx(3.60), "B": pytest.approx(4.10)}


def test_top_performers_analysis_still_sums_additive_metric():
    df = pd.DataFrame({
        "store": ["A", "A", "B"],
        "revenue": [100, 100, 50],
    })
    result = top_performers_analysis(df, "store", "revenue")
    assert dict(zip(result["store"], result["total"])) == {"A": 200, "B": 50}


def test_time_trend_analysis_averages_rate_like_metric():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "price": [10.0, 20.0, 40.0],
    })
    result = time_trend_analysis(df, "date", "price")
    assert list(result["total"]) == [15.0, 40.0]


def test_industry_kpi_summary_uses_overall_mean_for_rate_like_metric():
    df = pd.DataFrame({
        "store": ["A", "A", "B", "B", "B"],
        "fuel_price": [3.50, 3.70, 4.00, 4.20, 4.40],
    })
    result = industry_kpi_summary(df, "store", "fuel_price")
    assert result["agg"] == "mean"
    assert result["total"] == pytest.approx(df["fuel_price"].mean(), abs=0.01)


def test_industry_kpi_summary_agg_is_sum_for_additive_metric():
    df = pd.DataFrame({
        "department": ["ER", "ER", "Cardiology"],
        "patients": [40, 10, 30],
    })
    result = industry_kpi_summary(df, "department", "patients")
    assert result["agg"] == "sum"
    assert result["total"] == 80.0


# ── detect_industry ───────────────────────────────────────────────────────────

def test_detect_industry_identifies_retail_data():
    df = pd.DataFrame(columns=[
        "Store", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"
    ])
    industry, keywords = detect_industry(df)
    assert industry == "sales_retail"
    assert "sales" in keywords
    assert "store" in keywords


def test_detect_industry_identifies_healthcare_data():
    df = pd.DataFrame(columns=[
        "Patient Name", "Age", "Gender", "Blood Type", "Medical Condition",
        "Date of Admission", "Doctor", "Hospital", "Insurance Provider",
        "Billing Amount", "Admission Type", "Discharge Date", "Medication",
    ])
    industry, keywords = detect_industry(df)
    assert industry == "healthcare"
    assert "patient" in keywords


def test_detect_industry_identifies_finance_data():
    df = pd.DataFrame(columns=[
        "Customer_ID", "Account_Number", "Account_Type", "Loan_Type", "Loan_Amount",
        "Outstanding_Amount", "EMI_Amount", "Loan_Status", "Credit_Score",
    ])
    industry, keywords = detect_industry(df)
    assert industry == "finance_banking"


def test_detect_industry_identifies_manufacturing_data():
    df = pd.DataFrame(columns=[
        "Machine_ID", "Timestamp", "Temperature", "Vibration", "Pressure",
        "Downtime_Risk_Score", "Failure_Type", "Maintenance_Required",
    ])
    industry, keywords = detect_industry(df)
    assert industry == "engineering_manufacturing"


def test_detect_industry_returns_none_for_ambiguous_columns():
    df = pd.DataFrame(columns=["id", "name", "date", "value"])
    industry, keywords = detect_industry(df)
    assert industry is None
    assert keywords == []


def test_detect_industry_ignores_internal_columns():
    df = pd.DataFrame(columns=["Store", "Weekly_Sales", "_anomaly", "_anomaly_score"])
    industry, keywords = detect_industry(df)
    assert industry == "sales_retail"
