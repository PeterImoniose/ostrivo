# 📊 Ostrivo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ostrivo-jdcstzn86pqn3chnhbjzf6.streamlit.app/)

> Upload your data. Get instant AI-powered insights, dashboards, and anomaly detection.

Ostrivo is an AI-powered business intelligence web app that transforms raw CSV and Excel files into interactive dashboards, anomaly detection reports, and plain-English executive summaries - in under 30 seconds.

**Live demo:** [ostrivo-jdcstzn86pqn3chnhbjzf6.streamlit.app](https://ostrivo-jdcstzn86pqn3chnhbjzf6.streamlit.app/)

---

## ✨ Features

- **Auto data cleaning** - removes duplicates, fills missing values, parses dates
- **Smart column labelling** - AI-inferred human-readable labels for cryptic column names (falls back to a heuristic without an API key)
- **Interactive dashboards** - histograms, box plots, scatter plots, correlation heatmaps, category breakdowns
- **Anomaly detection** - Isolation Forest flags unusual rows automatically
- **Advisor dashboard** - data quality scorecard plus severity-ranked recommendation cards
- **Forecasting** - trend + seasonality projection for datasets with a date column
- **AI executive summary** - automatically generates a plain-English business summary
- **Data Q&A** - ask natural language questions about your data
- **Download cleaned data** - export the cleaned CSV
- **PDF report export** - download a full report covering overview, quality scores, anomalies, recommendations, and the AI summary
- **Excel export (Power BI-ready)** - multi-sheet workbook (cleaned data, stats, anomalies) that imports cleanly into Power BI via Get Data
- **Auto-Pilot** - one-click full analysis: type an optional goal, get summary + recommendations + forecast in a single view
- **Help assistant** - a sidebar AI chatbot that answers questions about using Ostrivo itself (separate from the data Q&A)
- **Admin console** - password-gated view (`?admin=1`) with usage stats, estimated AI cost, and an activity log

---

## ✅ MVP Status

- [x] File upload (CSV + Excel)
- [x] Auto data cleaning (duplicates, missing values, date parsing)
- [x] AI-powered smart column labelling
- [x] 5 interactive chart types (histogram, box plot, scatter, heatmap, bar)
- [x] Isolation Forest anomaly detection
- [x] Advisor dashboard (data quality scorecard + recommendation cards)
- [x] Trend/seasonality forecasting for dated datasets
- [x] AI-powered executive summary
- [x] Natural language Q&A about your data
- [x] Download cleaned data
- [x] PDF report export
- [x] Power BI-ready Excel export
- [x] Auto-Pilot one-click analysis
- [x] Sidebar help assistant
- [x] Password-gated admin console
- [x] Custom UI styling

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/PeterImoniose/ostrivo.git
cd ostrivo
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
The app opens automatically at `http://localhost:8501`

### 5. Try it
Upload any CSV or Excel file - even a simple sales spreadsheet or bank export - and you'll see:
- Auto-cleaning stats (duplicates removed, missing values filled)
- Interactive charts and a correlation heatmap
- Anomaly detection results
- If you add an AI API key in the sidebar: an AI-generated executive summary and Q&A chat

---

## 🔑 API Key

Ostrivo uses an AI language model API for AI-powered summaries and Q&A.

1. Get a free API key from your AI provider account
2. Enter it in the sidebar when running the app

---

## 🔐 Admin Console

A password-gated view showing usage stats, an estimated AI API cost, and an activity log - no
customer-uploaded data is ever stored or shown here.

1. Set an `admin_password` secret: locally in `.streamlit/secrets.toml`, and on Streamlit Cloud
   under your app's **Settings → Secrets**
2. Visit `<your-app-url>/?admin=1` and enter the password

---

## 🧪 Testing

Core data-processing logic (cleaning, anomaly detection, forecasting, quality scoring, PDF/Excel
generation) lives in `ostrivo_core.py`, kept free of Streamlit and network dependencies so it can
be unit tested directly:

```bash
pip install pytest
pytest tests/
```

CI runs these tests automatically on every push via GitHub Actions.

---

## 📁 Supported File Types

- `.csv` - comma-separated values
- `.xlsx` - Excel (modern format)
- `.xls` - Excel (legacy format)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (Isolation Forest) |
| Visualisation | Plotly |
| AI | Large Language Model API |
| Statistics | SciPy |
| Reporting | fpdf2 (PDF export), xlsxwriter (Excel export) |
| Activity logging | SQLite |
| Testing | pytest, GitHub Actions CI |

---

## 📸 Screenshots

![Ostrivo Dashboard](screenshot.png)

Upload any CSV or Excel → instant dashboard, anomaly detection, and AI summary.

---

## 🗺️ Roadmap

- [ ] Email alerts for anomalies
- [ ] Multi-file comparison
- [ ] Industry-specific templates (retail, healthcare, finance)
- [ ] User accounts and saved dashboards
- [ ] Portfolio website showcasing the project

---

## 👤 Author

**Avwerosuo Peter Imoniose**  
MSc Applied Data Science in Engineering (Distinction) - Glasgow Caledonian University, 2026  
[LinkedIn](https://www.linkedin.com/in/avwerosuo-imoniose-bbb3b915a) · [GitHub](https://github.com/PeterImoniose)

---

## 📄 Licence

MIT
