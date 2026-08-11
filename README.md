# 📊 Ostrivo

> Upload your data. Get instant AI-powered insights, dashboards, and anomaly detection.

Ostrivo is an AI-powered business intelligence web app that transforms raw CSV and Excel files into interactive dashboards, anomaly detection reports, and plain-English executive summaries — in under 30 seconds.

---

## ✨ Features

- **Auto data cleaning** — removes duplicates, fills missing values, parses dates
- **Smart column labelling** — AI-inferred human-readable labels for cryptic column names (falls back to a heuristic without an API key)
- **Interactive dashboards** — histograms, box plots, scatter plots, correlation heatmaps, category breakdowns
- **Anomaly detection** — Isolation Forest flags unusual rows automatically
- **Advisor dashboard** — data quality scorecard plus severity-ranked recommendation cards
- **Forecasting** — trend + seasonality projection for datasets with a date column
- **AI executive summary** — automatically generates a plain-English business summary
- **Data Q&A** — ask natural language questions about your data
- **Download cleaned data** — export the cleaned CSV
- **PDF report export** — download a full report covering overview, quality scores, anomalies, recommendations, and the AI summary

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
Upload any CSV or Excel file — even a simple sales spreadsheet or bank export — and you'll see:
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

## 📁 Supported File Types

- `.csv` — comma-separated values
- `.xlsx` — Excel (modern format)
- `.xls` — Excel (legacy format)

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
| Reporting | fpdf2 (PDF export) |

---

## 📸 Screenshots

Upload any CSV or Excel → instant dashboard, anomaly detection, and AI summary.

---

## 🗺️ Roadmap

- [ ] Email alerts for anomalies
- [ ] Multi-file comparison
- [ ] Industry-specific templates (retail, healthcare, finance)
- [ ] User accounts and saved dashboards
- [ ] GitHub repo + Streamlit Cloud deployment (free hosting)
- [ ] Portfolio website showcasing the project

---

## 👤 Author

**Avwerosuo Peter Imoniose**  
MSc Applied Data Science in Engineering (Distinction) — Glasgow Caledonian University, 2026  
[LinkedIn](https://www.linkedin.com/in/avwerosuo-imoniose-bbb3b915a) · [GitHub](https://github.com/PeterImoniose)

---

## 📄 Licence

MIT
