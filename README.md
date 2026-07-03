# 🤖 NobiChan AI — Offline AI Data Analyst Agent

> **"Just like Nobita needed Doraemon, your data needs NobiChan!"**

NobiChan is a fully offline, zero-cost AI-powered Data Analyst Agent built with Python, Flask, Ollama, and Mistral LLM. Upload any dataset and NobiChan will automatically analyze it, detect problems, generate visualizations, and chat with you about your data — all without internet or API keys.

---

## ✨ Features

- 📤 **Upload & Analyze** — Supports CSV and Excel files up to 200MB
- 🧹 **Auto Data Cleaning** — Detects missing values, duplicates, outliers, and skewness
- 📊 **Smart Visualizations** — Auto-generated interactive charts using Plotly
- 📋 **PDF Report Generation** — Professional downloadable PDF with full analysis
- 🤖 **AI Chat** — Ask NobiChan questions about your data in plain English
- 💡 **Smart Suggestions** — Get actionable recommendations for data improvements
- 📈 **Statistical Analysis** — Mean, median, std dev, skewness, kurtosis, correlations
- 🔗 **Correlation Detection** — Automatically finds strong relationships between columns
- 💾 **Export Clean Data** — Download cleaned dataset as CSV or Excel
- ⚡ **100% Offline** — No internet, no API key, no cost — runs entirely on your laptop

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core programming language |
| Flask | Web framework / API backend |
| Ollama + Mistral | Local LLM for AI chat and insights |
| Pandas + NumPy | Data manipulation and analysis |
| Plotly | Interactive visualizations |
| Matplotlib + Seaborn | Statistical charts |
| ReportLab | PDF report generation |
| HTML / CSS / JS | Frontend UI |
| SciPy | Statistical computations |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com) installed on your machine
- Mistral model pulled via Ollama

### Step 1 — Install Ollama & Mistral

```bash
# Download Ollama from https://ollama.com
# Then pull the Mistral model
ollama pull mistral
```

### Step 2 — Clone the Repository

```bash
git clone https://github.com/yourusername/nobichan-ai.git
cd nobichan-ai
```

### Step 3 — Install Python Dependencies

```bash
pip install flask pandas numpy matplotlib seaborn plotly scipy reportlab openpyxl requests
```

### Step 4 — Run NobiChan

```bash
# First start Ollama in a separate terminal
ollama serve

# Then run NobiChan
python app.py
```

### Step 5 — Open in Browser

```
http://localhost:5000
```

---

## 📁 Project Structure

```
nobichan-ai/
│
├── app.py              # Flask backend — main application
├── index.html          # Frontend UI
├── uploads/            # Uploaded datasets (auto-created)
├── reports/            # Generated PDF reports (auto-created)
└── README.md           # This file
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve the main UI |
| POST | `/api/upload` | Upload CSV or Excel file |
| GET | `/api/analysis` | Get full dataset analysis |
| POST | `/api/chat` | Chat with NobiChan AI |
| GET | `/api/report` | Download PDF report |
| GET | `/api/export_clean` | Export cleaned data (CSV/Excel) |
| GET | `/api/status` | Check Ollama & data status |

---

## 📸 How It Works

1. **Upload** your CSV or Excel file
2. NobiChan **automatically analyzes** your data — shape, types, missing values, outliers, correlations
3. View **interactive charts** and **statistical summaries**
4. **Chat** with NobiChan in plain English about your data
5. Download a **professional PDF report** with full insights
6. Export your **cleaned dataset**

---

## 💡 Example Questions to Ask NobiChan

- *"What are the key insights from this dataset?"*
- *"Which columns have the most missing values?"*
- *"Write a SQL query to find the top 5 customers by revenue"*
- *"What trends do you see in the sales data?"*
- *"Which columns are most correlated?"*

---

## ⚙️ Configuration

You can change the AI model in `app.py`:

```python
OLLAMA_MODEL = "mistral"   # Change to llama3, phi3, gemma, etc.
```

Change the max file upload size:

```python
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
```

---

## 🧑‍💻 Built By

**Kalaivani** — B.Tech Information Technology  

---

## 📌 Why NobiChan?

Named after the beloved anime characters **Nobita** and **Shinchan** —  
just like Nobita always needed Doraemon's gadgets to solve problems,  
**your data needs NobiChan** to unlock its hidden insights! 🎌



---

> *"Data is just numbers until NobiChan gives it a story."* 🤖📊
