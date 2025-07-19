# Kupids Classifier

Kupids Classifier is an end-to-end sentiment analysis platform for Google Play app reviews. It features automated scraping, data cleaning, model inference, EDA, and a modern web UI, with natural language explanations powered by Gemini.

---

## Model Performance Summary

| Model                | Accuracy  | F1      |
|----------------------|-----------|---------|
| VADER                | 0.754417  | 0.763008|
| Logistic Regression  | 0.871583  | 0.869771|
| DeBERTa-v3-small     | 0.881917  | 0.882514|

**Conclusion:**

The DeBERTa-v3-small model achieved the highest accuracy and F1 score, making it the best performing model for sentiment analysis on Google Play app reviews in this project.

---

## Directory Structure

```
├── api.py                # FastAPI backend
├── app.py                # Streamlit frontend
├── src/                  # Pipeline modules (preprocessing, inference, etc.)
├── models/               # Saved models and vectorizers
├── results/              # Model checkpoints
├── data/                 # Raw and processed data
├── temp/                 # Temporary files (plots, etc.)
├── README.md
└── ...
```

---

## Setup & Installation

- **Python 3.8+ required**
- All dependencies are managed via `pyproject.toml` and [uv](https://github.com/astral-sh/uv)

### 1. Clone the repository
```bash
git clone <repo-url>
cd kupids-classifier
```

### 2. Install uv
```bash
pip install uv
```

### 3. Install dependencies
```bash
uv pip install -r pyproject.toml
```

---

## How to Run

### 1. Start the FastAPI backend
```bash
uvicorn api:app --reload
```
Backend runs at `http://localhost:8000`

### 2. Start the Streamlit frontend
```bash
streamlit run app.py
```
Frontend runs at `http://localhost:8501`

### 3. Usage
- Enter a Google Play app ID (e.g., `com.tinder`) or upload a CSV in the Streamlit UI.
- Click "Analyze" to run the pipeline.
- View EDA plots, summary statistics, model comparison, and a Gemini-generated explanation.

---

## API Contract (OpenAPI YAML)

See [`api_contract.yaml`](./api_contract.yaml) for the full OpenAPI spec. Key endpoint:

```
POST /analyze
{
  "app_id": "com.tinder",         # Google Play app ID
  "csv_path": "/path/to/file.csv" # (optional) Path to CSV of reviews
}
```

**Response:**
- EDA plots (paths to images)
- Summary statistics
- Model comparison table
- Best model
- Gemini explanation

---

## For Contributors & Developers

- Notebooks for EDA and modeling are in `notebooks/`.
- Data scraping utilities are in `src/scraper.py`.
- All pipeline logic is modularized in `src/`.
- To add new models or EDA, extend the relevant modules and update the API/frontend as needed.

---

For more details, see the code and comments in each module.
