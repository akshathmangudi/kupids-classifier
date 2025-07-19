# kupids-classifier

kupids-classifier is a sentiment analysis tool designed to analyze the intent behind user reviews of the Tinder app, scraped from the Google Play Store. The project focuses on understanding user sentiment and intent through a combination of data cleaning, preprocessing, exploratory data analysis (EDA), and machine learning modeling.

---

## Project Structure
```
kupids-classifier/
│
├── data/
│   ├── tinder_reviews.csv
│   └── tinder_reviews_randomized.csv
│
├── notebooks/
│   ├── 01_exploration.ipynb      # Data loading, cleaning, and initial EDA
│   ├── 02_eda.ipynb              # Further EDA (structure for future work)
│   └── 03_modelling.ipynb        # Modeling (structure for future work)
│
├── src/
│   ├── __init__.py
│   └── scraper.py                # Data scraping utilities
│
├── tests/                        # (For future test scripts)
├── pyproject.toml                # Project dependencies and metadata
├── README.md
└── uv.lock
```

---

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd kupids-classifier
   ```

2. **Install dependencies:**
   - The project uses Python 3.8+.
   - All dependencies are listed in `pyproject.toml`. You can install them using [pip](https://pip.pypa.io/en/stable/) or a tool like [uv](https://github.com/astral-sh/uv):
     ```sh
     pip install -r requirements.txt
     ```
     Or, if using `uv`:
     ```sh
     uv pip install -r pyproject.toml
     ```

   - **Key dependencies:**
     - `pandas`, `numpy` (data handling)
     - `matplotlib`, `seaborn`, `wordcloud` (visualization)
     - `scikit-learn`, `nltk`, `spacy` (NLP and ML)
     - `emoji`, `transformers`, `datasets`, `torch`, `sentencepiece`, `accelerate` (advanced NLP/ML)
     - `google-play-scraper` (data collection)

3. **Download NLTK and spaCy resources:**
   - For NLTK:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     ```
   - For spaCy:
     ```sh
     python -m spacy download en_core_web_sm
     ```

---

## Data Preprocessing & Cleaning

- **Initial Data Loading:**
  - Reviews are loaded from `data/tinder_reviews.csv`.
  - Data is randomized and saved as `tinder_reviews_randomized.csv`.

- **Cleaning Steps:**
  - Drop duplicate rows.
  - Drop rows with missing essential text (`content`).
  - Fill missing values in categorical columns (`reviewCreatedVersion`, `appVersion`) with `'unknown'`.
  - Fill missing numeric columns (`thumbsUpCount`) with `0`.
  - Only keep relevant columns for analysis.

- **Text Preprocessing:**
  - Lowercasing, URL and HTML tag removal.
  - Expand contractions (e.g., "can't" → "cannot").
  - Replace exclamation and question marks with tokens (`_EXCLAMATION_`, `_QUESTION_`).
  - Remove non-alphanumeric characters (except special tokens).
  - Tokenization and stopword removal (with negations retained).
  - Lemmatization using spaCy.
  - Emoji extraction and demojization (e.g., 😂 → `face_with_tears_of_joy`).

---

## Exploratory Data Analysis (EDA)

- **Data Inspection:**
  - Shape, column types, and null value counts.
  - Sample review inspection.

- **Text Analysis:**
  - Distribution of review scores and star ratings.
  - Frequency analysis of words and emojis.
  - Visualization-ready columns for further EDA (e.g., `clean_content`, `emoji_text`).

- **(Further EDA and modeling are scaffolded in `02_eda.ipynb` and `03_modelling.ipynb` for future work.)**

---

## How to Use

- Run the Jupyter notebooks in the `notebooks/` directory to reproduce the analysis and modeling steps.
- Use the scripts in `src/` for scraping or additional data processing.

---
