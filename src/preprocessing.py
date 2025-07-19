# src/preprocessing.py

import re
import pandas as pd

def clean_text(text: str) -> str:
    print('[preprocessing] Cleaning text')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_reviews(df: pd.DataFrame, text_col: str = 'review') -> pd.DataFrame:
    print(f'[preprocessing] Preprocessing reviews in column: {text_col}')
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    print('[preprocessing] Preprocessing complete')
    return df 