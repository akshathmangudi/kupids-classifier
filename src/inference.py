# src/inference.py

import pandas as pd
from typing import List, Dict

# VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# Logistic Regression
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Transformers (DistilBERT)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

# --- VADER ---
def load_vader():
    print('[inference] Loading VADER model')
    return SentimentIntensityAnalyzer()

def infer_vader(df: pd.DataFrame, text_col: str = 'review') -> List[int]:
    print(f'[inference] Running VADER inference on column: {text_col}')
    sia = load_vader()
    def vader_predict(text):
        score = sia.polarity_scores(text)['compound']
        return 1 if score >= 0.05 else 0
    preds = df[text_col].astype(str).apply(vader_predict).tolist()
    print('[inference] VADER inference complete')
    return preds

# --- Logistic Regression ---
def load_logreg_model(model_path: str, vectorizer_path: str):
    print(f'[inference] Loading Logistic Regression model from {model_path} and vectorizer from {vectorizer_path}')
    clf = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    return clf, tfidf

def infer_logreg(df: pd.DataFrame, text_col: str = 'review', model_path: str = 'models/logreg_model.pkl', vectorizer_path: str = 'models/tfidf_vectorizer.pkl') -> List[int]:
    print(f'[inference] Running Logistic Regression inference on column: {text_col}')
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        print('[inference] Model or vectorizer not found, returning all negative predictions')
        return [0] * len(df)
    clf, tfidf = load_logreg_model(model_path, vectorizer_path)
    X = tfidf.transform(df[text_col].astype(str))
    preds = clf.predict(X).tolist()
    print('[inference] Logistic Regression inference complete')
    return preds

# --- DistilBERT/Transformers ---
def load_transformer(model_name_or_path: str = 'results/checkpoint-3500'):
    model_path = os.path.abspath(model_name_or_path)
    print(f'[inference] Loading DistilBERT model from: {model_path}')
    safetensors_path = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(safetensors_path):
        print(f'[inference] Using safetensors weights: {safetensors_path}')
        model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer, model

def infer_transformer(df: pd.DataFrame, text_col: str = 'review', model_name_or_path: str = 'results/checkpoint-3500', batch_size: int = 32) -> List[int]:
    print(f'[inference] Running DistilBERT inference on column: {text_col}')
    tokenizer, model = load_transformer(model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[inference] Using device: {device}')
    model.to(device)
    model.eval()
    texts = df[text_col].astype(str).tolist()
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).tolist()
        results.extend(preds)
    print('[inference] DistilBERT inference complete')
    return results

# --- Unified Inference ---
def run_all_inference(df: pd.DataFrame, text_col: str = 'review', logreg_model_path: str = 'models/logreg_model.pkl', logreg_vectorizer_path: str = 'models/tfidf_vectorizer.pkl', transformer_model_path: str = 'results/checkpoint-3500') -> Dict[str, List[int]]:
    print('[inference] Running all model inferences')
    results = {
        'VADER': infer_vader(df, text_col),
        'Logistic Regression': infer_logreg(df, text_col, logreg_model_path, logreg_vectorizer_path),
        'DistilBERT': infer_transformer(df, text_col, transformer_model_path),
    }
    print('[inference] All model inferences complete')
    return results 