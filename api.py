# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os

from src.preprocessing import preprocess_reviews
from src.inference import run_all_inference
from src.eda import plot_sentiment_distribution, plot_wordcloud, get_summary_stats
from src.model_comparison import compare_models
from src.scraper import fetch_reviews

app = FastAPI()

class AnalyzeRequest(BaseModel):
    app_id: str
    csv_path: Optional[str] = None

@app.post('/analyze')
def analyze(request: AnalyzeRequest):
    print('--- [Pipeline] Step 1: Scrape or load reviews ---')
    if request.csv_path and os.path.exists(request.csv_path):
        print(f'Loading reviews from CSV: {request.csv_path}')
        df = pd.read_csv(request.csv_path)
    else:
        print(f'Scraping reviews for app_id: {request.app_id}')
        df = fetch_reviews(request.app_id, count=50000)
        print(f'Scraped {len(df)} reviews.')
        if df.empty:
            print('No reviews found!')
            raise HTTPException(status_code=404, detail='No reviews found for this app ID.')

    # Ensure we have a string column name for text
    text_col = 'content' if 'content' in df.columns else df.columns[0]
    print(f'Using text column: {text_col}')

    print('--- [Pipeline] Step 2: Preprocessing ---')
    df = preprocess_reviews(df, text_col=text_col)
    print('Preprocessing complete.')

    print('--- [Pipeline] Step 3: Inference ---')
    preds_dict = run_all_inference(df, text_col=text_col)
    print('Inference complete.')

    print('--- [Pipeline] Step 4: EDA ---')
    pred_series = pd.Series(preds_dict['DistilBERT'])
    sent_dist_path = plot_sentiment_distribution(pred_series)
    wordcloud_path = plot_wordcloud(df[text_col])
    summary_stats = get_summary_stats(df, text_col=text_col)
    print('EDA complete.')

    print('--- [Pipeline] Step 5: Model Comparison ---')
    y_true = df['sentiment'].tolist() if 'sentiment' in df.columns else None
    comparison_df = compare_models(preds_dict, y_true)
    comparison_table = comparison_df.to_dict(orient='records')
    print('Model comparison complete.')

    print('--- [Pipeline] Step 6: Gemini API (placeholder) ---')
    gemini_explanation = 'Gemini explanation will appear here.'
    print('Pipeline complete. Returning results.')

    return {
        'eda': {
            'sentiment_distribution_plot': sent_dist_path,
            'wordcloud_plot': wordcloud_path,
            'summary_stats': summary_stats
        },
        'model_comparison': comparison_table,
        'best_model': comparison_df.sort_values('F1', ascending=False)['Model'].iloc[0] if 'F1' in comparison_df else None,
        'gemini_explanation': gemini_explanation
    } 