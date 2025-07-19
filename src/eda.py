# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def plot_sentiment_distribution(preds: pd.Series, out_path: str = 'temp/sentiment_distribution.png') -> str:
    print('[eda] Plotting sentiment distribution')
    plt.figure(figsize=(6,4))
    preds.value_counts().sort_index().plot(kind='bar', color=['red', 'green'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks([0,1], ['Negative', 'Positive'], rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f'[eda] Sentiment distribution plot saved to {out_path}')
    return out_path

def plot_wordcloud(texts: pd.Series, out_path: str = 'temp/wordcloud.png') -> str:
    print('[eda] Generating word cloud')
    text = ' '.join(texts.astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f'[eda] Word cloud saved to {out_path}')
    return out_path

def get_summary_stats(df: pd.DataFrame, text_col: str = 'review') -> dict:
    print(f'[eda] Calculating summary statistics for column: {text_col}')
    stats = {
        'num_reviews': len(df),
        'avg_length': df[text_col].astype(str).apply(len).mean(),
        'min_length': df[text_col].astype(str).apply(len).min(),
        'max_length': df[text_col].astype(str).apply(len).max(),
        'num_unique': df[text_col].nunique(),
    }
    print('[eda] Summary statistics calculated')
    return stats 