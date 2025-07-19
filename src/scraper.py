import pandas as pd
from google_play_scraper import reviews, Sort
from typing import Optional

def fetch_reviews(app_id: str, count: int = 50000, lang: str = 'en', country: str = 'us') -> pd.DataFrame:
    print(f'[scraper] Fetching up to {count} reviews for app_id: {app_id}')
    REVIEWS_PER_STAR = count // 5
    all_reviews = []
    for star in range(1, 6):
        print(f'[scraper] Scraping {REVIEWS_PER_STAR} reviews for {star}-star ratings...')
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=REVIEWS_PER_STAR,
            filter_score_with=star
        )
        for r in result:
            r['star'] = star
        all_reviews.extend(result)
    print(f'[scraper] Total reviews scraped: {len(all_reviews)}')
    df = pd.DataFrame(all_reviews)
    return df 