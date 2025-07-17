import pandas as pd
from google_play_scraper import reviews, Sort

APP_PACKAGE = 'com.tinder'
REVIEWS_PER_STAR = 10000  # 5 stars x 10k = 50k total

all_reviews = []

for star in range(1, 6):
    print(f"Scraping {REVIEWS_PER_STAR} reviews for {star}-star ratings...")
    result, _ = reviews(
        APP_PACKAGE,
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        count=REVIEWS_PER_STAR,
        filter_score_with=star
    )
    for r in result:
        r['star'] = star  # Ensure star rating is present
    all_reviews.extend(result)

print(f"Total reviews scraped: {len(all_reviews)}")

df = pd.DataFrame(all_reviews)
df.to_csv('data/tinder_reviews.csv', index=False)
print("Saved reviews to data/tinder_reviews.csv") 