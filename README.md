# Sentiment Analysis Simobi+ Bank Sinarmas

Sentiment analysis of **Simobi+ Bank Sinarmas** app reviews from the Google Play Store, using a lexicon-based approach for automatic labeling and machine learning for classification.

> **Disclaimer:** This project is built purely for learning and analytical purposes. It is not affiliated with, endorsed by, or intended to represent Bank Sinarmas in any way.

## Project Structure

```
├── data/
│   ├── reviews_simobi.csv       # Scraped reviews dataset (16,802 rows)
│   ├── slang_words.csv          # Indonesian slang word dictionary
│   ├── lexicon_positive.csv     # Positive sentiment lexicon (InSet)
│   └── lexicon_negative.csv     # Negative sentiment lexicon (InSet)
├── src/
│   ├── scrapping_reviews.ipynb  # Scraping reviews from Google Play Store
│   └── sentiment_analysis_simobi.ipynb  # Model training notebook
├── models/
│   ├── lr_sentiment.pkl         # Final Logistic Regression model
│   └── tfidf.pkl                # TF-IDF vectorizer
└── requirements.txt
```

## Workflow

### 1. Data Scraping
Reviews are collected from the Google Play Store using `google-play-scraper` with app ID `com.simas.mobile.SimobiPlus`.

### 2. Preprocessing

Indonesian text preprocessing pipeline:

1. Emoji to text conversion
2. Cleaning (remove mentions, URLs, numbers, special characters)
3. Case folding (lowercase)
4. Elongation normalization (`mantaaaaap` → `mantaap`)
5. Slang word normalization
6. Tokenization
7. Stopword removal (NLTK + Sastrawi + custom)
8. Stemming (Sastrawi) with caching

### 3. Automatic Labeling (Lexicon)

Uses the **InSet** (Indonesian Sentiment Lexicon) with:

- Negation handling within a 2-word window (`tidak bagus` → negative)
- Diminisher handling (`kurang bagus` → weight reduced by 0.5)
- Labels: `positive`, `negative`, `neutral`

### 4. Model Training
Binary classification (positive vs negative) using TF-IDF with multiple algorithms:

| Model | F1-macro | Acc Test |
|-------|----------|----------|
| **Logistic Regression** | **0.89** | **0.89** |
| Linear SVM | 0.88 | 0.887 |
| Random Forest | 0.85 | 0.856 |
| Decision Tree | 0.83 | 0.837 |
| Naive Bayes | 0.82 | 0.829 |

Best model: **Logistic Regression** with `C=10`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Scrape new reviews

Run `src/scrapping_reviews.ipynb`

### Train the model

Run `src/sentiment_analysis_simobi.ipynb` from top to bottom.

### Inference
An inference cell is available at the end of `sentiment_analysis_simobi.ipynb`:

```python
sentiment_prediction("aplikasi sangat membantu transaksi cepat dan mudah")
# → POSITIVE

sentiment_prediction("sering error saat login transaksi gagal terus")
# → NEGATIVE
```

## Main Libraries

- `google-play-scraper` — Play Store review scraping
- `PySastrawi` — Indonesian stemming
- `NLTK` — tokenization and stopwords
- `scikit-learn` — TF-IDF and ML models
- `wordcloud`, `matplotlib`, `seaborn` — visualization
