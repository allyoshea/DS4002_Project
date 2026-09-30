"""
Script: 04_sentiment_vader.py

Purpose:
    Perform sentiment analysis on the collected BCI articles using
    the VADER sentiment analysis tool.

Inputs:
    data/articles.csv
    data/articles/*.txt

Outputs:
    data/vader_sentiment_data.csv

Process:
    1. Load the article metadata.
    2. Initialize the VADER sentiment analyzer.
    3. Read each article from its corresponding text file.
    4. Split each article into sentences.
    5. Calculate VADER sentiment scores for each sentence.
    6. Average the sentence-level scores to obtain article-level results.
    7. Save the results as a CSV file.

Notes:
    VADER produces negative, neutral, positive, and compound scores.
    The output contains the mean of each score across the sentences
    in each article.
"""

import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


ARTICLES_FILE = Path("data/articles.csv")
ARTICLES_DIR = Path("data/articles")
OUTPUT_FILE = Path("data/vader_sentiment_data.csv")


# Load the article metadata to determine which articles to analyze.
articles = pd.read_csv(ARTICLES_FILE)

# Initialize the VADER sentiment analyzer.
analyzer = SentimentIntensityAnalyzer()

results = []


# Analyze each article in the dataset.
for _, row in articles.iterrows():

    article_id = int(row["article_id"])

    # Each article is stored using its three-digit article ID.
    text_file = ARTICLES_DIR / f"{article_id:03d}.txt"

    if not text_file.exists():
        print(f"File not found: {text_file}")
        continue

    with open(
        text_file,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Split the article into sentences before calculating VADER scores.
    sentences = sent_tokenize(text)

    neg_scores = []
    neu_scores = []
    pos_scores = []
    compound_scores = []

    # Calculate VADER scores for each sentence.
    for sentence in sentences:

        scores = analyzer.polarity_scores(sentence)

        neg_scores.append(scores["neg"])
        neu_scores.append(scores["neu"])
        pos_scores.append(scores["pos"])
        compound_scores.append(scores["compound"])

    # Average the sentence-level scores to obtain one set of scores
    # representing each article.
    if sentences:

        mean_neg = sum(neg_scores) / len(sentences)
        mean_neu = sum(neu_scores) / len(sentences)
        mean_pos = sum(pos_scores) / len(sentences)
        mean_compound = sum(compound_scores) / len(sentences)

    else:

        mean_neg = 0
        mean_neu = 0
        mean_pos = 0
        mean_compound = 0

    results.append({
        "article_id": article_id,
        "sentences": len(sentences),
        "mean_neg": mean_neg,
        "mean_neu": mean_neu,
        "mean_pos": mean_pos,
        "mean_compound": mean_compound
    })


# Convert the results into a dataframe.
vader_df = pd.DataFrame(results)

if vader_df.empty:
    print("ERROR: No articles were analyzed.")
    raise SystemExit(1)


# Save the article-level VADER results.
vader_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nVADER analysis complete.")
print(f"Articles analyzed: {len(vader_df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst five articles:")
print(vader_df.head())
