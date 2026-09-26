import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# File paths
ARTICLES_FILE = Path("data/articles.csv")
ARTICLES_DIR = Path("data/articles")
OUTPUT_FILE = Path("output/vader_sentiment_data.csv")

# Load article metadata
articles = pd.read_csv(ARTICLES_FILE)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

results = []

# Analyze each article
for _, row in articles.iterrows():
    article_id = int(row["article_id"])

    # Find article text file
    text_file = ARTICLES_DIR / f"{article_id:03d}.txt"

    if not text_file.exists():
        print(f"File not found: {text_file}")
        continue

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split article into sentences
    sentences = sent_tokenize(text)

    neg_scores = []
    neu_scores = []
    pos_scores = []
    compound_scores = []

    # Calculate VADER scores for each sentence
    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)

        neg_scores.append(scores["neg"])
        neu_scores.append(scores["neu"])
        pos_scores.append(scores["pos"])
        compound_scores.append(scores["compound"])

    # Average sentence-level VADER scores
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

# Create dataframe
vader_df = pd.DataFrame(results)

# Check results
if vader_df.empty:
    print("ERROR: No articles were analyzed.")
    raise SystemExit(1)

# Save results
vader_df.to_csv(OUTPUT_FILE, index=False)

print("\nVADER analysis complete.")
print(f"Articles analyzed: {len(vader_df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst five articles:")
print(vader_df.head())