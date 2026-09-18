
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the EDA dataset
df = pd.read_csv("eda_data.csv")

# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Store sentiment scores
negative = []
neutral = []
positive = []
compound = []

# Analyze each article based on row order
for i in range(len(df)):

    article_path = f"articles/{i + 1:03d}.txt"

    with open(article_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split article into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove very short/empty pieces
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]

    # Score each sentence
    sentence_scores = []

    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)
        sentence_scores.append(score)

    # Average the sentence-level scores to get article-level scores
    negative.append(sum(score["neg"] for score in sentence_scores) / len(sentence_scores))
    neutral.append(sum(score["neu"] for score in sentence_scores) / len(sentence_scores))
    positive.append(sum(score["pos"] for score in sentence_scores) / len(sentence_scores))
    compound.append(sum(score["compound"] for score in sentence_scores) / len(sentence_scores))

# Add sentiment scores to dataframe
df["negative"] = negative
df["neutral"] = neutral
df["positive"] = positive
df["compound"] = compound

# Save results
df.to_csv("sentiment_data.csv", index=False)

print("\nSENTIMENT ANALYSIS COMPLETE")
print("---------------------------")
print(df[
    ["article_id", "article_type", "negative",
     "neutral", "positive", "compound"]
].to_string(index=False))

print("\nSaved: sentiment_data.csv")
# Compare sentiment between article types
print("\nSENTIMENT BY ARTICLE TYPE")
print("-------------------------")

summary = df.groupby("article_type")["compound"].agg(
    ["mean", "std", "min", "max"]
)

print(summary)