
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
ARTICLES_FILE = "articles.csv"
SENTENCE_FILE = "sentence_sentiment_results.csv"

# -----------------------------
# Load article metadata
# -----------------------------
articles = pd.read_csv(ARTICLES_FILE)

# -----------------------------
# Assign article type
# -----------------------------
# Based on the order of articles in articles.csv:
# First 15 = Scientific
# Last 15 = News/Media

articles["article_type"] = [
    "Scientific" if i < 15 else "News/Media"
    for i in range(len(articles))
]

# -----------------------------
# Calculate word count
# -----------------------------
# text_file already contains paths like:
# articles/002.txt
# articles/003.txt
# etc.

word_counts = []

for _, row in articles.iterrows():

    text_path = Path(row["text_file"])

    try:
        text = text_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        word_counts.append(word_count)

    except Exception as e:
        print(f"Could not read {text_path}: {e}")
        word_counts.append(None)

articles["word_count"] = word_counts

# -----------------------------
# Check for missing word counts
# -----------------------------
if articles["word_count"].isna().any():
    print("\nWARNING: Some articles could not be read.")

    print(
        articles.loc[
            articles["word_count"].isna(),
            ["article_id", "text_file"]
        ]
    )

# -----------------------------
# Word count by article
# -----------------------------
print("\n==============================")
print("WORD COUNT BY ARTICLE")
print("==============================")

print(
    articles[
        ["article_id", "article_type", "title", "word_count"]
    ].to_string(index=False)
)

# -----------------------------
# Word count summary
# -----------------------------
print("\n==============================")
print("WORD COUNT SUMMARY")
print("==============================")

word_summary = (
    articles
    .groupby("article_type")["word_count"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max"
    )
)

print(word_summary)

# -----------------------------
# Word count boxplot
# -----------------------------
scientific_words = articles.loc[
    articles["article_type"] == "Scientific",
    "word_count"
].dropna()

news_words = articles.loc[
    articles["article_type"] == "News/Media",
    "word_count"
].dropna()

plt.figure(figsize=(7, 5))

plt.boxplot(
    [scientific_words, news_words],
    tick_labels=["Scientific", "News/Media"]
)

# Individual article points
plt.scatter(
    [1] * len(scientific_words),
    scientific_words,
    alpha=0.7
)

plt.scatter(
    [2] * len(news_words),
    news_words,
    alpha=0.7
)

plt.ylabel("Word Count")
plt.title("Article Word Count by Article Type")
plt.tight_layout()

plt.savefig(
    "article_word_count_boxplot.png",
    dpi=300
)

plt.show()

# -----------------------------
# Load sentence-level results
# -----------------------------
sentences = pd.read_csv(SENTENCE_FILE)

# -----------------------------
# Count sentences per article
# -----------------------------
sentence_counts = (
    sentences
    .groupby("article_id")
    .size()
    .reset_index(name="sentence_count")
)

# Add sentence counts to article dataframe
articles = articles.merge(
    sentence_counts,
    on="article_id",
    how="left"
)

# -----------------------------
# Sentence count by article
# -----------------------------
print("\n==============================")
print("SENTENCE COUNT BY ARTICLE")
print("==============================")

print(
    articles[
        ["article_id", "article_type", "sentence_count"]
    ].to_string(index=False)
)

# -----------------------------
# Sentence count summary
# -----------------------------
print("\n==============================")
print("SENTENCE COUNT SUMMARY")
print("==============================")

sentence_summary = (
    articles
    .groupby("article_type")["sentence_count"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max"
    )
)

print(sentence_summary)

# -----------------------------
# Sentence count boxplot
# -----------------------------
scientific_sentences = articles.loc[
    articles["article_type"] == "Scientific",
    "sentence_count"
].dropna()

news_sentences = articles.loc[
    articles["article_type"] == "News/Media",
    "sentence_count"
].dropna()

plt.figure(figsize=(7, 5))

plt.boxplot(
    [scientific_sentences, news_sentences],
    tick_labels=["Scientific", "News/Media"]
)

# Individual article points
plt.scatter(
    [1] * len(scientific_sentences),
    scientific_sentences,
    alpha=0.7
)

plt.scatter(
    [2] * len(news_sentences),
    news_sentences,
    alpha=0.7
)

plt.ylabel("Sentence Count")
plt.title("Sentence Count by Article Type")
plt.tight_layout()

plt.savefig(
    "article_sentence_count_boxplot.png",
    dpi=300
)

plt.show()

# -----------------------------
# Finished
# -----------------------------
print("\nSaved:")
print("  article_word_count_boxplot.png")
print("  article_sentence_count_boxplot.png")
