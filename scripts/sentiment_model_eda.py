import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ==============================
# FILE
# ==============================

SENTENCE_FILE = "sentence_sentiment_results.csv"

# ==============================
# LOAD DATA
# ==============================

print("Loading sentence-level sentiment data...")

df = pd.read_csv(SENTENCE_FILE)

print(f"Loaded {len(df)} sentences.")

# ==============================
# ASSIGN ARTICLE TYPE
# ==============================

# Articles 001-015 = Scientific
# Articles 016-030 = News/Media

def assign_article_type(article_id):
    number = int(str(article_id).replace("article_", "").lstrip("0") or "0")

    if number <= 15:
        return "Scientific"
    else:
        return "News/Media"


df["article_type"] = df["article_id"].apply(assign_article_type)

# ==============================
# CALCULATE SENTENCE-LEVEL
# NET SENTIMENT
# ==============================

# Positive score - negative score
#
# Positive values = more positive
# Negative values = more negative
# Values near 0 = neutral/balanced

df["net_sentiment"] = (
    df["positive_score"] - df["negative_score"]
)

# ==============================
# ARTICLE-LEVEL SUMMARY
# ==============================

article_results = df.groupby(
    ["article_id", "article_type"]
).agg(
    mean_positive_score=("positive_score", "mean"),
    mean_negative_score=("negative_score", "mean"),
    mean_neutral_score=("neutral_score", "mean"),
    mean_net_sentiment=("net_sentiment", "mean"),
    positive_sentence_pct=(
        "sentiment",
        lambda x: (x == "positive").mean()
    ),
    negative_sentence_pct=(
        "sentiment",
        lambda x: (x == "negative").mean()
    ),
    neutral_sentence_pct=(
        "sentiment",
        lambda x: (x == "neutral").mean()
    ),
    number_of_sentences=("sentiment", "count")
).reset_index()

# ==============================
# PRINT ARTICLE RESULTS
# ==============================

print("\nARTICLE-LEVEL RESULTS")
print("-------------------------")

print(
    article_results.to_string(index=False)
)

# Save article-level results
article_results.to_csv(
    "scientific_model_article_scores.csv",
    index=False
)

# ==============================
# GROUP SUMMARY
# ==============================

print("\nGROUP SUMMARY")
print("-------------------------")

group_summary = article_results.groupby("article_type")[
    [
        "mean_positive_score",
        "mean_negative_score",
        "mean_net_sentiment",
        "positive_sentence_pct",
        "negative_sentence_pct",
        "neutral_sentence_pct"
    ]
].agg(["mean", "std", "median", "min", "max"])

print(group_summary)

group_summary.to_csv(
    "scientific_model_group_summary.csv"
)

# ==============================
# STATISTICAL TESTS
# ==============================

scientific = article_results[
    article_results["article_type"] == "Scientific"
]

news = article_results[
    article_results["article_type"] == "News/Media"
]

print("\nSTATISTICAL COMPARISONS")
print("-------------------------")

measures = {
    "Mean Positive Score": "mean_positive_score",
    "Mean Negative Score": "mean_negative_score",
    "Mean Net Sentiment": "mean_net_sentiment",
    "Positive Sentence %": "positive_sentence_pct",
    "Negative Sentence %": "negative_sentence_pct"
}

for name, column in measures.items():

    sci_values = scientific[column].dropna()
    news_values = news[column].dropna()

    t_stat, p_value = ttest_ind(
        sci_values,
        news_values,
        equal_var=False
    )

    difference = news_values.mean() - sci_values.mean()

    print(f"\n{name}")
    print(f"  Scientific mean: {sci_values.mean():.4f}")
    print(f"  News/Media mean: {news_values.mean():.4f}")
    print(f"  Difference (News - Scientific): {difference:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

# ==============================
# PLOT 1: NET SENTIMENT
# ==============================

plt.figure(figsize=(8, 6))

data_to_plot = [
    scientific["mean_net_sentiment"],
    news["mean_net_sentiment"]
]

plt.boxplot(
    data_to_plot,
    tick_labels=["Scientific", "News/Media"]
)

plt.ylabel("Mean Net Sentiment")
plt.title("Net Sentiment by Article Type")

plt.tight_layout()

plt.savefig(
    "net_sentiment_by_article_type.png",
    dpi=300
)

plt.show()

# ==============================
# PLOT 2: POSITIVE SCORE
# ==============================

plt.figure(figsize=(8, 6))

data_to_plot = [
    scientific["mean_positive_score"],
    news["mean_positive_score"]
]

plt.boxplot(
    data_to_plot,
    tick_labels=["Scientific", "News/Media"]
)

plt.ylabel("Mean Positive Score")
plt.title("Positive Sentiment Score by Article Type")

plt.tight_layout()

plt.savefig(
    "positive_score_by_article_type.png",
    dpi=300
)

plt.show()

# ==============================
# PLOT 3: NEGATIVE SCORE
# ==============================

plt.figure(figsize=(8, 6))

data_to_plot = [
    scientific["mean_negative_score"],
    news["mean_negative_score"]
]

plt.boxplot(
    data_to_plot,
    tick_labels=["Scientific", "News/Media"]
)

plt.ylabel("Mean Negative Score")
plt.title("Negative Sentiment Score by Article Type")

plt.tight_layout()

plt.savefig(
    "negative_score_by_article_type.png",
    dpi=300
)

plt.show()

# ==============================
# DONE
# ==============================

print("\n========================================")
print("DONE")
print("========================================")

print("Saved:")
print("  scientific_model_article_scores.csv")
print("  scientific_model_group_summary.csv")
print("  net_sentiment_by_article_type.png")
print("  positive_score_by_article_type.png")
print("  negative_score_by_article_type.png")