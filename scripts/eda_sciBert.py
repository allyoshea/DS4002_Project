import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# File paths
ARTICLES_FILE = Path("data/articles.csv")
SENTIMENT_FILE = Path("output/sentiment_results.csv")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
articles = pd.read_csv(ARTICLES_FILE)
sentiment = pd.read_csv(SENTIMENT_FILE)

# Combine article metadata and sentiment results
df = articles.merge(
    sentiment,
    on="article_id",
    how="inner"
)

# Classify articles
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media"
)

print("\nDATASET OVERVIEW")
print("----------------")
print(f"Number of articles: {len(df)}")

print("\nArticles by type:")
print(df["article_type"].value_counts())

# Plot 1: Number of articles by type
type_counts = df["article_type"].value_counts()
type_counts = type_counts.reindex(["Scientific", "News/Media"])

plt.figure(figsize=(7, 5))
plt.bar(type_counts.index, type_counts.values)
plt.xlabel("Article Type")
plt.ylabel("Number of Articles")
plt.title("Number of Articles by Type")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "articles_by_type.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# Plot sentiment separately with error bars
sentiment_columns = {
    "positive_pct": "Positive",
    "negative_pct": "Negative",
    "neutral_pct": "Neutral"
}

colors = ["#A8C7E8", "#F2B6C6"]
article_order = ["Scientific", "News/Media"]

for column, label in sentiment_columns.items():

    summary = (
        df.groupby("article_type")[column]
        .agg(["mean", "std"])
        .reindex(article_order)
        * 100
    )

    x = np.arange(len(article_order))

    plt.figure(figsize=(5.5, 4.5))

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        width=0.4,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw={
            "elinewidth": 1,
            "capthick": 1
        }
    )

    plt.xticks(x, article_order)
    plt.xlabel("Article Type")
    plt.ylabel("Average Percentage of Sentences (%)")
    plt.title(f"{label} Sentiment by Article Type")

    # Keep percentage values above zero
    plt.ylim(bottom=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_by_type.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# Plot 3: Number of sentences per article
sentence_counts = [
    df[df["article_type"] == "Scientific"]["sentences"],
    df[df["article_type"] == "News/Media"]["sentences"]
]

plt.figure(figsize=(7, 5))
plt.boxplot(
    sentence_counts,
    tick_labels=["Scientific", "News/Media"]
)
plt.xlabel("Article Type")
plt.ylabel("Number of Sentences")
plt.title("Number of Sentences per Article")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sentence_count_by_type.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Save combined data
df.to_csv(
    OUTPUT_DIR / "sentiment_eda_data.csv",
    index=False
)
# Plot sentiment composition as 100% stacked bars
sentiment_summary = (
    df.groupby("article_type")[
        ["positive_pct", "negative_pct", "neutral_pct"]
    ]
    .mean()
    * 100
)

article_order = ["Scientific", "News/Media"]
sentiment_summary = sentiment_summary.reindex(article_order)

positive = sentiment_summary["positive_pct"]
negative = sentiment_summary["negative_pct"]
neutral = sentiment_summary["neutral_pct"]

plt.figure(figsize=(6, 4.2))

bar_width = 0.42

plt.bar(
    article_order,
    positive,
    width=bar_width,
    label="Positive",
    color="#7FA6C9",
    edgecolor="black",
    linewidth=0.8
)

plt.bar(
    article_order,
    negative,
    width=bar_width,
    bottom=positive,
    label="Negative",
    color="#D98FA3",
    edgecolor="black",
    linewidth=0.8
)

plt.bar(
    article_order,
    neutral,
    width=bar_width,
    bottom=positive + negative,
    label="Neutral",
    color="#C7CED8",
    edgecolor="black",
    linewidth=0.8
)

plt.xlabel("Article Type", fontsize=11)
plt.ylabel("Percentage of Sentences (%)", fontsize=11)
plt.title("Sentiment Composition by Article Type", fontsize=12, pad=10)

plt.ylim(0, 100)

plt.tick_params(axis="both", labelsize=10)

plt.legend(
    frameon=False,
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
)

plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentiment_composition_stacked.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
print("\nSaved plots:")
print("articles_by_type.png")
print("sentiment_composition_by_type.png")
print("sentence_count_by_type.png")
print("\nAll files saved to the output/ folder.")