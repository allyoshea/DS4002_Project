import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
SENTIMENT_FILE = Path("output/vader_sentiment_data.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load VADER results
df = pd.read_csv(SENTIMENT_FILE)

# Classify articles
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media"
)

article_order = ["Scientific", "News/Media"]

# Print basic dataset information
print("\nVADER EDA")
print("=" * 40)

print(f"Articles: {len(df)}")
print(f"Scientific: {(df['article_type'] == 'Scientific').sum()}")
print(f"News/Media: {(df['article_type'] == 'News/Media').sum()}")

print("\nSentence counts:")
print(df.groupby("article_type")["sentences"].describe().round(2))

# Summary statistics for VADER scores
score_columns = [
    "mean_neg",
    "mean_neu",
    "mean_pos",
    "mean_compound"
]

summary = (
    df.groupby("article_type")[score_columns]
    .agg(["mean", "std", "median"])
    .reindex(article_order)
)

print("\nVADER score summary:")
print(summary.round(3))

# Article-level score distributions
for column, label in [
    ("mean_pos", "Positive"),
    ("mean_neg", "Negative"),
    ("mean_neu", "Neutral"),
    ("mean_compound", "Compound")
]:

    plt.figure(figsize=(5.5, 4.2))

    data = [
        df.loc[df["article_type"] == group, column]
        for group in article_order
    ]

    plt.boxplot(
        data,
        tick_labels=article_order,
        widths=0.45
    )

    plt.ylabel(f"Mean VADER {label} Score", fontsize=11)
    plt.xlabel("Article Type", fontsize=11)
    plt.title(f"Distribution of VADER {label} Scores", fontsize=12, pad=10)

    plt.tick_params(axis="both", labelsize=10)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"vader_{column}_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Sentence count distribution
plt.figure(figsize=(5.5, 4.2))

for group in article_order:
    values = df.loc[df["article_type"] == group, "sentences"]

    plt.hist(
        values,
        bins=15,
        alpha=0.55,
        label=group,
        edgecolor="black",
        linewidth=0.6
    )

plt.xlabel("Number of Sentences", fontsize=11)
plt.ylabel("Number of Articles", fontsize=11)
plt.title("Article Length Distribution", fontsize=12, pad=10)

plt.legend(frameon=False, fontsize=9)
plt.tick_params(axis="both", labelsize=10)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_article_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Compound score by article
plt.figure(figsize=(6, 4.2))

for group in article_order:
    subset = df[df["article_type"] == group]

    x = subset["article_id"]
    y = subset["mean_compound"]

    plt.scatter(
        x,
        y,
        label=group,
        s=28,
        edgecolor="black",
        linewidth=0.5
    )

plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel("Article ID", fontsize=11)
plt.ylabel("Mean VADER Compound Score", fontsize=11)
plt.title("Article-Level VADER Compound Scores", fontsize=12, pad=10)

plt.ylim(-1, 1)
plt.tick_params(axis="both", labelsize=10)

plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_compound_by_article.png",
    dpi=300,
    bbox_inches="tight"
)
# Mean compound score by article type
compound_summary = (
    df.groupby("article_type")["mean_compound"]
    .agg(["mean", "std", "count"])
    .reindex(article_order)
)

# Calculate standard error of the mean
compound_summary["sem"] = (
    compound_summary["std"] / compound_summary["count"] ** 0.5
)

plt.figure(figsize=(5.5, 4.2))

plt.bar(
    article_order,
    compound_summary["mean"],
    width=0.42,
    yerr=compound_summary["sem"],
    capsize=5,
    error_kw={"elinewidth": 1.2, "capthick": 1.2},
    color=["#7FA6C9", "#D98FA3"],
    edgecolor="black",
    linewidth=0.8
)

plt.xlabel("Article Type", fontsize=11)
plt.ylabel("Mean VADER Compound Score", fontsize=11)
plt.title("Mean VADER Compound Score by Article Type", fontsize=12, pad=10)

# Zoom in on the observed means
plt.ylim(0, 0.14)

plt.tick_params(axis="both", labelsize=10)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_mean_compound_by_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
plt.close()

plt.close()

print("\nEDA figures saved to output/")