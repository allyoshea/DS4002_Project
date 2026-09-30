"""
Script: 05_eda_vader.py

Purpose:
    Perform exploratory data analysis on the VADER sentiment results
    and generate figures comparing scientific and news/media articles.

Input:
    data/vader_sentiment_data.csv

Outputs:
    output/vader_mean_pos_distribution.png
    output/vader_mean_neg_distribution.png
    output/vader_mean_neu_distribution.png
    output/vader_mean_compound_distribution.png
    output/vader_article_length_distribution.png
    output/vader_compound_by_article.png
    output/vader_mean_compound_by_type.png

Process:
    1. Load the VADER sentiment results.
    2. Classify articles as Scientific or News/Media based on article_id.
    3. Calculate summary statistics for VADER sentiment scores.
    4. Compare score distributions between article types.
    5. Compare article lengths between article types.
    6. Plot compound scores for individual articles.
    7. Compare mean compound scores between article types.

Notes:
    VADER produces negative, neutral, positive, and compound scores.
    Article-level scores represent the mean score across all sentences
    within each article.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#degine input file and folder and where the generated data and figures go. 
SENTIMENT_FILE = Path("data/vader_sentiment_data.csv")
OUTPUT_DIR = Path("output")
#make a afodler lapebeled output if it does not alr exist 
OUTPUT_DIR.mkdir(exist_ok=True)


# Load the VADER article-level results.
df = pd.read_csv(SENTIMENT_FILE)

# The article IDs were assigned by dataset group:
# 001-030 are Scientific and 031-060 are News/Media.
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media"
)
#PRESERBVE category as scientific or news 
article_order = ["Scientific", "News/Media"]

# Print a basic overview of the dataset and the number of articles
# in each article type.
print("\nVADER EDA")


print(f"Articles: {len(df)}")
print(
    f"Scientific: "
    f"{(df['article_type'] == 'Scientific').sum()}"
)
print(
    f"News/Media: "
    f"{(df['article_type'] == 'News/Media').sum()}"
)

print("\nSentence counts:")
print(
    df.groupby("article_type")["sentences"]
    .describe()
    .round(2)
)


# Calculate summary statistics for the VADER scores by article type.
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


# Compare the distribution of each VADER score between article types.
for column, label in [
    ("mean_pos", "Positive"),
    ("mean_neg", "Negative"),
    ("mean_neu", "Neutral"),
    ("mean_compound", "Compound")
]:

    plt.figure(figsize=(5.5, 4.2))

    data = [
        df.loc[
            df["article_type"] == group,
            column
        ]
        for group in article_order
    ]

    plt.boxplot(
        data,
        tick_labels=article_order,
        widths=0.45
    )

    plt.ylabel(
        f"Mean VADER {label} Score",
        fontsize=11
    )

    plt.xlabel(
        "Article Type",
        fontsize=11
    )

    plt.title(
        f"Distribution of VADER {label} Scores",
        fontsize=12,
        pad=10
    )

    plt.tick_params(
        axis="both",
        labelsize=10
    )

    plt.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.3
    )


    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"vader_{column}_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Compare the distribution of article lengths between the two groups.
plt.figure(figsize=(5.5, 4.2))

for group in article_order:

    values = df.loc[
        df["article_type"] == group,
        "sentences"
    ]

    plt.hist(
        values,
        bins=15,
        alpha=0.55,
        label=group,
        edgecolor="black",
        linewidth=0.6
    )

plt.xlabel(
    "Number of Sentences",
    fontsize=11
)

plt.ylabel(
    "Number of Articles",
    fontsize=11
)

plt.title(
    "Article Length Distribution",
    fontsize=12,
    pad=10
)

plt.legend(
    frameon=False,
    fontsize=9
)

plt.tick_params(
    axis="both",
    labelsize=10
)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.3
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_article_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Plot the mean compound score for each individual article.
plt.figure(figsize=(6, 4.2))

colors = {
    "Scientific": "#A8C7E8",
    "News/Media": "#F2B6C6"
}

for group in article_order:

    subset = df[
        df["article_type"] == group
    ]

    x = subset["article_id"]
    y = subset["mean_compound"]

    plt.scatter(
        x,
        y,
        color=colors[group],
        label=group,
        s=28,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8
    )

plt.axhline(
    0,
    color="black",
    linewidth=0.8
)

plt.xlabel(
    "Article ID",
    fontsize=11
)

plt.ylabel(
    "Mean VADER Compound Score",
    fontsize=11
)

plt.title(
    "Article-Level VADER Compound Scores",
    fontsize=12,
    pad=10
)

plt.ylim(-1, 1)

plt.tick_params(
    axis="both",
    labelsize=10
)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.3
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_compound_by_article.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Calculate the mean compound score and standard error for each group.
compound_summary = (
    df.groupby("article_type")["mean_compound"]
    .agg(["mean", "std", "count"])
    .reindex(article_order)
)

compound_summary["sem"] = (
    compound_summary["std"]
    / compound_summary["count"] ** 0.5
)


plt.figure(figsize=(5.5, 4.2))

plt.bar(
    article_order,
    compound_summary["mean"],
    width=0.42,
    yerr=compound_summary["sem"],
    capsize=5,
    error_kw={
        "elinewidth": 1.2,
        "capthick": 1.2
    },
    color=["#A8C7E8", "#F2B6C6"],
    edgecolor="black",
    linewidth=0.8
)

plt.xlabel(
    "Article Type",
    fontsize=11
)

plt.ylabel(
    "Mean VADER Compound Score",
    fontsize=11
)

plt.title(
    "Mean VADER Compound Score by Article Type",
    fontsize=12,
    pad=10
)

# The observed scores are close to zero, so this range makes
# differences between the group means easier to see.
plt.ylim(0, 0.14)

plt.tick_params(
    axis="both",
    labelsize=10
)

plt.grid(
    False,
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.3
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vader_mean_compound_by_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEDA figures saved to output/")
