"""
Script: 03_eda_sciBert.py

Purpose:
    Perform exploratory data analysis on the SciBERT sentiment results
    and generate figures comparing scientific and news/media articles.

Inputs:
    data/articles.csv
    data/sentiment_results.csv

Outputs:
    output/articles_by_type.png
    output/positive_pct_by_type.png
    output/negative_pct_by_type.png
    output/neutral_pct_by_type.png
    output/sentence_count_by_type.png
    output/sentiment_composition_stacked.png
    output/sentiment_eda_data.csv

Process:
    1. Load the article metadata and SciBERT sentiment results.
    2. Combine the two datasets using article_id.
    3. Classify articles as Scientific or News/Media based on article_id.
    4. Summarize the dataset by article type.
    5. Compare sentiment percentages between article types.
    6. Compare the number of sentences per article.
    7. Save the processed data and generated figures.

Notes:
    Articles 001-030 are classified as Scientific and articles 031-060
    are classified as News/Media.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


ARTICLES_FILE = Path("data/articles.csv")
SENTIMENT_FILE = Path("data/sentiment_results.csv")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)
# The article metadata contains information about each article,
# while the sentiment file contains the SciBERT analysis results.
# These files are combined below using the shared article ID.


# Load the article metadata and SciBERT sentiment results.
articles = pd.read_csv(ARTICLES_FILE)
sentiment = pd.read_csv(SENTIMENT_FILE)

# Combine the metadata and sentiment results using the article ID -- so can identify which category and characteristics of each sentiment result.
df = articles.merge(
    sentiment,
    on="article_id",
    how="inner"
)

# The article IDs were assigned by dataset group:
# 001-030 are Scientific and 031-060 are News/Media just did <30 for ease .
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media" #COUNTS FIRST 30 as scientific
)


print("\nDATASET OVERVIEW")
print(f"Number of articles: {len(df)}") #PRINT basic information about the dataset to help us get oriented

print("\nArticles by type:")
print(df["article_type"].value_counts()) #CHECK FOR MYSLEG make sure accurately getting 30 for both!!


# Compare the number of articles in each group.
type_counts = df["article_type"].value_counts()
type_counts = type_counts.reindex(
    ["Scientific", "News/Media"]
)

plt.figure(figsize=(7, 5))

plt.bar(
    type_counts.index,
    type_counts.values
)

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


# Calculate the mean and standard deviation of each sentiment category
# within the two article types.
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

    x = np.array([0, 0.45])

    plt.figure(figsize=(3.2, 2.8))

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        width=0.18,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw={
            "elinewidth": 1,
            "capthick": 1
        }
    )

    plt.xticks(
        x,
        article_order
    )

    plt.xlabel("Article Type")
    plt.ylabel("Avg. Percentage of Sentences (%)")
    plt.title(
    f"{label} Sentiment by Article Type",
    fontsize=10
)

    plt.ylim(bottom=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_by_type.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Compare the number of sentences contained in each article.
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


# Save the combined metadata and sentiment data for reference.
df.to_csv(
    OUTPUT_DIR / "sentiment_eda_data.csv",
    index=False
)


# Calculate the average percentage of positive, negative, and neutral
# sentences for each article type.
sentiment_summary = (
    df.groupby("article_type")[
        ["positive_pct", "negative_pct", "neutral_pct"]
    ]
    .mean()
    * 100
)

article_order = ["Scientific", "News/Media"]

sentiment_summary = sentiment_summary.reindex(
    article_order
)
print("\nAVERAGE SENTIMENT COMPOSITION (%)")
print(sentiment_summary)
positive = sentiment_summary["positive_pct"]
negative = sentiment_summary["negative_pct"]
neutral = sentiment_summary["neutral_pct"]


# Create a 100% stacked bar chart showing the average sentiment


# composition of each article type.
plt.figure(figsize=(5, 3.5)) #decrase size its taking up so much space

bar_width = 0.22
x = [0, 0.45] #makes bars closer together

positive_color = "#008C95"   # Teal
negative_color = "#7B2CBF"   # Purple
neutral_color = "#D9D9D9"    # Light gray #COLORS I THINK WILL POP

#GRAPHING COMPOSITION FIGURE, have to do eeach part of the bar separetly . . .  so many

plt.bar(
    x,
    positive,
    width=bar_width,
    label="Positive",
    color=positive_color,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x,
    negative,
    width=bar_width,
    bottom=positive,
    label="Negative",
    color=negative_color,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x,
    neutral,
    width=bar_width,
    bottom=positive + negative,
    label="Neutral",
    color=neutral_color,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5
)

plt.xlabel(
    "Article Type",
    fontsize=10
)

plt.ylabel(
    "Average Percentage of Sentences (%)",
    fontsize=10
)

plt.title(
    "Sentiment Composition by Article Type",
    fontsize=13,
    fontweight="bold",
    pad=10
)

plt.xticks(
    x,
    article_order,
    fontsize=10
)

plt.yticks(
    fontsize=9
)

plt.ylim(0, 100)

# Remove background gridlines.
plt.grid(False)

plt.legend(
    frameon=False,
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentiment_composition_stacked.png",
    dpi=300,
    bbox_inches="tight"
)
#PERFORM WELCH's t-test to see if there is signfiicance in the positive difference resutls
from scipy.stats import ttest_ind

# Compare article-level positive sentiment between article types.
scientific_positive = df.loc[
    df["article_type"] == "Scientific",
    "positive_pct"
]

news_positive = df.loc[
    df["article_type"] == "News/Media",
    "positive_pct"
]

t_stat, p_value = ttest_ind(
    scientific_positive,
    news_positive,
    equal_var=False
)

#PRINTING AT END TO SAVE EVERYTHING AND ENSURE NO ERRORS -- peace of mind knowing what was saved and printed!!
print("\nPOSITIVE SENTIMENT T-TEST")
print(f"Scientific mean: {scientific_positive.mean() * 100:.2f}%")
print(f"News/Media mean: {news_positive.mean() * 100:.2f}%")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3e}")
plt.close()
print("\nSaved plots:")
print("articles_by_type.png")
print("positive_pct_by_type.png")
print("negative_pct_by_type.png")
print("neutral_pct_by_type.png")
print("sentence_count_by_type.png")
print("sentiment_composition_stacked.png")
print("\nAll files saved to the output/ folder.")
