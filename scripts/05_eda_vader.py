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

#design input file and folder and where the generated data and figures go. 
SENTIMENT_FILE = Path("data/vader_sentiment_data.csv")
OUTPUT_DIR = Path("output")
#make a folder labeled output if it does not alrready exist 
OUTPUT_DIR.mkdir(exist_ok=True)


# Load the VADER article-level results.
df = pd.read_csv(SENTIMENT_FILE)

# The article IDs were assigned by dataset group:
# 001-030 are Scientific and 031-060 are News/Media.
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media"
)
# Assign category as scientific or news 
article_order = ["Scientific", "News/Media"]

# Print a basic overview of the dataset and the number of articles
# in each article type.
print("\nVADER EDA")

# Print the number of articles in the complete dataset and in each
# article type to verify that the articles were classified correctly.
print(f"Articles: {len(df)}")
print(
    f"Scientific: "
    f"{(df['article_type'] == 'Scientific').sum()}"
)
print(
    f"News/Media: "
    f"{(df['article_type'] == 'News/Media').sum()}"
)
# Calculate descriptive statistics for the number of sentences per article.
# This provides an initial comparison of article length between the groups.
print("\nSentence counts:")
print(
    df.groupby("article_type")["sentences"] #grouping by scientififc vs. news/media
    .describe()
    .round(2)
)


# Identify the VADER scores that will be included in the summary analysis.
# These include the negative, neutral, positive, and compound scores.
score_columns = [
    "mean_neg",
    "mean_neu",
    "mean_pos",
    "mean_compound"
]
# Calculate the mean, standard deviation, and median of each VADER score
# separately for Scientific and News/Media articles.
summary = (
    df.groupby("article_type")[score_columns]
    .agg(["mean", "std", "median"]) #CALCULATE basic meterics here
    .reindex(article_order)
)

print("\nVADER score summary:")
print(summary.round(3))


# Create a separate boxplot for each VADER sentiment measure.
# Boxplots allow the distributions of scores to be compared between
# Scientific and News/Media articles.
for column, label in [
    ("mean_pos", "Positive"),
    ("mean_neg", "Negative"),
    ("mean_neu", "Neutral"),
    ("mean_compound", "Compound")
]:
# Select the VADER score being analyzed and assign a readable label
# for the corresponding figure.
    plt.figure(figsize=(5.5, 4.2))
# Extract the article-level scores for each article type.
# The data are kept separate so the two distributions can be compared.
    data = [
        df.loc[
            df["article_type"] == group,
            column
        ]
        for group in article_order
    ]
# Use boxplots to display the median, spread, and potential outliers
# of the VADER scores for each article type.
    plt.boxplot(
        data,
        tick_labels=article_order,
        widths=0.45
    )
    # Add descriptive axis labels and a title identifying the VADER score
# shown in the current figure.

    plt.ylabel(
        f"Mean VADER {label} Score", #adding axes titles
        fontsize=11
    )

    plt.xlabel(
        "Article Type",
        fontsize=11
    )

    plt.title(
        f"Distribution of VADER {label} Scores",
        fontsize=12, #adding title
        pad=10
    )

    plt.tick_params(
        axis="both",
        labelsize=10
    )

    plt.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,  # Add horizontal reference lines to make differences in score values easier to visually compare.                                                     
        alpha=0.3
    )


    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
# Save a separate figure for each VADER score.
    plt.savefig(
        OUTPUT_DIR / f"vader_{column}_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Plot the sentence-count distribution for each article type on the
# same histogram so the two groups can be visually compared.
plt.figure(figsize=(5.5, 4.2))

for group in article_order:
# Select the sentence counts for the current article type.
    values = df.loc[
        df["article_type"] == group,
        "sentences"
    ]
   # Plot the distribution of sentence counts for the current article group.
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
# Save the article length comparison as a high-resolution figure.
plt.savefig(
    OUTPUT_DIR / "vader_article_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Plot the mean compound sentiment score for every individual article.
# This allows variation between individual articles to be examined
# rather than only comparing group averages.
plt.figure(figsize=(6, 4.2))
# Use consistent colors for Scientific and News/Media articles
# across the figures in all of this analysis for visual clarity. 
colors = {
    "Scientific": "#A8C7E8",
    "News/Media": "#F2B6C6"
}
# Plot each article type separately so the groups can be distinguished
# visually in the scatterplot.
for group in article_order:

    subset = df[
        df["article_type"] == group
    ]
# Use article ID to identify each article on the x-axis and its
# mean compound score as the y-axis measurement.
    x = subset["article_id"] # Use article ID for the x-axis and mean compound score
    # for the y-axis.
    y = subset["mean_compound"]
# Plot each article as an individual point.
# The color indicates whether the article is Scientific or News/Media.
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
# Add a horizontal line at zero to distinguish positive and negative
# compound sentiment scores.
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

plt.ylim(-1, 1) # VADER compound scores range from -1 to 1.
# Keeping the full range makes the direction and scale of the scores clear.

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


# Calculate the mean, standard deviation, and number of articles in each
# group to summarize the article-level compound scores.
compound_summary = (
    df.groupby("article_type")["mean_compound"]
    .agg(["mean", "std", "count"])
    .reindex(article_order)
)
# Calculate the standard error of the mean (SEM) for each article type.
# SEM estimates the uncertainty around the group mean.
compound_summary["sem"] = (
    compound_summary["std"]
    / compound_summary["count"] ** 0.5
)

# Create a bar chart comparing the average compound sentiment
# between Scientific and News/Media articles.
plt.figure(figsize=(5.5, 4.2))

plt.bar( # Plot the mean compound sentiment score for each article type,
# with error bars representing the standard error of the mean.
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
# Save the final comparison of mean compound scores as a high-resolution PNG.
plt.savefig(
    OUTPUT_DIR / "vader_mean_compound_by_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEDA figures saved to output/")
