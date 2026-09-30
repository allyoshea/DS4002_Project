"""
Script: 06_article_length_analysis.py

Purpose:
    Calculate article word counts and compare article length between
    scientific and news/media articles.

Inputs:
    data/articles.csv
    data/articles/*.txt

Outputs:
    data/article_length_results.csv
    output/article_length_comparison.png

Process:
    1. Load the article metadata.
    2. Classify articles as Scientific or News/Media based on article_id.
    3. Read the text file associated with each article.
    4. Count the words in each article.
    5. Calculate summary statistics for article length by article type.
    6. Save the word counts and article metadata as a CSV file.
    7. Generate a boxplot comparing article lengths.

Notes:
    Articles 001-030 are classified as Scientific and articles 031-060
    are classified as News/Media.
"""

import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt


ARTICLES_FILE = Path("data/articles.csv")
ARTICLES_DIR = Path("data/articles")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)


# Load the article metadata.
articles = pd.read_csv(ARTICLES_FILE)

# The article IDs were assigned by dataset group:
# 001-030 are Scientific and 031-060 are News/Media.
articles["article_type"] = articles["article_id"].apply(
    lambda x: "Scientific" if x <= 30 else "News/Media"
)


def count_words(text):
    """Count words in an article."""

    if not isinstance(text, str):
        return 0

    words = re.findall(
        r"\b[\w'-]+\b",
        text
    )

    return len(words)


word_counts = []

# Read each article and calculate its word count.
for _, row in articles.iterrows():

    # Extract the filename from the path stored in the metadata CSV.
    filename = Path(
        str(row["text_file"]).replace("\\", "/")
    ).name

    text_path = ARTICLES_DIR / filename

    if not text_path.exists():

        print(
            f"WARNING: File not found: "
            f"{text_path}"
        )

        word_counts.append(0)
        continue

    try:

        with open(
            text_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read()

        word_counts.append(
            count_words(text)
        )

    except Exception as error:

        print(
            f"ERROR reading "
            f"{text_path}: {error}"
        )

        word_counts.append(0)


# Add the word counts to the article metadata.
articles["word_count"] = word_counts


# Exclude articles whose text could not be read.
valid_articles = articles[
    articles["word_count"] > 0
].copy()


print("\nARTICLE COUNTS")
print("----------------")

print(
    valid_articles["article_type"].value_counts()
)


# Calculate word count summary statistics for each article type.
summary = (
    valid_articles
    .groupby("article_type")["word_count"]
    .agg([
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max"
    ])
)

print("\nWORD COUNT SUMMARY")
print("------------------")

print(
    summary.round(2)
)


# Save the article-level word counts and metadata.
results_file = (
    OUTPUT_DIR / "article_length_results.csv"
)

valid_articles.to_csv(
    results_file,
    index=False
)

print(
    f"\nResults saved to: "
    f"{results_file}"
)


scientific = valid_articles.loc[
    valid_articles["article_type"] == "Scientific",
    "word_count"
]

news = valid_articles.loc[
    valid_articles["article_type"] == "News/Media",
    "word_count"
]


# Compare article lengths using a boxplot and individual article points.
plt.figure(figsize=(8, 6))

plt.boxplot(
    [scientific, news],
    tick_labels=["Scientific", "News/Media"]
)

plt.scatter(
    [1] * len(scientific),
    scientific,
    alpha=0.6
)

plt.scatter(
    [2] * len(news),
    news,
    alpha=0.6
)

plt.xlabel("Article Type")
plt.ylabel("Word Count")
plt.title(
    "Article Length: Scientific vs. News/Media"
)

plt.tight_layout()


figure_file = (
    OUTPUT_DIR / "article_length_comparison.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Figure saved to: "
    f"{figure_file}"
)

plt.show()
