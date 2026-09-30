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

#DEFINE location of input folder and desired output folder 
ARTICLES_FILE = Path("data/articles.csv")
ARTICLES_DIR = Path("data/articles")
OUTPUT_DIR = Path("output")
#create folder if doesnt exist
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
#return 0 if article text is missing or is nto a string
    if not isinstance(text, str):
        return 0
#use a regular expression to identify words, words cotnainign apostrophes or hypens
    words = re.findall(
        r"\b[\w'-]+\b",
        text
    )
  # Return the total number of words identified in the article.
    return len(words)


word_counts = [] # Store the word count for each article in the same order
# as the rows in the metadata dataframe.

# Read each article and calculate its word count.
for _, row in articles.iterrows():

    # Extract the filename from the path stored in the metadata CSV.
    filename = Path(
        str(row["text_file"]).replace("\\", "/")
    ).name
    # Build the path to the corresponding article text file.
    text_path = ARTICLES_DIR / filename
 #If the article file cannot be found, record a word count of zero
# and continue to the next article.
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
#COUNT words in article and append the results
        word_counts.append(
            count_words(text)
        )

    # Report files that cannot be read and assign a word count of zero.
    except Exception as error:

        print(
            f"ERROR reading "
            f"{text_path}: {error}"
        )

        word_counts.append(0)


# Add the word counts to the article metadata.
articles["word_count"] = word_counts


# Exclude articles whose text could not be read or had 0 counts
valid_articles = articles[
    articles["word_count"] > 0
].copy()


print("\nARTICLE COUNTS")

# Display the number of valid articles in each dataset group.
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

print(
    summary.round(2)
)


# Save the article-level word counts and metadata.
results_file = (
    OUTPUT_DIR / "article_length_results.csv" #NOW results can be used in later analysis and graph making 
)

valid_articles.to_csv(
    results_file,
    index=False
)

print(
    f"\nResults saved to: "
    f"{results_file}"
)

# Separate the word counts into the two article groups
# for comparison in the boxplot.
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

# The boxplot shows the distribution of word counts within each group.
plt.boxplot(
    [scientific, news],
    tick_labels=["Scientific", "News/Media"]
)
# Add individual article points to show the actual variation
plt.scatter(
    [1] * len(scientific),
    scientific,
    alpha=0.6 #keep in mine word cound
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
#Save figure
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
