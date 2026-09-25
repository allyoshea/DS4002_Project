import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# File paths
# -----------------------------
ARTICLES_FILE = Path("data/articles.csv")
ARTICLES_DIR = Path("data/articles")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load article metadata
# -----------------------------
articles = pd.read_csv(ARTICLES_FILE)

# -----------------------------
# Classify article type
# -----------------------------
scientific_sources = {
    "www.nature.com",
    "www.frontiersin.org",
    "iopscience.iop.org"
}

articles["article_type"] = articles["source"].apply(
    lambda source: (
        "Scientific"
        if source in scientific_sources
        else "News/Media"
    )
)

# -----------------------------
# Count words in article text
# -----------------------------
def count_words(text):
    """Count words in an article."""
    if not isinstance(text, str):
        return 0

    words = re.findall(r"\b[\w'-]+\b", text)
    return len(words)


word_counts = []

for _, row in articles.iterrows():

    # CSV paths look like:
    # articles\006.txt
    #
    # Convert to:
    # 006.txt
    filename = Path(
        str(row["text_file"]).replace("\\", "/")
    ).name

    text_path = ARTICLES_DIR / filename

    if not text_path.exists():
        print(f"WARNING: File not found: {text_path}")
        word_counts.append(0)
        continue

    try:
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()

        word_counts.append(count_words(text))

    except Exception as error:
        print(f"ERROR reading {text_path}: {error}")
        word_counts.append(0)

# Add word counts to dataframe
articles["word_count"] = word_counts

# -----------------------------
# Remove articles where text
# could not be found
# -----------------------------
valid_articles = articles[articles["word_count"] > 0].copy()

# -----------------------------
# Print article counts
# -----------------------------
print("\n==============================")
print("ARTICLE COUNTS")
print("==============================")

print(
    valid_articles["article_type"].value_counts()
)

# -----------------------------
# Word count summary
# -----------------------------
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

print("\n==============================")
print("WORD COUNT SUMMARY")
print("==============================")

print(summary.round(2))

# -----------------------------
# Save results
# -----------------------------
results_file = OUTPUT_DIR / "article_length_results.csv"

valid_articles.to_csv(
    results_file,
    index=False
)

print(f"\nResults saved to: {results_file}")

# -----------------------------
# Prepare data for plot
# -----------------------------
scientific = valid_articles.loc[
    valid_articles["article_type"] == "Scientific",
    "word_count"
]

news = valid_articles.loc[
    valid_articles["article_type"] == "News/Media",
    "word_count"
]

# -----------------------------
# Create boxplot
# -----------------------------
plt.figure(figsize=(8, 6))

plt.boxplot(
    [scientific, news],
    tick_labels=["Scientific", "News/Media"]
)

# Add individual article points
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
plt.title("Article Length: Scientific vs. News/Media")

plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
figure_file = OUTPUT_DIR / "article_length_comparison.png"

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to: {figure_file}")

plt.show()