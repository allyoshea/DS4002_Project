import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# File locations

CSV_FILE = "articles.csv"
ARTICLE_DIR = Path("articles")


# Load metadata


df = pd.read_csv(CSV_FILE)
df["article_type"] = ["Scientific"] * 15 + ["News/Media"] * 15

print("\nDATASET OVERVIEW")
print("----------------")
print(f"Number of articles: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate URLs:")
print(df["url"].duplicated().sum())


# Load article text


word_counts = []

for _, row in df.iterrows():
    article_id = int(row["article_id"])
    filename = f"{article_id:03d}.txt"
    filepath = ARTICLE_DIR / filename

    if filepath.exists():
        text = filepath.read_text(encoding="utf-8")
        word_count = len(text.split())
    else:
        word_count = None
        print(f"WARNING: Missing file {filename}")

    word_counts.append(word_count)

df["word_count"] = word_counts


# Word count summary


print("\nWORD COUNT SUMMARY")
print("------------------")
print(df["word_count"].describe())


# Save updated EDA data


df.to_csv("eda_data.csv", index=False)

print("\nSaved: eda_data.csv")


# Plot 1: Articles by source


# Plot 1: Number of articles by type
type_counts = df["article_type"].value_counts()

plt.figure(figsize=(7, 5))
type_counts.plot(kind="bar")
plt.title("Number of Articles by Type")
plt.xlabel("Article Type")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("articles_by_type.png", dpi=300)
plt.close()


# Plot 2: Word count by article type
scientific_words = df[df["article_type"] == "Scientific"]["word_count"]
news_words = df[df["article_type"] == "News/Media"]["word_count"]

plt.figure(figsize=(7, 5))
plt.boxplot(
    [scientific_words, news_words],
    tick_labels=["Scientific", "News/Media"]
)
plt.title("Article Word Count by Type")
plt.xlabel("Article Type")
plt.ylabel("Word Count")
plt.tight_layout()
plt.savefig("word_count_by_type.png", dpi=300)
plt.close()

print("\nSaved plots:")
print("- articles_by_type.png")
print("- word_count_by_type.png")
