
import pandas as pd

articles = pd.read_csv("articles.csv")

# Get the number from filenames like articles/001.txt
articles["article_number"] = (
    articles["text_file"]
    .str.extract(r"(\d+)\.txt$")[0]
    .astype(int)
)

# Assign expected group
articles["article_type"] = articles["article_number"].apply(
    lambda x: "Scientific" if 1 <= x <= 15 else "News/Media"
)

# Print sorted assignment
print(
    articles[
        ["article_id", "text_file", "article_number", "article_type"]
    ]
    .sort_values("article_number")
    .to_string(index=False)
)

# Check counts
print("\nCOUNTS:")
print(articles["article_type"].value_counts())

# Check for missing/duplicate article numbers
print("\nARTICLE NUMBERS:")
print(sorted(articles["article_number"].tolist()))

