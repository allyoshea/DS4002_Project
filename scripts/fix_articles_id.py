import pandas as pd
from pathlib import Path

# Load metadata
df = pd.read_csv("articles.csv")

# Get article number directly from the actual text filename
df["article_id"] = df["text_file"].apply(
    lambda x: int(Path(x).stem)
)

# Assign the correct group based on the filename number
df["article_type"] = df["article_id"].apply(
    lambda x: "Scientific" if 1 <= x <= 15
    else "News/Media" if 16 <= x <= 30
    else None
)

# Check for unexpected files
if df["article_type"].isna().any():
    print("WARNING: Some files have article numbers outside 001–030:")
    print(df[df["article_type"].isna()][["article_id", "text_file"]])

# Save corrected metadata
df.to_csv("articles_fixed.csv", index=False)

print("\nSaved: articles_fixed.csv")
print("\nArticle counts:")
print(df.groupby("article_type")["article_id"].nunique())

print("\nRows per article:")
print(df.groupby("article_id").size())