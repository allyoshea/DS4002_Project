
import pandas as pd

# Load sentiment results
df = pd.read_csv("sentiment_data.csv")

# Compare sentiment between article types
print("\nSENTIMENT BY ARTICLE TYPE")
print("-------------------------")

summary = df.groupby("article_type")["compound"].agg(
    ["mean", "std", "min", "max"]
)

print(summary)

