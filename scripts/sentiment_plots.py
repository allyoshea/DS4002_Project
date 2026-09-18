import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load data
df = pd.read_csv("sentiment_data.csv")


# Plot 1: Scientific vs. News/Media
summary = df.groupby("article_type")["compound"].agg(
    ["mean", "std", "count"]
)

article_order = ["Scientific", "News/Media"]
summary = summary.reindex(article_order)

fig, ax = plt.subplots(figsize=(3.8, 3.5))

x = np.array([0, 0.45])

ax.bar(
    x,
    summary["mean"],
    width=0.28,
    yerr=summary["std"],
    capsize=4,
    color=["#0072B2", "#CC79A7"],
    edgecolor="none",
    error_kw={
        "elinewidth": 1.2,
        "capthick": 1.2
    }
)

ax.set_xticks(x)
ax.set_xticklabels(summary.index)
ax.set_xlim(-0.25, 0.70)

ax.set_ylabel(
    "Mean VADER Compound Score",
    fontsize=10
)

ax.set_xlabel(
    "Article Type",
    fontsize=10
)

ax.set_title(
    "Sentiment by Article Type",
    fontsize=12,
    fontweight="bold"
)

for i, row in enumerate(summary.itertuples()):
    ax.text(
        x[i],
        row.mean + row.std + 0.008,
        f"{row.mean:.3f}\n(n={row.count})",
        ha="center",
        fontsize=8,
        fontweight="bold"
    )

ax.set_ylim(
    0,
    max(summary["mean"] + summary["std"]) + 0.08
)

plt.tight_layout()

fig.savefig(
    "sentiment_by_article_type_errorbars.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# Plot 2: Scientific articles by journal
scientific = df[
    df["article_type"] == "Scientific"
].copy()


def classify_journal(source):
    source = str(source).lower()

    if "frontiers" in source:
        return "Frontiers"
    elif "nature" in source:
        return "Nature"
    elif (
        "iop" in source
        or "journal of neural engineering" in source
    ):
        return "JNE"
    else:
        return "Other"


scientific["journal_group"] = scientific[
    "source"
].apply(classify_journal)

print("\nScientific articles by journal")
print(
    scientific["journal_group"].value_counts()
)

journal_summary = scientific.groupby(
    "journal_group"
)["compound"].agg(
    ["mean", "std", "count"]
)

journal_order = [
    "Nature",
    "Frontiers",
    "JNE"
]

journal_summary = journal_summary.reindex(
    journal_order
)

fig, ax = plt.subplots(figsize=(4.5, 3.5))

x = np.array([0, 0.45, 0.90])

ax.bar(
    x,
    journal_summary["mean"],
    width=0.28,
    yerr=journal_summary["std"],
    capsize=4,
    color=[
        "#56B4E9",
        "#4A90C2",
        "#8EC9E8"
    ],
    edgecolor="none",
    error_kw={
        "elinewidth": 1.2,
        "capthick": 1.2
    }
)

ax.set_xticks(x)
ax.set_xticklabels(journal_summary.index)
ax.set_xlim(-0.25, 1.15)

ax.set_ylabel(
    "Mean VADER Compound Score",
    fontsize=10
)

ax.set_xlabel(
    "Scientific Publication",
    fontsize=10
)

ax.set_title(
    "Sentiment Across Scientific Publications",
    fontsize=12,
    fontweight="bold"
)

for i, row in enumerate(journal_summary.itertuples()):
    ax.text(
        x[i],
        row.mean + row.std + 0.008,
        f"{row.mean:.3f}\n(n={row.count})",
        ha="center",
        fontsize=8,
        fontweight="bold"
    )

ax.set_ylim(
    0,
    max(
        journal_summary["mean"]
        + journal_summary["std"]
    ) + 0.08
)

plt.tight_layout()

fig.savefig(
    "scientific_sentiment_by_journal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# Plot 3: Individual article sentiment
fig, ax = plt.subplots(figsize=(4.5, 3.8))

scientific_scores = df[
    df["article_type"] == "Scientific"
]["compound"]

news_scores = df[
    df["article_type"] == "News/Media"
]["compound"]

np.random.seed(42)

scientific_x = np.random.normal(
    0,
    0.035,
    len(scientific_scores)
)

news_x = np.random.normal(
    1,
    0.035,
    len(news_scores)
)

ax.scatter(
    scientific_x,
    scientific_scores,
    s=35,
    color="#0072B2",
    alpha=0.8
)

ax.scatter(
    news_x,
    news_scores,
    s=35,
    color="#CC79A7",
    alpha=0.8
)

ax.scatter(
    0,
    scientific_scores.mean(),
    s=100,
    facecolors="none",
    edgecolors="#0072B2",
    linewidths=2
)

ax.scatter(
    1,
    news_scores.mean(),
    s=100,
    facecolors="none",
    edgecolors="#CC79A7",
    linewidths=2
)

ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Scientific", "News/Media"]
)

ax.set_xlim(-0.35, 1.35)

ax.set_ylabel(
    "VADER Compound Score",
    fontsize=10
)

ax.set_xlabel(
    "Article Type",
    fontsize=10
)

ax.set_title(
    "Article-Level Sentiment",
    fontsize=12,
    fontweight="bold"
)

plt.tight_layout()

fig.savefig(
    "individual_article_sentiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# Plot 4: Sentiment by publication year

df["year"] = pd.to_datetime(
    df["date"],
    errors="coerce"
).dt.year

print("\nPublication years")
print(df["year"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(4.5, 3.5))

scientific_year = df[
    df["article_type"] == "Scientific"
].dropna(subset=["year"])

news_year = df[
    df["article_type"] == "News/Media"
].dropna(subset=["year"])

# Add a small amount of horizontal jitter
np.random.seed(42)

scientific_x = (
    scientific_year["year"].values
    + np.random.uniform(-0.08, 0.08, len(scientific_year))
)

news_x = (
    news_year["year"].values
    + np.random.uniform(-0.08, 0.08, len(news_year))
)

ax.scatter(
    scientific_x,
    scientific_year["compound"],
    s=45,
    color="#0072B2",
    alpha=0.8,
    label="Scientific"
)

ax.scatter(
    news_x,
    news_year["compound"],
    s=45,
    color="#CC79A7",
    alpha=0.8,
    label="News/Media"
)

# Clean year labels
years = sorted(df["year"].dropna().unique())

ax.set_xticks(years)
ax.set_xticklabels(
    [str(int(year)) for year in years]
)

ax.set_xlabel(
    "Publication Year",
    fontsize=10
)

ax.set_ylabel(
    "VADER Compound Score",
    fontsize=10
)

ax.set_title(
    "Sentiment by Publication Year",
    fontsize=12,
    fontweight="bold"
)

ax.legend(
    frameon=False,
    fontsize=9
)

plt.tight_layout()

fig.savefig(
    "sentiment_by_publication_year.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# PCA
sentiment_features = [
    "negative",
    "neutral",
    "positive",
    "compound"
]

X = df[sentiment_features].copy()

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(
    X_scaled
)

df["PC1"] = principal_components[:, 0]
df["PC2"] = principal_components[:, 1]

print("\nPCA explained variance")
print(
    f"PC1: {pca.explained_variance_ratio_[0] * 100:.1f}%"
)
print(
    f"PC2: {pca.explained_variance_ratio_[1] * 100:.1f}%"
)

fig, ax = plt.subplots(figsize=(5, 4))

scientific_pca = df[
    df["article_type"] == "Scientific"
]

news_pca = df[
    df["article_type"] == "News/Media"
]

ax.scatter(
    scientific_pca["PC1"],
    scientific_pca["PC2"],
    s=45,
    color="#0072B2",
    alpha=0.8,
    label="Scientific"
)

ax.scatter(
    news_pca["PC1"],
    news_pca["PC2"],
    s=45,
    color="#CC79A7",
    alpha=0.8,
    label="News/Media"
)

ax.axhline(
    0,
    linewidth=0.7,
    color="gray",
    alpha=0.4
)

ax.axvline(
    0,
    linewidth=0.7,
    color="gray",
    alpha=0.4
)

ax.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
    fontsize=10
)

ax.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
    fontsize=10
)

ax.set_title(
    "PCA of Sentiment Features",
    fontsize=12,
    fontweight="bold"
)

ax.legend(
    frameon=False,
    fontsize=9
)

plt.tight_layout()

fig.savefig(
    "pca_sentiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nAll plots saved:")
print("sentiment_by_article_type_errorbars.png")
print("scientific_sentiment_by_journal.png")
print("individual_article_sentiment.png")
print("sentiment_by_publication_year.png")
print("pca_sentiment.png")