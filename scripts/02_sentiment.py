"""
Script: 02_sentiment.py

Purpose:
    Perform sentence-level sentiment analysis on the collected BCI
    articles using the SciBERT sentiment classification model.

Input:
    data/articles/*.txt

Outputs:
    data/sentiment_results.csv
        Article-level sentiment results.

    data/sentence_sentiment_results.csv
        Sentiment scores and classifications for individual sentences.

Process:
    1. Load the pretrained scientific sentiment model.
    2. Read each article from data/articles/.
    3. Split each article into sentences.
    4. Classify each sentence as positive, negative, or neutral.
    5. Calculate the proportion of each sentiment for each article.
    6. Save article-level and sentence-level results as CSV files.

Notes:
    The model used is puzzz21/sci-sentiment-classify.
    Sentences shorter than 20 characters are excluded from analysis.
"""
import os
import re
import pandas as pd
from transformers import pipeline


# Scientific sentiment model used for the analysis and setting more input/output variables.
#Found documentation for downloading article online 
MODEL_NAME = "puzzz21/sci-sentiment-classify"
ARTICLE_DIR = "data/articles"
ARTICLE_OUTPUT_FILE = "data/sentiment_results.csv"
SENTENCE_OUTPUT_FILE = "data/sentence_sentiment_results.csv"

#Adding more text to terminal output for clarity to see where the script and model failed 

print("Loading scientific sentiment model...")

#
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None
)

print("Model loaded successfully!")


def split_sentences(text):
    """Split article text into sentences and remove very short sentences."""

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 20
    ]

    return sentences


def convert_label(label):
    """Convert the model's labels to positive, negative, or neutral."""

    label = label.lower()

    if label == "p":
        return "positive"
    elif label == "n":
        return "negative"
    elif label == "o":
        return "neutral"
    else:
        return label


def analyze_sentence(sentence):
    """Run the sentiment model on one sentence and return its scores."""

    results = classifier(
        sentence,
        truncation=True,
        max_length=512
    )[0]

    # Store the model probability for each sentiment category
    scores = {}

    for item in results:
        sentiment = convert_label(item["label"])
        scores[sentiment] = item["score"]

    # Assign the sentence the sentiment with the highest probability/score
    best_sentiment = max(
        scores,
        key=scores.get
    )

    return {
        "sentiment": best_sentiment,
        "confidence": scores[best_sentiment],
        "positive_score": scores.get("positive", 0),
        "negative_score": scores.get("negative", 0),
        "neutral_score": scores.get("neutral", 0)
    }


# Pulls everything together and runs analysis on articles in articles folder.
def main():

    if not os.path.exists(ARTICLE_DIR):
        print(
            f"ERROR: Could not find "
            f"'{ARTICLE_DIR}' folder."
        )
        return

    article_files = sorted(
        filename
        for filename in os.listdir(ARTICLE_DIR)
        if filename.endswith(".txt")
    )

    if not article_files:
        print("ERROR: No .txt articles found.")
        return

    print(
        f"\nFound {len(article_files)} articles."
    )

    all_results = []
    all_sentence_results = []

    # Analyze each article and its individual sentences.
    for article_number, filename in enumerate(
        article_files,
        start=1
    ):

        article_id = filename.replace(
            ".txt",
            ""
        )

        filepath = os.path.join(
            ARTICLE_DIR,
            filename
        )

        print(
            f"\n[{article_number}/{len(article_files)}] "
            f"Analyzing {filename}"
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        sentences = split_sentences(text)

        print(
            f"Sentences found: "
            f"{len(sentences)}"
        )

        sentence_results = []

        for sentence_number, sentence in enumerate(
            sentences,
            start=1
        ):

            result = analyze_sentence(sentence)

            sentence_result = {
                "article_id": article_id,
                "sentence_number": sentence_number,
                "sentence": sentence,
                "sentiment": result["sentiment"],
                "confidence": round(
                    result["confidence"],
                    6
                ),
                "positive_score": round(
                    result["positive_score"],
                    6
                ),
                "negative_score": round(
                    result["negative_score"],
                    6
                ),
                "neutral_score": round(
                    result["neutral_score"],
                    6
                )
            }

            sentence_results.append(
                sentence_result
            )

            all_sentence_results.append(
                sentence_result
            )

        # Calculate the proportion of sentences in each sentiment category.
        if sentence_results:

            positive_count = sum(
                1
                for r in sentence_results
                if r["sentiment"] == "positive"
            )

            negative_count = sum(
                1
                for r in sentence_results
                if r["sentiment"] == "negative"
            )

            neutral_count = sum(
                1
                for r in sentence_results
                if r["sentiment"] == "neutral"
            )

            total = len(sentence_results)

            positive_pct = positive_count / total
            negative_pct = negative_count / total
            neutral_pct = neutral_count / total

            overall_scores = {
                "positive": positive_pct,
                "negative": negative_pct,
                "neutral": neutral_pct
            }

            overall_sentiment = max(
                overall_scores,
                key=overall_scores.get
            )

        else:
            positive_pct = 0
            negative_pct = 0
            neutral_pct = 0
            overall_sentiment = "unknown"

        all_results.append({
            "article_id": article_id,
            "sentences": len(sentences),
            "positive_pct": round(
                positive_pct,
                4
            ),
            "negative_pct": round(
                negative_pct,
                4
            ),
            "neutral_pct": round(
                neutral_pct,
                4
            ),
            "overall_sentiment": overall_sentiment
        })

        print(
            f"Overall: "
            f"{overall_sentiment}"
        )

    # Save the article-level sentiment summaries.
    df = pd.DataFrame(
        all_results
    )

    df.to_csv(
        ARTICLE_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # Save the sentiment results for every individual sentence.
    sentence_df = pd.DataFrame(
        all_sentence_results
    )

    sentence_df.to_csv(
        SENTENCE_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nDONE")

    print(
        f"Article results saved to: "
        f"{ARTICLE_OUTPUT_FILE}"
    )

    print(
        f"Sentence results saved to: "
        f"{SENTENCE_OUTPUT_FILE}"
    )

    print(
        f"Total sentences analyzed: "
        f"{len(all_sentence_results)}"
    )


if __name__ == "__main__":
    main()
