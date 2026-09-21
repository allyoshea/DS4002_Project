import os
import re
import pandas as pd

from transformers import pipeline


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "puzzz21/sci-sentiment-classify"

ARTICLE_DIR = "articles"

ARTICLE_OUTPUT_FILE = "sentiment_results.csv"

SENTENCE_OUTPUT_FILE = "sentence_sentiment_results.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading scientific sentiment model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None
)

print("Model loaded successfully!")


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):

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


# ============================================================
# CONVERT MODEL LABELS
# ============================================================

def convert_label(label):

    label = label.lower()

    if label == "p":
        return "positive"

    elif label == "n":
        return "negative"

    elif label == "o":
        return "neutral"

    else:
        return label


# ============================================================
# ANALYZE ONE SENTENCE
# ============================================================

def analyze_sentence(sentence):

    results = classifier(
        sentence,
        truncation=True,
        max_length=512
    )[0]

    # Convert labels and store all probabilities
    scores = {}

    for item in results:

        sentiment = convert_label(
            item["label"]
        )

        scores[sentiment] = item["score"]


    # Find highest probability
    best_sentiment = max(
        scores,
        key=scores.get
    )


    return {
        "sentiment": best_sentiment,
        "confidence": scores[best_sentiment],

        "positive_score":
            scores.get("positive", 0),

        "negative_score":
            scores.get("negative", 0),

        "neutral_score":
            scores.get("neutral", 0)
    }


# ============================================================
# ANALYZE ALL ARTICLES
# ============================================================

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

        print(
            "ERROR: No .txt articles found."
        )

        return


    print(
        f"\nFound {len(article_files)} articles."
    )


    # Article-level results
    all_results = []

    # Sentence-level results
    all_sentence_results = []


    # ========================================================
    # ARTICLE LOOP
    # ========================================================

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


        # ----------------------------------------------------
        # READ ARTICLE
        # ----------------------------------------------------

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()


        # ----------------------------------------------------
        # SPLIT INTO SENTENCES
        # ----------------------------------------------------

        sentences = split_sentences(text)


        print(
            f"Sentences found: "
            f"{len(sentences)}"
        )


        # ----------------------------------------------------
        # SENTENCE ANALYSIS
        # ----------------------------------------------------

        sentence_results = []


        for sentence_number, sentence in enumerate(
            sentences,
            start=1
        ):

            result = analyze_sentence(
                sentence
            )


            sentence_result = {

                "article_id":
                    article_id,

                "sentence_number":
                    sentence_number,

                "sentence":
                    sentence,

                "sentiment":
                    result["sentiment"],

                "confidence":
                    round(
                        result["confidence"],
                        6
                    ),

                "positive_score":
                    round(
                        result["positive_score"],
                        6
                    ),

                "negative_score":
                    round(
                        result["negative_score"],
                        6
                    ),

                "neutral_score":
                    round(
                        result["neutral_score"],
                        6
                    )
            }


            # Keep for this article
            sentence_results.append(
                sentence_result
            )


            # Keep for entire dataset
            all_sentence_results.append(
                sentence_result
            )


        # ----------------------------------------------------
        # ARTICLE-LEVEL SUMMARY
        # ----------------------------------------------------

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


            total = len(
                sentence_results
            )


            positive_pct = (
                positive_count / total
            )

            negative_pct = (
                negative_count / total
            )

            neutral_pct = (
                neutral_count / total
            )


            overall_scores = {

                "positive":
                    positive_pct,

                "negative":
                    negative_pct,

                "neutral":
                    neutral_pct

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


        # ----------------------------------------------------
        # ADD ARTICLE SUMMARY
        # ----------------------------------------------------

        all_results.append({

            "article_id":
                article_id,

            "sentences":
                len(sentences),

            "positive_pct":
                round(
                    positive_pct,
                    4
                ),

            "negative_pct":
                round(
                    negative_pct,
                    4
                ),

            "neutral_pct":
                round(
                    neutral_pct,
                    4
                ),

            "overall_sentiment":
                overall_sentiment

        })


        print(
            f"Overall: "
            f"{overall_sentiment}"
        )


    # ========================================================
    # SAVE ARTICLE RESULTS
    # ========================================================

    df = pd.DataFrame(
        all_results
    )


    df.to_csv(
        ARTICLE_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # SAVE SENTENCE RESULTS
    # ========================================================

    sentence_df = pd.DataFrame(
        all_sentence_results
    )


    sentence_df.to_csv(
        SENTENCE_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "DONE"
    )

    print(
        "========================================"
    )

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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()