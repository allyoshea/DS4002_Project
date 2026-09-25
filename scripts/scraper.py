import csv
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import requests
import trafilatura



# SETTINGS


URL_FILE = "urls.txt"

ARTICLE_DIR = "articles"
OUTPUT_FILE = "articles.csv"
ERROR_FILE = "scrape_errors.csv"

# Wait between different article requests
DELAY_SECONDS = 10

# If a server returns 202, try again this many times
MAX_202_RETRIES = 3

# Wait between 202 retries
RETRY_DELAY_SECONDS = 10



# CSV COLUMNS


CSV_FIELDS = [
    "article_id",
    "source",
    "date",
    "title",
    "author",
    "url",
    "description",
    "text_file",
    "retrieved_at"
]


# HTTP SESSION


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


# CREATE ARTICLE DIRECTORY


os.makedirs(ARTICLE_DIR, exist_ok=True)



# CLEAN TEXT


def clean_text(text):

    if not text:
        return ""

    # Turn newlines, tabs, and repeated spaces into one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ERROR LOGGING


def log_error(
    url,
    status_code,
    error_type,
    message
):

    file_exists = os.path.exists(ERROR_FILE)

    with open(
        ERROR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "url",
                "domain",
                "status_code",
                "error_type",
                "message"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            url,
            urlparse(url).netloc,
            status_code,
            error_type,
            message
        ])


# GET NEXT ARTICLE ID


def get_next_article_id():

    existing_files = [
        f for f in os.listdir(ARTICLE_DIR)
        if f.endswith(".txt")
    ]

    numbers = []

    for filename in existing_files:

        match = re.match(
            r"(\d+)\.txt$",
            filename
        )

        if match:
            numbers.append(
                int(match.group(1))
            )

    if not numbers:
        return 1

    return max(numbers) + 1



# SAVE ARTICLE METADATA


def save_metadata(
    article,
    article_id,
    filepath
):

    file_exists = os.path.exists(
        OUTPUT_FILE
    )

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({

            "article_id":
                f"{article_id:03d}",

            "source":
                article["source"],

            "date":
                article["date"],

            "title":
                article["title"],

            "author":
                article["author"],

            "url":
                article["url"],

            "description":
                article["description"],

            "text_file":
                filepath,

            "retrieved_at":
                article["retrieved_at"]
        })


# DOWNLOAD + EXTRACT ONE ARTICLE


def scrape_article(url):

    print("\n" + "=" * 70)
    print(f"URL: {url}")

    response = None

   
    # DOWNLOAD
   

    for attempt in range(
        1,
        MAX_202_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                timeout=30,
                allow_redirects=True
            )

            print(
                f"HTTP status: "
                f"{response.status_code}"
            )

            if response.status_code == 202:

                if attempt < MAX_202_RETRIES:

                    print(
                        f"202 received. "
                        f"Waiting "
                        f"{RETRY_DELAY_SECONDS} seconds "
                        f"before retry..."
                    )

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

                    continue

                else:

                    log_error(
                        url,
                        202,
                        "202_RETRIES_EXHAUSTED",
                        (
                            f"Received 202 after "
                            f"{MAX_202_RETRIES} attempts"
                        )
                    )

                    print(
                        "FAILED: Still receiving 202"
                    )

                    return None

            if response.status_code != 200:

                # Special message for rate limiting
                if response.status_code == 429:

                    retry_after = (
                        response.headers.get(
                            "Retry-After",
                            ""
                        )
                    )

                    message = (
                        "Rate limited"
                    )

                    if retry_after:
                        message += (
                            f". Retry-After: "
                            f"{retry_after}"
                        )

                    log_error(
                        url,
                        429,
                        "RATE_LIMITED",
                        message
                    )

                    print(
                        "FAILED: Rate limited "
                        "(HTTP 429)"
                    )

                else:

                    log_error(
                        url,
                        response.status_code,
                        "HTTP_ERROR",
                        response.reason
                    )

                    print(
                        f"FAILED: "
                        f"HTTP {response.status_code}"
                    )

                return None

            break

        except requests.exceptions.Timeout:

            log_error(
                url,
                "",
                "TIMEOUT",
                "Request timed out after 30 seconds"
            )

            print(
                "FAILED: Request timed out"
            )

            return None

        except requests.exceptions.ConnectionError as e:

            log_error(
                url,
                "",
                "CONNECTION_ERROR",
                str(e)
            )

            print(
                "FAILED: Connection error"
            )

            return None

        except requests.exceptions.RequestException as e:

            log_error(
                url,
                "",
                "REQUEST_ERROR",
                str(e)
            )

            print(
                f"FAILED: {e}"
            )

            return None

        except Exception as e:

            log_error(
                url,
                "",
                "UNKNOWN_ERROR",
                str(e)
            )

            print(
                f"FAILED: {e}"
            )

            return None

    try:

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_precision=True
        )

    except Exception as e:

        log_error(
            url,
            response.status_code,
            "EXTRACTION_ERROR",
            str(e)
        )

        print(
            f"FAILED: Extraction error: {e}"
        )

        return None

    if not text:

        log_error(
            url,
            response.status_code,
            "EXTRACTION_FAILED",
            "Trafilatura could not extract article text"
        )

        print(
            "FAILED: Could not extract article text"
        )

        return None

    text = clean_text(text)


    try:

        metadata = trafilatura.extract_metadata(
            response.text
        )

    except Exception:

        metadata = None

    if metadata:

        title = clean_text(
            metadata.title or ""
        )

        author = clean_text(
            metadata.author or ""
        )

        date = clean_text(
            metadata.date or ""
        )

        description = clean_text(
            metadata.description or ""
        )

    else:

        title = ""
        author = ""
        date = ""
        description = ""

    print("SUCCESS")

    print(
        f"Title: {title}"
    )

    print(
        f"Characters extracted: "
        f"{len(text):,}"
    )

    return {
        "url": url,
        "source": urlparse(url).netloc,
        "title": title,
        "author": author,
        "date": date,
        "description": description,
        "text": text,
        "retrieved_at":
            datetime.now().isoformat(
                timespec="seconds"
            )
    }


def save_article(
    article,
    article_id
):

    filename = (
        f"{article_id:03d}.txt"
    )

    filepath = os.path.join(
        ARTICLE_DIR,
        filename
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(article["text"])

    return filepath

def main():

    if not os.path.exists(URL_FILE):

        print(
            f"ERROR: Could not find "
            f"{URL_FILE}"
        )

        return


    # READ URLS
   

    with open(
        URL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        urls = [
            line.strip()
            for line in f
            if line.strip()
            and not line.strip().startswith("#")
        ]

    print("=" * 70)
    print("ARTICLE SCRAPER")
    print("=" * 70)

    print(
        f"Found {len(urls)} URLs."
    )

    print(
        f"Delay between articles: "
        f"{DELAY_SECONDS} seconds"
    )

    next_id = get_next_article_id()

    successful = 0
    failed = 0

    for i, url in enumerate(
        urls,
        start=1
    ):

        print(
            f"\nArticle {i} "
            f"of {len(urls)}"
        )

        article = scrape_article(url)

        if article:

            article_id = next_id

            filepath = save_article(
                article,
                article_id
            )

            save_metadata(
                article,
                article_id,
                filepath
            )

            print(
                f"Saved: {filepath}"
            )

            successful += 1
            next_id += 1

        else:

            failed += 1

        if i < len(urls):

            print(
                f"\nWaiting "
                f"{DELAY_SECONDS} seconds..."
            )

            for remaining in range(
                DELAY_SECONDS,
                0,
                -1
            ):

                print(
                    f"\rNext article in "
                    f"{remaining} seconds...",
                    end="",
                    flush=True
                )

                time.sleep(1)

            print()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"URLs attempted: {len(urls)}"
    )

    print(
        f"Successfully scraped: "
        f"{successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"\nMetadata: "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

    print(
        f"Article text: "
        f"{os.path.abspath(ARTICLE_DIR)}"
    )

    print(
        f"Errors: "
        f"{os.path.abspath(ERROR_FILE)}"
    )

if __name__ == "__main__":
    main()
