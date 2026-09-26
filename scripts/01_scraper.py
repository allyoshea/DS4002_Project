"""
Script: 01_scraper.py

Purpose:
    Collect article text and metadata from the URLs listed in
    data/urls.txt for the BCI sentiment analysis Project 1. 

Input:
    data/urls.txt

Outputs:
    data/articles.csv
    data/articles/*.txt
    scrape_errors.csv

Process:
    1. Read article URLs from data/urls.txt.
    2. Download each webpage using requests.
    3. Extract article text and metadata using Trafilatura.
    4. Clean the extracted text.
    5. Save each article as a numbered text file.
    6. Save article metadata to articles.csv.
    7. Record unsuccessful requests or extraction errors in
       scrape_errors.csv.

Notes:
    The scraper waits between requests to reduce the frequency of
    requests sent to individual websites. HTTP 202 responses are
    retried, and other HTTP or connection errors are recorded in
    the error log.
"""

import csv
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import requests
import trafilatura


# SETTINGS
# These paths and timing settings control where input and output
# files are stored and how frequently article requests are made.

URL_FILE = "data/urls.txt"

ARTICLE_DIR = "articles"
OUTPUT_FILE = "articles.csv"
ERROR_FILE = "scrape_errors.csv"

# Wait between different article requests.
# This reduces the frequency of requests sent to websites.
DELAY_SECONDS = 10

# If a server returns HTTP 202, retry the request up to this
# number of times before recording it as a failed request.
MAX_202_RETRIES = 3

# Wait between retries after receiving an HTTP 202 response.
RETRY_DELAY_SECONDS = 10


# CSV COLUMNS
# These are the fields that will be stored for each successfully
# scraped article in the metadata CSV file.

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
# A persistent requests session is used so that the same request
# settings and browser-like headers are applied to each webpage.

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
# Create the directory for individual article text files if it
# does not already exist.

os.makedirs(ARTICLE_DIR, exist_ok=True)


# CLEAN TEXT

def clean_text(text):
    """
    Clean extracted article text by replacing repeated whitespace
    with single spaces and removing whitespace at the beginning
    and end of the text.
    """

    if not text:
        return ""

    # Replace newlines, tabs, and repeated spaces with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ERROR LOGGING

def log_error(
    url,
    status_code,
    error_type,
    message
):
    """
    Record a scraping or extraction error in scrape_errors.csv.

    The error log stores the time of the error, URL, website domain,
    HTTP status code when available, error type, and a description
    of the problem.
    """

    file_exists = os.path.exists(ERROR_FILE)

    with open(
        ERROR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        # Add column headers when creating the error file for the
        # first time.
        if not file_exists:

            writer.writerow([
                "timestamp",
                "url",
                "domain",
                "status_code",
                "error_type",
                "message"
            ])

        # Add information about the current error.
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
    """
    Determine the next available article ID by examining the
    existing text files in the article directory.

    This allows the scraper to continue numbering articles without
    overwriting previously collected article files.
    """

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
    """
    Append metadata for a successfully scraped article to
    articles.csv.
    """

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

        # Write the CSV header only when the output file is new.
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
    """
    Download one webpage and extract its article text and metadata.

    The function handles HTTP errors, connection errors, timeouts,
    and Trafilatura extraction failures. Failed URLs are recorded
    in scrape_errors.csv and return None.
    """

    print("\n" + "=" * 70)
    print(f"URL: {url}")

    response = None

    # DOWNLOAD
    # Request the webpage and retry temporary HTTP 202 responses.
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

            # HTTP 202 indicates that the server accepted the
            # request but has not completed processing it. Retry
            # before treating the request as unsuccessful.
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

            # Any status other than 200 indicates that the webpage
            # could not be downloaded successfully.
            if response.status_code != 200:

                # HTTP 429 indicates that the server has rate-limited
                # the scraper.
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

            # Record requests that take longer than the 30-second
            # timeout period.
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

            # Record errors caused by problems connecting to the
            # website.
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

            # Record other errors raised by the requests library.
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

            # Record unexpected errors so that one failed article
            # does not stop the entire scraping process.
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

    # Extract the main article text from the downloaded webpage.
    # Trafilatura is configured to prioritize precise article text
    # while excluding comments and links.
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

    # If no article text was extracted, record the failure rather
    # than saving an empty article file.
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

    # Standardize whitespace in the extracted article text.
    text = clean_text(text)

    # Extract additional metadata such as title, author, date  and description from the webpage.
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

        # If metadata cannot be extracted, retain the article text but leave the metadata fields blank.
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

    # Return all information needed to save the article text + metadata later in the main workflow.
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


# SAVE ARTICLE

def save_article(
    article,
    article_id
):
    """
    Save the extracted text for an article as a numbered text file.
    """

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


# MAIN SCRAPING WORKFLOW

def main():
    """
    Run the complete article scraping workflow.

    This function reads the URLs, processes each article, saves
    successful results, records failures, and prints a summary
    when scraping is complete.
    """

    if not os.path.exists(URL_FILE):

        print(
            f"ERROR: Could not find "
            f"{URL_FILE}"
        )

        return

    # READ URLS
    # Load non empty URLs from the input file. Lines beginning with
    # '#' are treated as comments and ignored.
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

    # Determine the next available article ID so that new articles receive unique filenames.
    next_id = get_next_article_id()

    successful = 0
    failed = 0

    # Process each URL one at a time.
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

            # Save the article text and metadata after a successful scrape.
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

        # Wait between requests to avoid sending requests to websites too frequently and risk getting banned.
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

    # Print a summary of the scraping process and the locations of
    # the generated output files.
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


# Run the main scraping workflow only when this file is executed
# directly, rather than when it is imported by another script.
if __name__ == "__main__":
    main()
