# DS 4002 Project 1
The Synapses

This repository contains the process for developing and running a sentiment analysis model trained on scientific and news articles regarding brain-computer interfaces (BCIs).

## Contents of the Repository
### Software and Platform

-**IDE**: Visual Studio Code (VS Code)
-**Terminal**: Windows PowerShell
-**Programming Language**: Python

### Add-On Packages and Libraries

The project uses the following Python packages:

- pandas
- requests
- trafilatura
- torch
- transformers
- nltk
- vaderSentiment
- scikit-learn
- matplotlib

See `requirements.txt` for the complete list of required packages.

### Map of the Documentation

The repository is organized as follows:
```text
DS4002_Project/
│
├── LICENSE
├── README.md
├── data
│   ├── urls.txt
│   ├── articles
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   ├── 003.txt
│   │   ├── . . .
│   │   └── 060.txt
│   ├── articles.csv
│   ├── article_length_results.csv
│   ├── sentence_sentiment_results.csv
│   ├── sentiment_results.csv
│   └── vader_sentiment_data.csv
├── output
│   ├── article_length_comparison.png
│   ├── negative_pct_by_type.png
│   ├── neutral_pct_by_type.png
│   ├── positive_pct_by_type.png
│   ├── sentiment_composition_stacked.png
│   ├── vader_compound_by_article.png
│   ├── vader_mean_compound_by_type.png
│   └── vader_sentiment_composition_stacked.png
├── requirements.txt
├── scrape_errors.csv
└── scripts
    ├── README.md
    ├── article_length_analysis.py
    ├── eda_sciBert.py
    ├── eda_vader.py
    ├── scraper.py
    ├── sentiment.py
    └── sentiment_vader.py
```
**Folder and File Descriptions**

- `data/` — Contains the article data, sentiment results, and other processed datasets.
- `data/articles/` — Contains the 60 individual article text files collected by the web scraper.
- `data/articles.csv` — Contains metadata for the articles, including source, date, title, author, URL, and text file.
- `data/urls.txt` — Contains the URLs used to collect the articles.
- `data/article_length_results.csv` — Contains the results of the article length analysis.
- `data/sentence_sentiment_results.csv` — Contains sentence-level sentiment results from SciBERT.
- `data/sentiment_results.csv` — Contains article-level sentiment results from SciBERT.
- `data/vader_sentiment_data.csv` — Contains article-level sentiment results from VADER.
- `output/` — Contains the figures generated during exploratory data analysis.
- `scripts/` — Contains the Python scripts used to collect, analyze, and visualize the data.
- `scripts/scraper.py` — Scrapes article text and metadata from the URLs in `urls.txt`.
- `scripts/sentiment.py` — Performs SciBERT sentiment analysis.
- `scripts/sentiment_vader.py` — Performs VADER sentiment analysis.
- `scripts/article_length_analysis.py` — Analyzes article length.
- `scripts/eda_sciBert.py` — Generates EDA figures for the SciBERT results.
- `scripts/eda_vader.py` — Generates EDA figures for the VADER results.
- `requirements.txt` — Lists the Python packages required to reproduce the analysis.
- `scrape_errors.csv` — Records articles that could not be successfully scraped.
- `README.md` — Provides an overview of the project and instructions for reproducing the results.
- `LICENSE` — Contains the project's license.

### Instructions for Reproducing the Results
Step 1: Clone the Repository and Install Requirements

Clone the GitHub repository and navigate into the project folder:

git clone https://github.com/allyoshea/DS4002_Project.git
cd DS4002_Project


Install the required Python packages:

pip install -r requirements.txt

Step 2: Scrape the Articles

This step is optional if the pre-scraped articles are already included in the repository.

The file data/urls.txt contains the URLs of all the scientific and news articles used in the project.

To run the web scraper, use:

python scripts/scraper.py


Wait for the script to finish running. The scraper should produce:

articles.csv

An articles/ folder containing the scraped articles

These files are required for the sentiment analysis step.

Step 3: Perform Sentiment Analysis

The sentiment analysis uses a SciBERT model to analyze the scraped articles.

Run the sentiment analysis script from the main project directory:

python scripts/sentiment.py


Wait for the script to finish running. The analysis should produce:

sentiment_results.csv — overall sentiment results

sentence_sentiment_results.csv — sentiment results for individual sentences

Step 4: Create EDA Plots and Analyze the Results

[Add instructions for running the EDA scripts and/or creating the plots here.]

For example:

python scripts/[EDA_SCRIPT_NAME].py


The resulting plots and analysis should be saved in:

[LOCATION OF RESULTS]


Review the resulting figures and datasets to analyze the sentiment trends found in the BCI-related articles.

Project Workflow

The overall workflow for reproducing the project results is:

Clone Repository
       ↓
Install Requirements
       ↓
Scrape Articles
       ↓
articles.csv + articles/
       ↓
Run SciBERT Sentiment Analysis
       ↓
sentiment_results.csv
sentence_sentiment_results.csv
       ↓
Create EDA Plots
       ↓
Analyze Results
