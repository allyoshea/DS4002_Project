DS 4002 Project 1
The Synapses

This repository contains the process for developing and running a sentiment analysis model trained on scientific and news articles regarding brain-computer interfaces (BCIs).

Software and Platform
Software

IDE: Visual Studio Code (VS Code)

Terminal: Windows PowerShell

Programming Language: Python

Add-On Packages and Libraries

The project uses the following Python packages:

pandas

requests

trafilatura

torch

transformers

See requirements.txt for the complete list of required packages.

Map of the Documentation

The repository is organized as follows:

DS4002_Project/
│
├── data/
│   ├── articles/
│   │   └── [scraped article files]
│   ├── articles.csv
│   ├── sentiment_results.csv
│   ├── sentence_sentiment_results.csv
│   └── urls.txt
│
├── scripts/
│   ├── scraper.py
│   └── sentiment.py
│
├── requirements.txt
├── README.md
└── LICENSE.md

Folder and File Descriptions

data/ — Contains the datasets and files used throughout the project.

data/articles/ — Contains the individual articles collected by the web scraper.

data/articles.csv — Contains the scraped article data in CSV format.

data/urls.txt — Contains the URLs of the scientific and news articles used in the project.

data/sentiment_results.csv — Contains the overall sentiment results produced by the sentiment analysis model.

data/sentence_sentiment_results.csv — Contains sentiment results for individual sentences.

scripts/ — Contains the Python scripts used to scrape articles and perform sentiment analysis.

scripts/scraper.py — Scrapes the articles listed in urls.txt.

scripts/sentiment.py — Uses the SciBERT model to perform sentiment analysis.

requirements.txt — Lists the Python packages required to run the project.

README.md — Provides an overview of the project and instructions for reproducing the results.

LICENSE.md — Contains the project's MIT License.

Instructions for Reproducing the Results
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
