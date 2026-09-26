# DS 4002 Project 1
The Synapses

This repository contains the process for developing and running a sentiment analysis model trained on scientific and news articles regarding brain-computer interfaces (BCIs).

## Contents of the Repository
### Software and Platform

- **IDE**: Visual Studio Code (VS Code)
- **Terminal**: Windows PowerShell
- **Programming Language**: Python

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
### Section 3: Instructions for Reproducing the Results

The following steps describe how to download the project, set up the Python environment, collect the article data, run the sentiment analyses, and generate the final figures.

1. Go to the GitHub repository and copy the HTTPS URL by clicking the <Code> button:

```text
https://github.com/allyoshea/DS4002_Project.git
```

2. Open the **Terminal application on your laptop**. This is the regular computer terminal, not the terminal inside your coding platform.

3. Navigate to the location where you want to save the project, such as your Desktop:

```bash
cd ~/Desktop
```

4. Clone the repository:

```bash
git clone https://github.com/allyoshea/DS4002_Project.git
```

5. Navigate into the project folder:

```bash
cd DS4002_Project
```

6. Open the `DS4002_Project` folder in your preferred coding platform, such as Visual Studio Code.

#### Step 2: Set Up the Python Environment

1. Once the project is open in your coding platform, open the **integrated terminal inside the coding platform**. In Visual Studio Code, select **Terminal → New Terminal on top bar of computer normally**.

2. Confirm that the terminal is located in the `DS4002_Project` folder. If needed, navigate to the folder:

```bash
cd ~/Desktop/DS4002_Project
```
3. Create a Python virtual environment:

```bash
python3 -m venv .venv
```
4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Install the required packages:

```bash
pip install -r requirements.txt
```

#### Step 3: Collect the Article Data

The repository already contains the collected article data, so rerunning the scraper is optional when reproducing the existing results.

The article URLs are stored in `data/urls.txt`.

To collect the articles again, run the following command in the **coding platform's integrated terminal**:

```bash
python3 scripts/scraper.py
```

This script reads the URLs from `data/urls.txt` and saves the article metadata, article text files, and any scraping errors. This will take awhile and to avoid any issues with potential firewalls reference the data/articles folder with the already scraped .txt files. 

#### Step 4: Run SciBERT Sentiment Analysis

In the **coding platform's integrated terminal**, run:

```bash
python3 scripts/sentiment.py
```

This produces:

- `data/sentiment_results.csv`
- `data/sentence_sentiment_results.csv`

#### Step 5: Run VADER Sentiment Analysis

Run:

```bash
python3 scripts/sentiment_vader.py
```

This produces:

- `data/vader_sentiment_data.csv`

#### Step 6: Analyze Article Length

Run:

```bash
python3 scripts/article_length_analysis.py
```

This produces:

- `data/article_length_results.csv`
- `output/article_length_comparison.png`

#### Step 7: Generate SciBERT Figures

Run:

```bash
python3 scripts/eda_sciBert.py
```

This generates the following figures in the `output/` folder:

- `sentiment_composition_stacked.png`
- `positive_pct_by_type.png`
- `negative_pct_by_type.png`
- `neutral_pct_by_type.png`

#### Step 8: Generate VADER Figures

Run:

```bash
python3 scripts/eda_vader.py
```

This generates the following figures in the `output/` folder:

- `vader_sentiment_composition_stacked.png`
- `vader_compound_by_article.png`
- `vader_mean_compound_by_type.png`

#### Step 9: Review the Results

After running the scripts, the processed datasets will be located in the `data/` folder and the generated figures will be located in the `output/` folder.

The project contains 60 BCI-related articles:

- 30 scientific articles
- 30 news/media articles

The resulting datasets and figures can be used to compare sentiment patterns and article length between the two article types.

### Project Workflow

```text
Clone GitHub Repository
        ↓
Open Project in VS Code
        ↓
Create and Activate Virtual Environment
        ↓
Install Required Packages
        ↓
Scrape Articles
        ↓
articles.csv + article text files
        ↓
Run SciBERT Sentiment Analysis
        ↓
SciBERT Results
        ↓
Run VADER Sentiment Analysis
        ↓
VADER Results
        ↓
Analyze Article Length
        ↓
Run EDA Scripts
        ↓
Figures saved in output/
        ↓
Review and Compare Results
```
