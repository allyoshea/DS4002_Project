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

#### Step 1: Clone the Repository

1. Open a terminal on your computer.
2. Navigate to the location where you want to store the project. For example, to save the project on the Desktop:

```bash
cd ~/Desktop
```

3. Clone the GitHub repository:

```bash
git clone https://github.com/allyoshea/DS4002_Project.git
```

4. Navigate into the project folder:

```bash
cd DS4002_Project
```

5. Open the project in Visual Studio Code:

```bash
code .
```

If the `code` command is not available, open Visual Studio Code manually and select **File → Open Folder**, then select the `DS4002_Project` folder.

#### Step 2: Set Up the Python Environment

### Section 3: Instructions for Reproducing the Results

The following steps explain how to clone the repository, set up the Python environment, run the analysis scripts, and reproduce the results.

#### Step 1: Clone the Repository

1. Open **Visual Studio Code**.

2. Select the **Source Control** icon from the left sidebar.

3. Select **Clone Repository** and choose **Clone from GitHub**.

4. Select the `allyoshea/DS4002_Project` repository. You can also clone the repository using the HTTPS URL:

```text
https://github.com/allyoshea/DS4002_Project.git
```

5. When prompted to select a location for the repository, choose the **Desktop** or another preferred location.

6. Once the repository has finished cloning, select **Open** when prompted to open the repository in Visual Studio Code.

7. Confirm that the `DS4002_Project` folder appears in the Explorer on the left side of Visual Studio Code.

8. Open the integrated terminal by selecting **Terminal → New Terminal**. The terminal should automatically open in the `DS4002_Project` folder.

The repository can also be cloned directly from a terminal using:

```bash
git clone https://github.com/allyoshea/DS4002_Project.git
```

#### Step 2: Set Up the Python Environment

1. In the VS Code terminal, create a Python virtual environment:

```bash
python3 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

#### Step 3: Collect the Article Data

The repository already contains the collected article data, so rerunning the scraper is optional when reproducing the existing results.

The article URLs are stored in `data/urls.txt`.

To collect the articles again, run:

```bash
python3 scripts/scraper.py
```

This script reads the URLs from `data/urls.txt` and saves the article metadata, article text files, and any scraping errors.

#### Step 4: Run SciBERT Sentiment Analysis

Run the SciBERT sentiment analysis script:

```bash
python3 scripts/sentiment.py
```

This produces:

- `data/sentiment_results.csv`
- `data/sentence_sentiment_results.csv`

#### Step 5: Run VADER Sentiment Analysis

Run the VADER sentiment analysis script:

```bash
python3 scripts/sentiment_vader.py
```

This produces:

- `data/vader_sentiment_data.csv`

#### Step 6: Analyze Article Length

Run the article length analysis:

```bash
python3 scripts/article_length_analysis.py
```

This produces:

- `data/article_length_results.csv`
- `output/article_length_comparison.png`

#### Step 7: Generate SciBERT Figures

Run the SciBERT exploratory data analysis script:

```bash
python3 scripts/eda_sciBert.py
```

This generates the following figures in the `output/` folder:

- `sentiment_composition_stacked.png`
- `positive_pct_by_type.png`
- `negative_pct_by_type.png`
- `neutral_pct_by_type.png`

#### Step 8: Generate VADER Figures

Run the VADER exploratory data analysis script:

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
