**DS 4002 Project 1
The Synapses**

This repository contains the process for a sentiment analysis model trained on scientific and news articles regarding brain-computer interfaces (BCIs).

**Software**
Software used: Python IDE (VSCode), Windows Powershell Terminal

Programming language: Python

Add-on packages/libraries: requests, tralifura, torch, transformers
See requirements.txt

Section 2: Map of the Documentation
Project Folder/
│
├── [Folder/File]
│   ├── [File]
│   └── [File]
│
├── [Folder]
│   └── [File]
│
├── [File]
└── [File]

Folder and File Descriptions

[File/Folder] — [Description]

[File/Folder] — [Description]

[File/Folder] — [Description]

[File/Folder] — [Description]

**Instructions for Reproducing the Results**

Step 1: Clone GitHub repo folder and run requirements.txt to install packages

git clone ...
pip install -r requirements.txt

(Optional Step 2): Run web scraper to scrape articles or use pre-scraped articles

"urls.txt" contains all of the web links of the articles we scraped.
In Terminal, run code.py and wait for script to complete. This should result in "articles.csv" and the "articles" folder, which are required for the sentiment analysis model.

Step 3: Perform sentiment analysis using SciBERT model

In Terminal, run sentiment.py and wait for script to complete. This should result in "sentiment_results.csv" and "sentence_sentiment_results.csv", respectively.

Step 4: Create EDA plots and run analysis on your findings

[Instructions]
