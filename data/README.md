# Metadata 
## Data Summary 
Our dataset consists of 60 BCI-focused articles published between 2024 and 2026,
including 30 academic articles and 30 news/media articles. The academic sample contains an even amount of original research papers and review papers, selected as an even mix from three well-established
publication groups. Articles were selected by the students based on their relevance to BCIs and
publication date. Article text was collected through web scraping in Python and stored as
individual .txt files (001.txt-060.txt – articles folder in data folder), while metadata and sentiment results were
stored in CSV files (articles.csv, sentiment_results.csv). The text was analyzed using the VADER
sentiment analysis package in Python and SciBERT model used Python. The dataset is available in the group's GitHub repository,
here which contains the the files above as well as any generated plots during EDA
## Provenance 
The dataset was collected between September 14th 2026 and September 24th 2026,
by three University of Virginia students (Carter Grohe, Coco Zhao, and Ally O'Shea) for their
Data science project (aka prototyping) (DS 4002) class. Articles were collected from publicly
accessible webpages using Python and stored as 60 individual text files labeled 001.txt -060.txt
with a corresponding metadata CSV file. Sentiment analysis was conducted using VADER, SciBERT and
resulting data and scripts are present in the github as well. The dataset can be accessed through
the group's GitHub repository or from the local project files on individual laptops maintained by
the students.
## Ethical Statements 
The dataset consists of publicly available articles and does not have private individual
information. However, original article content remains subject to the rights and policies of the
original publishers. 
## Data dictionary 
| Variable | Description | Data Type | Uncertainty |
|---|---|---|---|
| `article_id` | Identifier assigned to the article during scraping | Integer | IDs reused across scraping batches – not unique identifiers |
| `article_type` | Category identifying whether the article is from a scientific publication or news/media | Categorical | Assigned by project team |
| `source` | Publication or website where the article was found | Categorical/Text | May vary across webpages (i.e. some include www. before) |
| `date` | Publication date of the article | Date | N/A |
| `title` | Title of the article | Text | N/A |
| `author` | Listed author(s) of the article | Text | N/A |
| `url` | Original webpage URL for the article (provided as input) | Text | N/A |
| `description` | Short description or summary associated with the article | Text | May vary based on what publisher includes |
| `text_file` | Path to the locally stored text file | Text | File may not be captured perfectly because of web scraping |
| `retrieved_at` | Date & time when the article was collected | Date/Time | N/A |
| `negative` | Average VADER negative sentiment score across sentences in the article | Numeric | VADER may not interpret scientific/technical language perfectly |
| `neutral` | Average VADER neutral sentiment score across sentences in the article | Numeric | VADER may not interpret scientific/technical language perfectly |
| `positive` | Average VADER positive sentiment score across sentences in the article | Numeric | VADER may not interpret scientific/technical language perfectly |
| `compound` | Average VADER compound sentiment score across sentences in the article; ranges from -1 to +1 | Numeric | VADER is a general-purpose sentiment tool and may not fully capture sentiment in scientific/technical writing |
| `positive_pct` | Proportion of sentences classified as positive by SciBERT | Numeric | SciBERT classifications may not perfectly capture sentiment in scientific/technical language |
| `negative_pct` | Proportion of sentences classified as negative by SciBERT | Numeric | SciBERT classifications may not perfectly capture sentiment in scientific/technical language |
| `neutral_pct` | Proportion of sentences classified as neutral by SciBERT | Numeric | SciBERT classifications may not perfectly capture sentiment in scientific/technical language |
| `overall_sentiment` | Sentiment category with the largest proportion of sentences classified by SciBERT | Categorical | Depends on the model's sentence-level classifications |
| `sentences` | Number of sentences analyzed in the article | Integer | N/A |
## Exploratory plots
<img width="2370" height="1770" alt="image" src="https://github.com/user-attachments/assets/5af592e6-b081-466d-8954-4699982c9189" />
**Figure 1.** Word count comparison between scientific versus news/media articles
<img width="1318" height="1020" alt="image" src="https://github.com/user-attachments/assets/53c6a193-7a4c-4f75-9f6d-16abf11dc254" />
**Figure 2.** Exploratory plot from when there were 30 articles of spread of VADER sentiment score over year published.
