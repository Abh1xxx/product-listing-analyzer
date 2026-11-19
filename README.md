# Product Listing Analyzer

A beginner-friendly NLP course application for exploring e-commerce product titles and descriptions. It demonstrates real local text processing with NLTK and corpus-fitted TF-IDF with scikit-learn. It does not use external LLMs, paid APIs, or a product-classification model.

## Problem statement and objectives

Product listings combine short titles and longer descriptions. This application demonstrates how basic NLP can clean and summarize that text, surface repeated terms and phrases, compare title and description vocabulary, and identify attributes explicitly mentioned in a dataset. The goal is an explainable classroom demonstration, not a commercial ranking system.

## Dataset and columns

The supplied workbook is `data/Product_Listing_Analyzer_500_Corpus_Detailed.xlsx`. Its `Corpus` sheet contains 500 records and 18 columns:

`Product_ID`, `Category`, `Product_Type`, `Brand`, `Product_Title`, `Product_Description`, `Color`, `Material`, `Size_or_Variant`, `Feature_1`, `Feature_2`, `Target_Audience`, `Price`, `Currency`, `Rating`, `Review_Count`, `Stock_Quantity`, and `Country_of_Origin`.

The inspected workbook has five categories with 100 records each. The `Corpus` sheet has no missing values, exact duplicate rows, or duplicate product IDs. The loader still handles missing values in the text model and removes exact duplicate records. Only actual columns present are used; CSV files are supported too. If the bundled file is missing, a small labeled demonstration corpus is used instead.

## NLP techniques

- **Lowercasing and cleaning:** standardizes case, replaces punctuation, and normalizes whitespace.
- **Word and sentence tokenization:** splits text into words and sentence-like units using NLTK tokenizers.
- **Stop-word removal:** removes common English words (for example, “the” and “is”) with NLTK's stop-word list.
- **Porter stemming:** trims word endings to approximate roots; a stem is not always a real dictionary word.
- **Lemmatization:** maps words to dictionary forms using WordNet (noun form by default).
- **Frequency and repetition:** counts tokens and reports repeated words.
- **N-grams:** lists adjacent two-word and three-word sequences (bigrams and trigrams).
- **TF-IDF:** learns term importance from the available title/description corpus, then transforms new text with that fitted vectorizer. Scores depend on this dataset and are not hardcoded per product.
- **Rule-based attributes:** recognizes values in available color, material, size/variant, and feature fields. A failure to detect an attribute is not proof that it is absent.
- **Cosine similarity:** optionally compares the title and description as vectors in the corpus TF-IDF space. It is a lexical-overlap measure, not a semantic judgment.

## Processing workflow

1. Load the workbook's `Corpus` sheet (or a CSV) and inspect its columns.
2. Join available title and description text to fit one corpus-wide TF-IDF vectorizer.
3. Clean and tokenize a user-entered title and description.
4. Compute stop-word-filtered text statistics, word forms, n-grams, and keywords.
5. Search dataset-backed attribute dictionaries and compute the transparent score.
6. Present results, limitations, and suggestions in Streamlit.

## Quality scoring formula

The total is the sum of five equally weighted components, each capped at 20 points (100 maximum):

1. **Title completeness:** 20 points for at least 5 tokens; 12 for 3–4; 5 for 1–2; 0 for empty.
2. **Keyword coverage:** `20 × (number of extracted TF-IDF keywords found in the title ÷ number extracted)`, rounded to a whole point; 0 when no keywords are extracted.
3. **Description quality:** up to 8 points for 30+ words (4 for 15–29), up to 6 for 2+ sentences (3 for 1), and up to 6 for 10+ unique words (3 for 5–9).
4. **Attribute coverage:** `20 × (number of available attribute categories detected ÷ number of categories supported by dataset columns)`, rounded to a whole point. It measures explicit textual coverage only; a non-detection is not a claim that the product lacks an attribute.
5. **Repetition and readability:** start at 20. Repeated-token ratios above 5%, 10%, and 20% subtract 2, 5, and 10 points, respectively. A title over 150 characters and a sentence over 35 words each subtract 5 points.

This is a project-defined educational metric. It is not an official marketplace ranking or a validated prediction of sales.

## Installation and execution

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
streamlit run app.py
```

NLTK's word and sentence tokenizers used here do not require a separate Punkt model. Stop words and WordNet are downloaded by the setup command; the application also attempts to retrieve each resource on first use and displays an actionable message if it cannot.

Run the focused tests:

```powershell
python -m unittest discover -s tests -v
```

## Example input and expected output types

Example title: `Northstar Blue Cotton T-Shirt for Everyday Wear`

Example description: `This lightweight cotton T-shirt has a comfortable fit for everyday wear. The soft fabric and classic crew neck make it easy to pair with casual outfits.`

The app displays cleaned text, tokens, stems, lemmas, integer text counts, repeated-word counts, ranked keyword/TF-IDF pairs, common phrase counts, dataset-backed attribute matches, possible basic issues, suggestions, cosine similarity (when computable), and the five score components.

## Limitations and future improvements

- Attribute detection is a simple phrase lookup over values found in the corpus; it can miss synonyms and ambiguous matches.
- The untrained Punkt sentence tokenizer uses general punctuation rules and may mis-handle abbreviations.
- Porter stems and noun-default WordNet lemmas are educational approximations.
- TF-IDF captures word overlap and corpus rarity, not meaning or product quality.
- The score is heuristic and has not been validated against human ratings, marketplace policies, or sales.
- Future work could add language-aware lemmatization, a reviewed synonym list, user-selected CSV uploads, and evaluation against human annotations—without presenting a heuristic as a validated outcome.

## Viva / demonstration checklist

1. Open **Dataset overview** and show the actual 500-row corpus, its columns, types, and preview.
2. Analyze the example title/description and point out punctuation removal and lowercasing.
3. Compare tokens before and after stop-word removal; show a Porter stem and a lemma.
4. Explain how word frequencies, repeated terms, bigrams, and trigrams are counted.
5. Show that TF-IDF is fitted on the corpus, then used to score terms in the entered listing.
6. Demonstrate a dataset-backed color/material/size/feature match and explain why “no match” does not mean “missing.”
7. Walk through each 20-point scoring component and its rule in **Scoring guide**.
8. State the limitations: corpus-dependent keywords, simple rules, and an unvalidated educational score.
