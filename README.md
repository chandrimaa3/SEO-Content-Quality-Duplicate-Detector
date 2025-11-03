# SEO Content Quality & Duplicate Detector

## Project Overview
This project builds a machine learning pipeline for analyzing web content for SEO quality and duplicate detection using a dataset of URLs and HTML content. The Jupyter notebook (notebooks/seo_pipeline.ipynb) implements data loading, HTML parsing, feature engineering (readability, keyword density, BERT embeddings), duplicate detection via cosine similarity, and content quality scoring with a regression model. The Streamlit app (app.py) deploys this pipeline for real-time URL analysis, providing an interactive UI for users to input URLs and view results including quality labels, thin content flags, and similar content matches.
Setup Instructions

## Clone the repository:
```
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
```

## Install dependencies:
```
pip install -r requirements.txt
```

## Run the Jupyter notebook for analysis:
```
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Run the Streamlit app locally:
```
streamlit run app.py
```

## Quick Start

+ Open the notebook and execute cells sequentially to load data.csv, process features, detect duplicates, train the model, and test the analyze_url function.
+ For example, analyze a sample URL:
```{python}
sample_url = "https://www.cisa.gov/topics/cybersecurity-best-practices"
result = analyze_url(sample_url)
print(result)
```

This will output a dictionary with title, word count, readability score, quality label, thin content flag, and similar URLs from the dataset.

## Deployed Streamlit URL
The app is deployed at: https://seo-content-detector.streamlit.app


## Key Decisions

+ **Libraries:** Used BeautifulSoup for HTML parsing (robust and lightweight), textstat for readability metrics (standard Flesch scores), BERT via transformers for embeddings (captures semantic similarity effectively), scikit-learn for modeling (simple and efficient for small datasets), and Streamlit for deployment (easy interactive UI for ML apps).
+ **HTML Parsing Approach:** Extract title and strip scripts/styles with BeautifulSoup to focus on clean main text, ensuring accurate feature computation without noise.
+ **Similarity Threshold Rationale:** Set at 0.9 for cosine similarity on embeddings to identify near-duplicates conservatively, balancing false positives and capturing high overlaps in content.
+ **Model Selection Reasoning:** Linear Regression for quality scoring in the notebook (predicts continuous score from features; MSE 10.48, R² 0.86); switched to RandomForestClassifier in Streamlit for categorical labels (High/Medium/Low) based on synthetic rules for better interpretability in the app.

## Results Summary

+ **Model Performance:** Linear Regression achieved MSE of 10.4823 and R² of 0.8585 on test set for quality score prediction. RandomForestClassifier in app uses synthetic labels for classification.
+ **Duplicates Found:** 0 near-duplicate pairs with similarity >0.9 in the 81-row dataset.
+ **Sample Quality Scores:** For https://www.cisa.gov/topics/cybersecurity-best-practices: Word count 1171, Readability -2.46 (Very Difficult), Quality Label 'Low', Not thin, 8 similar URLs (e.g., 0.92 similarity to https://www.cm-alliance.com/cybersecurity-blog).

## Limitations

+ Relies on synthetic/dummy labels for quality; needs real labeled data for improved accuracy.
+ Duplicate detection limited to the provided dataset; doesn't handle large-scale or dynamic indexes.
+ BERT embeddings are computationally intensive; may slow down on low-resource environments.
