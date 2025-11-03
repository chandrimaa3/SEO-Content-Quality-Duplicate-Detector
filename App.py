import os
import subprocess
import sys

# Ensure BeautifulSoup is installed
subprocess.run([sys.executable, "-m", "pip", "install", "beautifulsoup4"], check=False)

import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import textstat
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import requests

st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="Magnifying Glass",
    layout="centered"
)

st.title("SEO Content Quality & Duplicate Detector")
st.caption("Analyze any URL for SEO quality, readability, and duplicate content using AI.")

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # <-- THIS IS CRITICAL
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# Call it once
download_nltk_data()

# Predefined keywords for density calculation
keywords = ['cybersecurity', 'AI', 'artificial intelligence', 'security', 'data']

# Similarity threshold for duplicates (from your notebook)
similarity_threshold = 0.9  # Adjust if needed
thin_content_threshold = 500  # From your notebook

# Functions from your notebook (adapted slightly for app)
def extract_text_from_html(html_content):
    if pd.isna(html_content):
        return "", ""
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else "No title"
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    text = soup.get_text()
    text = ' '.join(text.split())
    return title, text

def calculate_keyword_density(text, keywords):
    if not text:
        return 0
    text = text.lower()
    words = word_tokenize(text)
    word_count = len(words)
    if word_count == 0:
        return 0
    keyword_count = sum(words.count(keyword) for keyword in keywords)
    return keyword_count / word_count

@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def generate_embedding(text, tokenizer, model):
    if not text:
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def create_quality_label(row):
    word_count = row['word_count']
    readability = row['readability_score']
    if word_count > 1500 and 50 <= readability <= 70:
        return 'High'
    elif word_count < 500 or readability < 30:
        return 'Low'
    else:
        return 'Medium'

# Load and process data on app startup (cached for performance)
@st.cache_resource
def load_and_process_data():
    df = pd.read_csv("data.csv")
    
    # Parse HTML
    df[['title', 'main_text']] = df['html_content'].apply(lambda x: pd.Series(extract_text_from_html(x)))
    
    # Feature engineering
    df['readability_score'] = df['main_text'].apply(lambda x: textstat.flesch_reading_ease(x) if x else 0)
    df['keyword_density'] = df['main_text'].apply(lambda x: calculate_keyword_density(x, keywords))
    df['word_count'] = df['main_text'].apply(lambda x: len(x.split()) if x else 0)
    df['sentence_count'] = df['main_text'].apply(lambda x: textstat.sentence_count(x) if x else 0)
    
    tokenizer, model_embedding = load_bert_model()
    df['text_embeddings'] = df['main_text'].apply(lambda x: generate_embedding(x, tokenizer, model_embedding))
    
    # Embeddings matrix for duplicate detection
    valid_embeddings = [emb for emb in df['text_embeddings'].tolist() if isinstance(emb, np.ndarray) and emb.size > 0]
    embeddings_matrix = np.vstack(valid_embeddings) if valid_embeddings else np.array([])
    
    # Synthetic labels and train model
    df['quality_label'] = df.apply(create_quality_label, axis=1)
    features = df[['readability_score', 'keyword_density', 'word_count', 'sentence_count']].copy()
    if embeddings_matrix.size > 0:
        embeddings_df = pd.DataFrame(valid_embeddings, index=df.index[df['text_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)])
        features = features.merge(embeddings_df, left_index=True, right_index=True, how='left').fillna(0)
    features.columns = features.columns.astype(str)
    target = df['quality_label']
    
    X_train, _, y_train, _ = train_test_split(features, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return df, embeddings_matrix, model, tokenizer, model_embedding

# Load data and model once
df, embeddings_matrix, quality_model, tokenizer, bert_model = load_and_process_data()

def interpret_readability(score):
    if score >= 90: return "Very Easy (5th grade)"
    elif score >= 80: return "Easy (6th grade)"
    elif score >= 70: return "Fairly Easy (7th grade)"
    elif score >= 60: return "Standard (8th–9th grade)"
    elif score >= 50: return "Fairly Difficult (10th–12th grade)"
    elif score >= 30: return "Difficult (College)"
    else: return "Very Difficult (Graduate)"
        
# Analyze URL function (from your notebook)
def analyze_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch URL: {e}"}
    
    title, main_text = extract_text_from_html(html_content)
    if not main_text:
        return {"error": "Could not extract main text from the URL."}
    
    readability_score = textstat.flesch_reading_ease(main_text) if main_text else 0
    keyword_density = calculate_keyword_density(main_text, keywords)
    word_count = len(main_text.split()) if main_text else 0
    sentence_count = textstat.sentence_count(main_text) if main_text else 0
    text_embedding = generate_embedding(main_text, tokenizer, bert_model)
    
    # Prepare features for quality scoring
    features_dict = {
        'readability_score': readability_score,
        'keyword_density': keyword_density,
        'word_count': word_count,
        'sentence_count': sentence_count
    }
    for i, emb_val in enumerate(text_embedding):
        features_dict[str(i)] = emb_val
    features_df = pd.DataFrame([features_dict])
    features_df.columns = features_df.columns.astype(str)
    
    # Quality scoring
    predicted_quality_label = quality_model.predict(features_df)[0]
    
    # Duplicate detection
    near_duplicates_found = []
    if embeddings_matrix.shape[0] > 0:
        similarity_scores = cosine_similarity([text_embedding], embeddings_matrix)[0]
        duplicate_indices = np.where(similarity_scores > similarity_threshold)[0]
        for idx in duplicate_indices:
            if idx < len(df) and df.iloc[idx]['url'] != url:
                near_duplicates_found.append({
                    'url': df.iloc[idx]['url'],
                    'similarity': float(similarity_scores[idx])
                })
    
    # Thin content
    is_thin = word_count < thin_content_threshold
    
    return {
        "url": url,
        "title": title,
        "word_count": word_count,
        "readability": readability_score,
        "quality_label": predicted_quality_label,
        "is_thin": is_thin,
        "similar_to": near_duplicates_found
    }

# Streamlit UI
st.title("SEO Content Quality & Duplicate Detector")

url = st.text_input("Enter URL to analyze:")
if st.button("Analyze"):
    if url:
        with st.spinner("Analyzing..."):
            result = analyze_url(url)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Analysis Complete!")
            # === 1. Title ===
            st.markdown(f"**Title:** {result['title']}")

            # === 2. Word Count ===
            st.markdown(f"**Word Count:** `{result['word_count']}`")

            # === 3. Readability (with interpretation) ===
            score = result["readability"]
            level = interpret_readability(score)
            st.markdown(f"**Readability Score:** `{score:.2f}` — *{level}*")

            # === 4. Quality Label (color-coded) ===
            quality = result["quality_label"]
            color = {"High": "#2E8B57", "Medium": "#FFA500", "Low": "#DC143C"}.get(quality, "#808080")
            st.markdown(f"**Quality Label:** <span style='color:{color};font-weight:bold'>{quality}</span>", unsafe_allow_html=True)

            # === 5. Thin Content ===
            thin_text = "Yes" if result["is_thin"] else "No"
            thin_color = "#DC143C" if result["is_thin"] else "#2E8B57"
            st.markdown(f"**Is Thin Content?** <span style='color:{thin_color}'>{thin_text}</span>", unsafe_allow_html=True)

            # === 6. Similar Content (clickable links) ===
            if result["similar_to"]:
                st.markdown("**Similar Content Found:**")
                for sim in result["similar_to"]:
                    url = sim['url']
                    sim_score = sim['similarity']
                    st.markdown(f"• [{url}]({url}) *(Similarity: {sim_score:.2f})*")
            else:
                st.markdown("**No Similar Content Found.**")

            # === 7. Export Button ===
            if st.button("Download Report as CSV"):
                report_data = {
                    "URL": result["url"],
                    "Title": result["title"],
                    "Word Count": result["word_count"],
                    "Readability Score": result["readability"],
                    "Quality Label": result["quality_label"],
                    "Is Thin Content": result["is_thin"],
                    "Similar URLs": "; ".join([f"{s['url']} ({s['similarity']:.2f})" for s in result["similar_to"]])
                }
                report_df = pd.DataFrame([report_data])
                csv = report_df.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"seo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            st.write("**Word Count:**", result["word_count"])
            st.write("**Readability Score:**", result["readability"])
            st.write("**Quality Label:**", result["quality_label"])
            st.write("**Is Thin Content?**", "Yes" if result["is_thin"] else "No")
            
            if result["similar_to"]:
                st.write("**Similar Content Found:**")
                for sim in result["similar_to"]:
                    st.write(f"- {sim['url']} (Similarity: {sim['similarity']:.2f})")
            else:
                st.write("**No Similar Content Found.**")
    else:
        st.warning("Please enter a URL.")
