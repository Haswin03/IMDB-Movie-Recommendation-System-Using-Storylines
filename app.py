import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="2024 IMDb Recommender",
    layout="centered"
)

st.title("2024 Movie Matchmaker")
st.write("Type in a storyline, plot, or vibe, and our NLP engine will find the top 5 closest matches from 2024!")

# ==========================================
# 2. LOAD MODELS (CACHED FOR SPEED)
# ==========================================
# st.cache_resource ensures we only load these heavy files once
@st.cache_resource
def load_models():
    # Paths assume app.py is in the root folder, right next to the 'models' folder
    model_dir = "models"
    
    try:
        with open(os.path.join(model_dir, 'tfidf_model.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
        with open(os.path.join(model_dir, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(os.path.join(model_dir, 'movie_df.pkl'), 'rb') as f:
            df = pickle.load(f)
        return tfidf, tfidf_matrix, df
    except FileNotFoundError:
        st.error("Error: Model files not found. Did you run `python src/processor.py` first?")
        st.stop()

tfidf_model, tfidf_matrix, df = load_models()

# ==========================================
# 3. NLP SETUP
# ==========================================
@st.cache_resource
def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            search_path = f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}'
            nltk.data.find(search_path)
        except LookupError:
            nltk.download(res, quiet=True)

setup_nltk()
stop_words = set(stopwords.words('english'))

def clean_input(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words and word.strip() != '']
    return ' '.join(cleaned_tokens)

# ==========================================
# 4. FRONTEND UI & LOGIC
# ==========================================

# The Input Area
user_plot = st.text_area(
    "Describe the movie you want to watch:",
    placeholder="Example: A sci-fi adventure where astronauts travel through a wormhole to save humanity.",
    height=120
)

# The Execution Button
if st.button("Find My Movie", type="primary", use_container_width=True):
    if user_plot.strip() == "":
        st.warning("Please type a storyline first!")
    else:
        with st.spinner("Analyzing your storyline..."):
            # 1. Clean the user's input
            cleaned_input = clean_input(user_plot)
            
            # 2. Convert to vector
            user_vector = tfidf_model.transform([cleaned_input])
            
            # 3. Calculate similarity against all movies
            sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
            
            # 4. Rank and extract top 5
            indexed_scores = list(enumerate(sim_scores))
            ranked_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
            top_5 = ranked_scores[0:5] # Get top 5 matches
            
            # 5. Display Results
            st.markdown("### 🎬 Top 5 Recommendations")
            
            for rank, item in enumerate(top_5, 1):
                idx = item[0]
                score = round(item[1] * 100, 1)
                
                # Fetch movie details
                movie_name = df.iloc[idx]['Movie_Name']
                storyline = df.iloc[idx]['Storyline']
                
                # Use Streamlit containers for a clean, dynamic card-like look
                with st.container():
                    st.subheader(f"#{rank}: {movie_name} (Match: {score}%)")
                    st.write(f"**Plot:** {storyline}")
                    st.divider() # Adds a nice horizontal line between movies