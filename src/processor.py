import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CENTRAL CONFIGURATION (Update paths here)
# ==========================================
DATA_PATH = '../data/movies_2024.csv'
TFIDF_MODEL_FILE = '../models/tfidf_model.pkl'
TFIDF_MATRIX_FILE = '../models/tfidf_matrix.pkl'
COSINE_SIM_FILE = '../models/cosine_sim.pkl'
MOVIE_DF_FILE = '../models/movie_df.pkl'

# ==========================================
# 1. NLTK RESOURCE MANAGEMENT
# ==========================================
def setup_nltk_resources():
    """
    Ensures necessary NLP libraries are available locally.
    Download check prevents redundant downloads across sessions.
    """
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            search_path = f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}'
            nltk.data.find(search_path)
        except LookupError:
            print(f"Downloading missing NLTK resource: {res}...")
            nltk.download(res, quiet=True)

# Initialize NLP tools
setup_nltk_resources()
stop_words = set(stopwords.words('english'))

# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================
def clean_storyline(text):
    """
    Cleans raw IMDb text by lowercasing, removing special characters,
    tokenizing, and filtering out English stopwords.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words and word.strip() != '']
    
    return ' '.join(cleaned_tokens)

# ==========================================
# 3. MAIN PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    """
    Executes the full data science pipeline:
    Loading -> Cleaning -> Vectorization -> Similarity -> Export
    """
    
    # Load the scraped dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully from {DATA_PATH}. Rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: '{DATA_PATH}' not found. Please verify the file path.")
        return

    # Preprocessing
    print("Preprocessing storylines...")
    df['Clean_Storyline'] = df['Storyline'].apply(clean_storyline)
    
    # Filtering empty results
    df = df[df['Clean_Storyline'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    # Vectorization
    print("Training TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(min_df=2, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Clean_Storyline'])
    print(f"Vectorization complete. Vocabulary size: {len(tfidf.get_feature_names_out())}")

    # Similarity Calculation
    print("Calculating similarity scores...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Exporting artifacts
    print("Saving model artifacts...")
    
    with open(TFIDF_MODEL_FILE, 'wb') as f:
        pickle.dump(tfidf, f)
        
    with open(TFIDF_MATRIX_FILE, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    with open(COSINE_SIM_FILE, 'wb') as f:
        pickle.dump(cosine_sim, f)
        
    with open(MOVIE_DF_FILE, 'wb') as f:
        pickle.dump(df, f)

    print("Process finished. All artifacts saved successfully.")

if __name__ == "__main__":
    run_pipeline()