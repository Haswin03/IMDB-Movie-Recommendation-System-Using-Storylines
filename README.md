# 2024 IMDb NLP Movie Recommender

An end-to-end Data Science pipeline that scrapes 2024 IMDb feature films and utilizes Natural Language Processing (NLP) to recommend movies based on thematic storyline similarity. Unlike traditional recommenders, this engine focuses on "vibe-based" matching using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🔗 Live Demo
Check out the interactive web application here:  
👉 **[Click here to Launch the App](https://imdb-movie-recommendation-system-using-storylines.streamlit.app/)** 🚀

---

## Basic Workflow

The project is divided into a 4-tier modular architecture to ensure scalability and maintainability:

1.  **Extraction (Scraper):** A Selenium-based engine that bypasses IMDb's infinite scroll limits using a "Date Chunking" strategy.
2.  **Processing (NLP):** A cleaning pipeline using `NLTK` to tokenize, remove stopwords, and normalize storylines.
3.  **Vectorization (Model):** Transforming text into numerical features using `Scikit-Learn`'s TF-IDF, followed by a Cosine Similarity calculation.
4.  **Deployment (Interface):** A dynamic `Streamlit` web application for real-time user interaction and result visualization.

---

## Technical Stack & Metrics

### Technical Tags
* **Languages:** Python 3.10+
* **Environment:** VS Code, Git, Virtual Environments (venv)
* **Data Manipulation:** Pandas, NumPy

### Libraries & Tools
| Category | Technology | Usage in Project |
| :--- | :--- | :--- |
| **Web Scraping** | **Selenium** | Dynamic content extraction from IMDb's React-based UI. |
| **NLP** | **NLTK, SpaCy** | Tokenization, Stopword removal, and Lemmatization. |
| **Machine Learning** | **Scikit-learn** | TF-IDF Vectorization & Count Vectorizer analysis. |
| **Algorithms** | **Cosine Similarity** | Mathematical distance calculation between movie vectors. |
| **Web Framework** | **Streamlit** | Interactive UI and Cloud Deployment. |
| **Visualization** | **Matplotlib, Seaborn** | Similarity distribution and Match Confidence charts. |

### Performance Metrics
* **Vectorization Strategy:** TF-IDF (Term Frequency-Inverse Document Frequency) with `min_df=2`.
* **Similarity Metric:** Cosine Similarity ($Range: 0 \text{ to } 1$).
* **Memory Optimization:** Designed to run within **1GB RAM** for free-tier cloud hosting.
* **Data Integrity:** automated `drop_duplicates` logic during the scraping and merging phases.

### Mathematical Core
The engine calculates the similarity between the user's input ($A$) and the movie dataset ($B$) using the Cosine Similarity formula:

$$Similarity = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

---

## 📂 Project Structure

```text
IMDB-Recommender/
├── data/               # Raw and Batch CSV files
├── models/             # Serialized (.pkl) models and matrices
├── src/                # Modular Source Code
│   ├── scraper.py      # Selenium scraping logic
│   └── processor.py    # NLP & Model generation logic
├── app.py              # Main Streamlit Application
├── requirements.txt    # Production dependencies
└── .gitignore          # Prevents heavy .pkl files from bloating the repo
```

Execution Instructions
Follow these steps to replicate the project locally.

1. Environment Setup
Clone the repository and install the dependencies:

git clone [https://github.com/Haswin03/IMDB-Movie-Recommendation-System-Using-Storylines.git](https://github.com/Haswin03/IMDB-Movie-Recommendation-System-Using-Storylines.git)
cd IMDB-Movie-Recommendation-System-Using-Storylines
pip install -r requirements.txt

2. Data Extraction (Optional)

If you wish to scrape fresh data from IMDb:
python src/scraper.py

Note: This script handles "Infinite Scroll" by processing monthly batches to optimize RAM usage.

3. Model Generation
Process the raw CSV and generate the similarity matrices:

python src/processor.py
This will create the models/ directory and populate it with .pkl artifacts.

4. Launch the Application
Start the Streamlit UI:

streamlit run app.py

👨‍💻 Author
Ashwanth Ram D
B.E. Computer Science & Engineering Graduate (2021-2025)
