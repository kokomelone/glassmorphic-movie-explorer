# glassmorphic-movie-explorer
A hybrid movie recommendation system combining Information Retrieval (TF-IDF, BM25) and Machine Learning (Logistic Regression, Random Forest, XGBoost). Includes a glassmorphic web interface built with Flask.

## Features

- Search by genre and keywords
- Hybrid ranking using BM25, cosine similarity, and ML models
- Personalized results based on learned behavior
- Glassmorphic web UI with real-time recommendations
- Model evaluation using NDCG and Precision@K

## Technologies Used

**Backend:**
- Python, Flask
- Pandas, NumPy
- Scikit-learn, XGBoost
- NLTK (for preprocessing)
- rank_bm25

**Frontend:**
- HTML, CSS, Bootstrap
- Glassmorphism-based styling

**Data:**
- MovieLens 100K dataset  
  [Download here](https://grouplens.org/datasets/movielens/100k/)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/glassmorphic-movie-explorer.git
cd glassmorphic-movie-explorer
