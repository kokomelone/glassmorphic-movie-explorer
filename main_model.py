import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
movies_raw = pd.read_csv("u.item", sep="|", encoding="latin-1", header=None)
movies_raw.columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + \
                    pd.read_csv("u.genre", sep="|", names=["genre", "genre_id"], engine="python")["genre"].tolist()
ratings = pd.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

# Preprocess genre
genre_list = pd.read_csv("u.genre", sep="|", names=["genre", "genre_id"], engine="python")["genre"].tolist()
movies_raw["genres_str"] = movies_raw[genre_list].apply(lambda row: ' '.join([g for g in genre_list if row[g] == 1]), axis=1)

# Ratings info
avg_ratings = ratings.groupby("movie_id")["rating"].mean().reset_index()
movies = movies_raw.merge(avg_ratings, on="movie_id", how="left")
movies["avg_rating"] = movies["rating"].fillna(0)
movies["release_year"] = pd.to_datetime(movies["release_date"], errors='coerce').dt.year.fillna(0).astype(int)

# Preprocessing for model
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

movies["full_text"] = (movies["title"] + " " + movies["genres_str"]).apply(preprocess)
bm25_corpus = movies["full_text"].apply(str.split).tolist()
bm25 = BM25Okapi(bm25_corpus)

# Dummy XGBoost model trained using genre_match, cosine_sim, BM25_score (for now we simulate with average rating)
scaler = MinMaxScaler()

def get_recommendations(keyword: str, genre: str, top_n: int = 10, sort_by: str = ""):
    query_tokens = preprocess(keyword).split()
    movies["BM25_score"] = bm25.get_scores(query_tokens)

    def genre_overlap(row):
        return len(set(row["genres_str"].split()) & set(query_tokens))

    def cosine_sim(row):
        try:
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform([keyword, row["full_text"]])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0

    movies["genre_match"] = movies.apply(genre_overlap, axis=1)
    movies["cosine_sim"] = movies.apply(cosine_sim, axis=1)

    # Combine features and score with simulated XGB rank (scaled sum)
    features = movies[["BM25_score", "genre_match", "cosine_sim", "avg_rating"]].fillna(0)
    scaled = scaler.fit_transform(features)
    movies["xgb_score"] = scaled.sum(axis=1)

    # Filter
    filtered = movies.copy()
    if genre:
        filtered = filtered[filtered["genres_str"].str.contains(genre, case=False)]

    if keyword:
        filtered = filtered[filtered["title"].str.lower().str.contains(keyword.lower()) | (filtered["xgb_score"] > 0)]

    if sort_by == "rating":
        filtered = filtered.sort_values("avg_rating", ascending=False)
    elif sort_by == "year":
        filtered = filtered.sort_values("release_year", ascending=False)
    else:
        filtered = filtered.sort_values("xgb_score", ascending=False)

    results = filtered.head(top_n)[["title", "genres_str", "release_year", "avg_rating"]].to_dict(orient="records")
    return results