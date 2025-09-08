import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# File paths
# ----------------------------
BASE_DIR = Path(__file__).parent
ratings_files = [
    BASE_DIR / "Downloads" / "ratings1.parquet",
    BASE_DIR / "Downloads" / "ratings2.parquet"
]
movies_path = BASE_DIR / "Downloads" / "movies.parquet"

# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data
def load_ratings():
    # Read all split files and concatenate
    dfs = [pd.read_parquet(f, engine="fastparquet") for f in ratings_files]
    return pd.concat(dfs, ignore_index=True)

@st.cache_data
def load_movies():
    return pd.read_parquet(movies_path, engine="fastparquet")

ratings = load_ratings()
movies = load_movies()

# ----------------------------
# Build sparse matrix once (cached)
# ----------------------------
@st.cache_data
def build_sparse_matrix(ratings_df):
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    
    rows = ratings_df['userId'].map(user_to_idx)
    cols = ratings_df['movieId'].map(movie_to_idx)
    vals = ratings_df['rating'].values
    user_item_matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    
    return user_item_matrix, user_ids, movie_ids, user_to_idx, movie_to_idx

user_item_matrix, user_ids, movie_ids, user_to_idx, movie_to_idx = build_sparse_matrix(ratings)

# ----------------------------
# Session state for favorites
# ----------------------------
if "favorites" not in st.session_state:
    st.session_state.favorites = {}  # {title: rating}
if "last_favorites" not in st.session_state:
    st.session_state.last_favorites = []

# ----------------------------
# UI
# ----------------------------
st.title("🎬 Movie Recommendation App")

search_string = st.text_input("Search for a movie title:")

if search_string:
    matches = movies[movies['title'].str.lower().str.contains(search_string.lower())]
    if not matches.empty:
        selected = st.selectbox("Select a movie from results:", matches['title'].head(20))
        if st.button("➕ Add to Favorites"):
            if selected:
                if len(st.session_state.favorites) < 4:
                    st.session_state.favorites[selected] = 5
                else:
                    st.warning("You already selected 4 movies!")
    else:
        st.warning("No matches found.")

# Show current favorites
if st.session_state.favorites:
    st.subheader("Your Favorite Movies (max 4):")
    fav_df = pd.DataFrame(list(st.session_state.favorites.items()), columns=["Title", "Rating"])
    st.table(fav_df)

# ----------------------------
# Compute recommendations only when favorites change
# ----------------------------
if len(st.session_state.favorites) == 4 and st.session_state.favorites != st.session_state.last_favorites:
    st.subheader("🔍 Generating recommendations...")

    # Map movie titles to IDs
    movie_title_to_id = dict(zip(movies['title'], movies['movieId']))

    # Build input vector
    my_vector = np.zeros(len(movie_ids))
    for title, rating in st.session_state.favorites.items():
        if title in movie_title_to_id and movie_title_to_id[title] in movie_to_idx:
            my_vector[movie_to_idx[movie_title_to_id[title]]] = rating

    # Indices of favorite movies
    my_rated_movies = [
        movie_to_idx[movie_title_to_id[title]]
        for title in st.session_state.favorites
        if title in movie_title_to_id and movie_title_to_id[title] in movie_to_idx
    ]

    # Select only users who rated at least one favorite
    mask = np.array((user_item_matrix[:, my_rated_movies] > 0).sum(axis=1)).flatten() > 0
    relevant_users_matrix = user_item_matrix[mask, :]
    relevant_user_ids = user_ids[mask]

    # Cosine similarity
    similarities = cosine_similarity([my_vector], relevant_users_matrix)[0]
    similar_users_df = pd.DataFrame({'userId': relevant_user_ids, 'similarity': similarities})
    similar_users_df = similar_users_df.sort_values(by='similarity', ascending=False)

    # Recommendation
    top_users = similar_users_df.head(25)['userId'].values
    top_users_ratings = ratings[ratings['userId'].isin(top_users)]

    # Remove already rated movies
    my_rated_movie_ids = [movie_title_to_id[title] for title in st.session_state.favorites if title in movie_title_to_id]
    top_users_ratings = top_users_ratings[~top_users_ratings['movieId'].isin(my_rated_movie_ids)]
    top_users_ratings = top_users_ratings.merge(similar_users_df[['userId','similarity']], on='userId', how='left')
    top_users_ratings['weighted_rating'] = top_users_ratings['rating'] * top_users_ratings['similarity']

    recommendation_scores = top_users_ratings.groupby('movieId').apply(
        lambda x: x['weighted_rating'].sum() / x['similarity'].sum()
    ).sort_values(ascending=False)

    recommendations = recommendation_scores.reset_index().merge(movies, on='movieId')[['title', 0]]
    recommendations.columns = ['title', 'score out of 5']
    recommendations['score out of 5'] = recommendations['score out of 5'].round(2)

    # Fix titles
    def fix_title(title):
        if ", The" in title:
            return "The " + title.replace(", The", "")
        elif ", A" in title:
            return "A " + title.replace(", A", "")
        elif ", An" in title:
            return "An " + title.replace(", An", "")
        else:
            return title

    recommendations['title'] = recommendations['title'].apply(fix_title)

    # Show top 25
    st.subheader("🎯 Top 25 Recommended Movies for You:")
    st.dataframe(recommendations.head(25))

    # Update last favorites
    st.session_state.last_favorites = st.session_state.favorites.copy()
