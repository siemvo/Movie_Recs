import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Run this app with `streamlit run app.py` in terminal in this location
# ----------------------------
BASE_DIR = Path(__file__).parent
ratings_path = BASE_DIR / "Downloads" / "ratings.csv"
movies_path = BASE_DIR / "Downloads" / "movies.csv"

@st.cache_data
def load_data():
    #ratings = pd.read_csv(ratings_path)
    #movies = pd.read_csv(movies_path)
    #return ratings, movies
    
    # Load raw data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Count ratings per user
    user_counts = ratings['userId'].value_counts()

    # Keep only users with <= 1000 ratings
    valid_users = user_counts[user_counts <= 1000].index
    ratings = ratings[ratings['userId'].isin(valid_users)]

    return ratings, movies


with st.spinner("Loading and filtering data: This will take 10-20 seconds"):
    ratings, movies = load_data()


# ----------------------------
# Session state for favorites
# ----------------------------
if "favorites" not in st.session_state:
    st.session_state.favorites = {}  # {title: rating}

# ----------------------------
# UI for selecting favorite movies
# ----------------------------
st.title("Movie Recommendation App")

search_string = st.text_input("Search for a movie title:")

if search_string:
    matches = movies[movies['title'].str.lower().str.contains(search_string.lower())]

    if not matches.empty:
        selected = st.selectbox("Select a movie from results (4 total):", matches['title'].head(20))
        if st.button("➕ Add to Favorites"):
            if selected:
                if len(st.session_state.favorites) < 4:
                    st.session_state.favorites[selected] = 5  # default rating = 5
                else:
                    st.warning("You already selected 4 movies!")
    else:
        st.warning("No matches found.")

# ----------------------------
# Show selected favorites
# ----------------------------
if st.session_state.favorites:
    st.subheader("Your Favorite Movies:")

    # Display each favorite with a remove button
    for title, rating in list(st.session_state.favorites.items()):
        col1, col2 = st.columns([4,1])
        with col1:
            #st.write(f" {title} (Rating: {rating})")
            st.write(f" {title}")
        with col2:
            if st.button(f"❌ Remove", key=f"remove_{title}"):
                del st.session_state.favorites[title]
                st.rerun()  # Refresh to update recommendations


# ----------------------------
# Build recommendations when 4 movies selected
# ----------------------------
if len(st.session_state.favorites) == 4:
    st.subheader("🔍 Generating recommendations...")

    filtered_data = ratings

    # Map user and movie IDs
    user_ids = filtered_data['userId'].unique()
    movie_ids = filtered_data['movieId'].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # Sparse user-item matrix
    rows = filtered_data['userId'].map(user_to_idx)
    cols = filtered_data['movieId'].map(movie_to_idx)
    vals = filtered_data['rating'].values
    user_item_matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    # Map movie titles to IDs
    movie_title_to_id = dict(zip(movies['title'], movies['movieId']))

    # Build input vector
    my_vector = np.zeros(len(movie_ids))
    for title, rating in st.session_state.favorites.items():
        if title in movie_title_to_id:
            mid = movie_title_to_id[title]
            if mid in movie_to_idx:
                my_vector[movie_to_idx[mid]] = rating

    # Indices of favorite movies
    my_rated_movies = [
        movie_to_idx[movie_title_to_id[title]]
        for title in st.session_state.favorites
        if title in movie_title_to_id and movie_title_to_id[title] in movie_to_idx
    ]

    # Select only users who rated at least one favorite (sparse-safe)
    mask = np.array((user_item_matrix[:, my_rated_movies] > 0).sum(axis=1)).flatten() > 0
    relevant_users_matrix = user_item_matrix[mask, :]

    # Cosine similarity
    similarities = cosine_similarity([my_vector], relevant_users_matrix)[0]
    relevant_user_ids = user_ids[mask]
    similar_users_df = pd.DataFrame({'userId': relevant_user_ids, 'similarity': similarities})
    similar_users_df = similar_users_df.sort_values(by='similarity', ascending=False)  # exact match with notebook

    # ----------------------------
    # Recommendation
    # ----------------------------
    top_users = similar_users_df.head(25)['userId'].values
    top_users_ratings = filtered_data[filtered_data['userId'].isin(top_users)]

    # Remove movies already rated
    my_rated_movie_ids = [
        movie_title_to_id[title] for title in st.session_state.favorites if title in movie_title_to_id
    ]
    top_users_ratings = top_users_ratings[~top_users_ratings['movieId'].isin(my_rated_movie_ids)]
    top_users_ratings = top_users_ratings.merge(similar_users_df[['userId','similarity']], on='userId', how='left')

    # Weighted rating
    top_users_ratings['weighted_rating'] = top_users_ratings['rating'] * top_users_ratings['similarity']

    # Aggregate
    recommendation_scores = top_users_ratings.groupby('movieId').apply(
        lambda x: x['weighted_rating'].sum() / x['similarity'].sum()
    ).sort_values(ascending=False)

    # Map back to titles
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

    # Show recommendations
    st.subheader("🎯 Top Recommended Movies for You:")
    st.dataframe(recommendations.head(25))
