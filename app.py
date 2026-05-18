import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Get movie recommendations based on genres using K-Means Clustering."
)

movies = pd.read_csv("tmdb_5000_movies.csv")

# Features for Clustering
features = [
    'budget',
    'popularity',
    'revenue',
    'runtime',
    'vote_average',
    'vote_count'
]

X = movies[features]

X = X.fillna(0)

for column in features:

    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X[column] = np.where(
        X[column] > upper_bound,
        upper_bound,
        X[column]
    )

    X[column] = np.where(
        X[column] < lower_bound,
        lower_bound,
        X[column]
    )

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=5,
    random_state=42
)

movies['Cluster'] = model.fit_predict(X_scaled)

genres_list = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Horror",
    "Romance",
    "Science Fiction",
    "Thriller"
]

selected_genre = st.selectbox(
    "Select Movie Genre",
    genres_list
)

if st.button("Recommend Movies"):

    filtered_movies = movies[
        movies['genres'].str.contains(
            selected_genre,
            case=False,
            na=False
        )
    ]

    recommendations = filtered_movies[
        ['title', 'vote_average', 'popularity']
    ]

    recommendations = recommendations.sort_values(
        by='popularity',
        ascending=False
    )

    st.subheader(f"Top {selected_genre} Movies")

    st.dataframe(
        recommendations.head(10),
        use_container_width=True
    )

st.sidebar.header("Project Information")

st.sidebar.write(
    """
    ### Algorithms Used
    - K-Means Clustering
    - StandardScaler

    ### Features Used
    - Budget
    - Popularity
    - Revenue
    - Runtime
    - Vote Average
    - Vote Count
    """
)

st.sidebar.header("Dataset Shape")

st.sidebar.write(movies.shape)