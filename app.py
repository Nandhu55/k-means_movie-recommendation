import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("🎬 Movie Recommendation System using K-Means")

st.write(
    """
    This application recommends movies using:
    
    - K-Means Clustering
    - Business Logic Clustering
    - Genre-Based Filtering
    """
)

# Load Dataset

movies = pd.read_csv("tmdb_5000_movies.csv")

# Feature Selection

features = [
    'budget',
    'popularity',
    'revenue',
    'runtime',
    'vote_average',
    'vote_count'
]

X = movies[features]

# Handle Missing Values


X = X.fillna(0)

# Outlier Detection & Handling


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

# Feature Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# K-Means Clustering


model = KMeans(
    n_clusters=5,
    random_state=42
)

movies['Cluster'] = model.fit_predict(X_scaled)

# Cluster Names

cluster_names = {
    0: "Blockbuster Movies",
    1: "Popular Commercial Movies",
    2: "Average Rated Movies",
    3: "Low Budget Movies",
    4: "Independent/Niche Movies"
}

movies['Cluster_Name'] = movies['Cluster'].map(cluster_names)

# Genre Selection


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
    "🎭 Select Movie Genre",
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
        [
            'title',
            'Cluster_Name',
            'vote_average',
            'popularity'
        ]
    ]

    recommendations = recommendations.sort_values(
        by='popularity',
        ascending=False
    )

    st.subheader(f"🔥 Top {selected_genre} Movies")

    st.dataframe(
        recommendations.head(15),
        use_container_width=True
    )

#

st.sidebar.header("📌 Cluster Information")

for key, value in cluster_names.items():
    st.sidebar.write(f"Cluster {key} → {value}")



st.sidebar.header("📊 Dataset Information")

st.sidebar.write(f"Rows: {movies.shape[0]}")
st.sidebar.write(f"Columns: {movies.shape[1]}")



st.sidebar.header("⚙️ Features Used")

for feature in features:
    st.sidebar.write(f"✔️ {feature}")

# Cluster Counts
st.sidebar.header("🎬 Movies per Cluster")

cluster_counts = movies['Cluster_Name'].value_counts()

st.sidebar.write(cluster_counts)



st.markdown("---")
