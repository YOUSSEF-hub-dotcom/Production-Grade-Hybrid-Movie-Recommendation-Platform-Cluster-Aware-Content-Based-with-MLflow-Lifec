import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import ast
import logging

logger = logging.getLogger("Model")

def build_and_train_model(df_full, max_features, n_components):
    logger.info("=================>> Build Machine Learning Model")

    tfidf = TfidfVectorizer(max_features=max_features)
    text_features = tfidf.fit_transform(df_full['final_text'])
    logger.info("Text Feature to numeric is Successful")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_reduced = svd.fit_transform(text_features)
    logger.info(f"Explained variance by SVD: {svd.explained_variance_ratio_.sum():.4f}")

    df_full['genres_list'] = df_full['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    df_full['genres_list'] = df_full['genres_list'].apply(
        lambda x: [g['name'] for g in x]
    )

    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df_full['genres_list'])

    numeric_features = df_full[['budget', 'popularity', 'revenue', 'runtime']].fillna(0)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    weighted_genres = genres_encoded * 2.0

    weighted_text = text_reduced * 1.5

    X = np.hstack([weighted_text, weighted_genres, numeric_scaled])
    print("Final feature matrix shape:", X.shape)

    linkage_matrix = linkage(X, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()

    scores = {}
    for n in [2, 3, 5, 10 , 13 , 15]:
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        score = silhouette_score(X, labels)
        scores[n] = score
        logger.info(f"Silhouette Score for {n} clusters: {score:.4f}")

    n_clusters = 10

    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_full['cluster'] = cluster_labels

    scores['k_best'] = scores[n_clusters]

    logger.info(f"Cluster distribution:\n{df_full['cluster'].value_counts().to_string()}")

    cos_sim = cosine_similarity(np.hstack([text_reduced, genres_encoded]))
    logger.info(f"Similarity matrix calculated. Shape: {cos_sim.shape}")

    def recommend_movies(movie_title, df_full=df_full, similarity_matrix=cos_sim, top_n=5):
        movie_title_clean = str(movie_title).strip().lower()
        logger.debug(f"Searching for recommendations for: {movie_title_clean}")
        df_titles_clean = df_full['title_x'].str.strip().str.lower()

        if movie_title_clean not in df_titles_clean.values:
            return f"Movie '{movie_title}' not found in dataset."

        idx = df_titles_clean[df_titles_clean == movie_title_clean].index[0]
        cluster = df_full.loc[idx, 'cluster']

        cluster_indices = df_full[df_full['cluster'] == cluster].index
        sim_scores = similarity_matrix[idx][cluster_indices]

        top_indices = np.argsort(sim_scores)[::-1][1:top_n + 1]

        # CRITICAL: Return all columns for the API to work without KeyError
        recommended = df_full.iloc[cluster_indices[top_indices]].copy()
        logger.info(f"Found {len(recommended)} recommendations for '{movie_title}'")
        recommended['similarity_score'] = sim_scores[top_indices]

        return recommended

    test_movie = "Four Rooms"
    recommended = recommend_movies(test_movie)
    print(f"\nRecommendations for '{test_movie}':\n", recommended[['title_x', 'cluster', 'similarity_score']])

    plt.figure(figsize=(8, 6))
    plt.scatter(text_reduced[:, 0], text_reduced[:, 1], c=df_full['cluster'], cmap='viridis', alpha=0.5)
    plt.title("Movie Clusters Visualization")
    plt.show()

    params = {"max_features": max_features, "n_components": n_components}
    logger.info(params)

    return (
        df_full, tfidf, svd, mlb, scaler, cluster_labels,
        scores, text_reduced, cos_sim, params
    )
