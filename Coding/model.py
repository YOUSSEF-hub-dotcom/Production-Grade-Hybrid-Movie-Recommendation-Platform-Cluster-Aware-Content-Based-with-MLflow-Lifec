import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Crucial bridge for Production Clustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import ast
import logging

logger = logging.getLogger("Model")

def build_and_train_model(df_full, max_features, n_components):
    logger.info("=================>> Build Production-Grade Machine Learning Model")

    # -------------------------------------------------------------------------
    # 1. THE FIREWALL: Train-Test Splitting for Production Lifecycle
    # -------------------------------------------------------------------------
    logger.info("Executing Train-Test Split to isolate evaluation streams...")
    df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=42)
    df_train = df_train.copy().reset_index(drop=True)
    df_test = df_test.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2. ANTI-DATA LEAKAGE NUMERICAL IMPUTATION
    # -------------------------------------------------------------------------
    # Computing runtime median STRICTLY from Training Set to avoid data bleeding
    runtime_median = df_train['runtime'].median()
    logger.info(f"Calculated Training Runtime Median: {runtime_median}")
    
    df_train['runtime'] = df_train['runtime'].fillna(runtime_median)
    df_test['runtime'] = df_test['runtime'].fillna(runtime_median)
    
    # Filling remaining numerical NaNs safely with 0
    numeric_cols = ['budget', 'popularity', 'revenue', 'runtime']
    df_train[numeric_cols] = df_train[numeric_cols].fillna(0)
    df_test[numeric_cols] = df_test[numeric_cols].fillna(0)

    # -------------------------------------------------------------------------
    # 3. TEXT VECTORIZATION & DIMENSIONALITY REDUCTION (Isolated Pipeline)
    # -------------------------------------------------------------------------
    logger.info("Fitting TF-IDF Vectorizer on Training Corpus...")
    tfidf = TfidfVectorizer(max_features=max_features)
    text_features_train = tfidf.fit_transform(df_train['final_text'])
    text_features_test = tfidf.transform(df_test['final_text']) # Transforming test via frozen vocab

    logger.info("Fitting TruncatedSVD for latent text semantic compression...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_reduced_train = svd.fit_transform(text_features_train)
    text_reduced_test = svd.transform(text_features_test) # Projecting test into train eigenspace
    logger.info(f"SVD Cumulative Explained Variance: {svd.explained_variance_ratio_.sum():.4f}")

    # -------------------------------------------------------------------------
    # 4. CATEGORICAL ENCODING & NUMERICAL SCALING (Isolated Pipeline)
    # -------------------------------------------------------------------------
    # Parsing JSON-like strings for genres
    for df_item in [df_train, df_test]:
        df_item['genres_list'] = df_item['genres'].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) else []
        ).apply(lambda x: [g['name'] for g in x])

    logger.info("Fitting MultiLabelBinarizer on Train Genres...")
    mlb = MultiLabelBinarizer()
    genres_encoded_train = mlb.fit_transform(df_train['genres_list'])
    genres_encoded_test = mlb.transform(df_test['genres_list']) # Strict template transformation

    logger.info("Fitting StandardScaler on Train Numeric Matrix...")
    scaler = StandardScaler()
    numeric_scaled_train = scaler.fit_transform(df_train[numeric_cols])
    numeric_scaled_test = scaler.transform(df_test[numeric_cols]) # Lazy transformation for test stream

    # Applying production architecture feature weight configurations
    weighted_genres_train = genres_encoded_train * 2.0
    weighted_text_train = text_reduced_train * 1.5
    
    weighted_genres_test = genres_encoded_test * 2.0
    weighted_text_test = text_reduced_test * 1.5

    # Concat all features into final distinct structural matrices
    X_train = np.hstack([weighted_text_train, weighted_genres_train, numeric_scaled_train])
    X_test = np.hstack([weighted_text_test, weighted_genres_test, numeric_scaled_test])
    print("Final Training Feature Matrix Shape:", X_train.shape)

    # -------------------------------------------------------------------------
    # 5. HIERARCHICAL CLUSTERING & STABILITY AUDITING
    # -------------------------------------------------------------------------
    logger.info("Constructing Agglomerative Linkage Tree via Ward's Method...")
    linkage_matrix = linkage(X_train, method='ward')

    # Evaluating cluster silhouette configurations internally
    scores = {}
    for n in [2, 3, 5, 10, 13, 15]:
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        score = silhouette_score(X_train, labels)
        scores[n] = score
        logger.info(f"Train Silhouette Score for {n} clusters: {score:.4f}")

    n_clusters = 10
    train_cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_train['cluster'] = train_cluster_labels
    scores['k_best'] = scores[n_clusters]

    # PRODUCTION ENGINE CORE BRIDGE:
    # Since Hierarchical Clustering lacks a '.predict()' method for inbound live API streams, 
    # we train a fast KNN classifier as a proxy router to map live data to established clusters.
    logger.info("Training KNN Cluster Proxy Router for Production live inferences...")
    cluster_router = KNeighborsClassifier(n_neighbors=5, weights='distance')
    cluster_router.fit(X_train, train_cluster_labels)

    # Safely routing test data using the proxy router (Zero Leakage Evaluation)
    df_test['cluster'] = cluster_router.predict(X_test)
    logger.info("Unseen Test Stream mapped to stable cluster addresses.")

    # Re-merging safely for consolidated analytical querying and reporting
    df_final_report = pd.concat([df_train, df_test]).reset_index(drop=True)
    logger.info(f"Global Consolidated Cluster distribution:\n{df_final_report['cluster'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # 6. SIMILARITY MATRIX & LIVE QUERY RECOMMENDATION ENGINE
    # -------------------------------------------------------------------------
    # Calculating similarity matrix globally for final pool presentation
    # Based solely on compressed semantic embeddings and encoded genres
    global_text_reduced = np.vstack([text_reduced_train, text_reduced_test])
    global_genres_encoded = np.vstack([genres_encoded_train, genres_encoded_test])
    cos_sim = cosine_similarity(np.hstack([global_text_reduced, global_genres_encoded]))
    logger.info(f"Global Similarity matrix calculated. Shape: {cos_sim.shape}")

    def recommend_movies(movie_title, df_pool=df_final_report, similarity_matrix=cos_sim, top_n=5):
        movie_title_clean = str(movie_title).strip().lower()
        df_titles_clean = df_pool['title_x'].str.strip().str.lower()

        if movie_title_clean not in df_titles_clean.values:
            return f"Movie '{movie_title}' not found in dataset."

        idx = df_titles_clean[df_titles_clean == movie_title_clean].index[0]
        cluster = df_pool.loc[idx, 'cluster']

        # Query isolated to the designated cluster block to speed up processing
        cluster_indices = df_pool[df_pool['cluster'] == cluster].index
        sim_scores = similarity_matrix[idx][cluster_indices]

        top_indices = np.argsort(sim_scores)[::-1][1:top_n + 1]
        recommended = df_pool.iloc[cluster_indices[top_indices]].copy()
        recommended['similarity_score'] = sim_scores[top_indices]

        return recommended

    # Testing Production Integrity
    test_movie = "Four Rooms"
    recommended = recommend_movies(test_movie)
    print(f"\nRecommendations for '{test_movie}':\n", recommended[['title_x', 'cluster', 'similarity_score']])

    # -------------------------------------------------------------------------
    # 7. TELEMETRY VISUALIZATION
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(global_text_reduced[:, 0], global_text_reduced[:, 1], c=df_final_report['cluster'], cmap='viridis', alpha=0.5)
    plt.title("Production Movie Clusters Spatial Mapping")
    plt.show()

    params = {"max_features": max_features, "n_components": n_components, "runtime_median": runtime_median}
    logger.info(params)

    # Returning frozen components required for external live inference APIs
    return (
        df_final_report, tfidf, svd, mlb, scaler, cluster_router,
        scores, global_text_reduced, cos_sim, params
    )
