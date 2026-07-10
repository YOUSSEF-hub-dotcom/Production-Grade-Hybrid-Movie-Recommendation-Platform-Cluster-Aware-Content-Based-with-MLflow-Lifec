import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Initialize logger for MLOps and tracking telemetry
logger = logging.getLogger("MLflow")

class MovieRecommenderModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PythonModel wrapper to package data indexes, similarity metrics,
    and clustering structures into a deployable serving endpoint.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def load_context(self, context):
        """
        Executes once at model serving initialization. Warm-boots high-memory artifacts
        into local container memory to optimize inference latency.
        """
        self.df_full = pd.read_parquet(context.artifacts["data_parquet"]).reset_index(drop=True)
        self.similarity_matrix = joblib.load(context.artifacts["similarity_matrix"])
        self.cluster_labels = joblib.load(context.artifacts["cluster_labels"])

    def predict(self, context, model_input):
        """
        Executes recommendation queries inside the target cluster space using calculated 
        cosine similarities to prevent exhaustive full-dataset scans.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_names)

        title = model_input.iloc[0, 0]

        # Handle out-of-vocabulary anomalies gracefully
        if title not in self.df_full["title_x"].values:
            logger.warning(f"Movie search failed: '{title}' not found in database.")
            return pd.DataFrame({"error": [f"الفيلم '{title}' مش موجود"]})

        # Locate target entry index and extract its assigned neighborhood cluster
        idx = self.df_full[self.df_full["title_x"] == title].index[0]
        cluster = self.cluster_labels[idx]

        # Isolate candidate nodes within the same cluster to save computation power
        cluster_indices = np.where(self.cluster_labels == cluster)[0]
        sim_scores = self.similarity_matrix[idx][cluster_indices]

        # Sort similarity metrics in descending order and slice the top 5 relative recommendations
        top_indices = np.argsort(sim_scores)[::-1][1:6]
        top_cluster_indices = cluster_indices[top_indices]

        # Construct final payload object containing target similarity coefficients
        result = self.df_full.iloc[top_cluster_indices].copy()
        result["similarity_score"] = sim_scores[top_indices].astype(float)
        
        logger.info(
            f"Recommendations generated for '{title}' (Cluster: {cluster}). "
            f"Average Similarity: {result['similarity_score'].mean():.2f}"
        )
        return result.reset_index(drop=True)


def run_mlflow(df_full, tfidf, svd, mlb, scaler, cluster_labels, silhouette_scores, params):
    """
    Core pipeline to extract hybrid text-categorical vectors, calculate similarity matrices,
    log metadata runs, and enforce CI/CD deployment rules based on silhouette scores.
    """
    EXPERIMENT_NAME = "Movie_Clustering_Project"
    REGISTERED_MODEL_NAME = "MovieRecommenderSystem"

    # Enforce or create the central experiment space
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run() as run:
        logger.info("Calculating similarity matrix and saving artifacts...")
        
        # -------------------------------------------------------------------------
        # 1. LATENT SPACE EXTRACTION & CORATING COMBINATION
        # -------------------------------------------------------------------------
        # Transforming raw strings into structured matrices and reducing text dimensionality 
        # using SVD prior to stacking with categorical multi-label columns.
        text_features = tfidf.transform(df_full["final_text"])
        text_reduced = svd.transform(text_features)
        genres_encoded = mlb.transform(df_full["genres_list"])

        combined_features = np.hstack([text_reduced, genres_encoded])
        logger.info(f"Feature Matrix combined. Dimensions: {combined_features.shape}")

        # Compute pairwise distance metrics via cosine matrix
        sim_matrix = cosine_similarity(combined_features)

        # -------------------------------------------------------------------------
        # 2. FILE ARTIFACT SERIALIZATION
        # -------------------------------------------------------------------------
        # Writing structural arrays to the server storage to map them as MLflow Run Assets.
        data_path = os.path.abspath("df_full.parquet")
        sim_path = os.path.abspath("similarity_matrix.pkl")
        cluster_path = os.path.abspath("cluster_labels.pkl")

        df_full.to_parquet(data_path)
        joblib.dump(sim_matrix, sim_path)
        joblib.dump(cluster_labels, cluster_path)

        artifacts = {
            "data_parquet": data_path,
            "similarity_matrix": sim_path,
            "cluster_labels": cluster_path
        }

        # -------------------------------------------------------------------------
        # 3. METRICS LOGGING & METADATA MANAGEMENT
        # -------------------------------------------------------------------------
        # Pushing structural hyper-parameters and quantitative performance scores to tracking servers.
        mlflow.log_params(params)
        mlflow.log_param("n_clusters", len(set(cluster_labels)))

        for k, v in silhouette_scores.items():
            mlflow.log_metric(f"silhouette_{k}", float(v))

        # Define data validation contracts for downstream microservices
        input_example = pd.DataFrame({"title_x": ["Toy Story"]})
        output_example = pd.DataFrame({
            "title_x": ["Film A", "Film B"],
            "similarity_score": [0.95, 0.88]
        })
        signature = infer_signature(input_example, output_example)

        # -------------------------------------------------------------------------
        # 4. CUSTOM MODEL REGISTRATION & MODEL REGISTRY INGESTION
        # -------------------------------------------------------------------------
        # Uploading custom pyfunc instance accompanied by environment specifications.
        recommender = MovieRecommenderModel(feature_names=["title_x"])

        mlflow.pyfunc.log_model(
            artifact_path="movie_recommender_model",
            python_model=recommender,
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "mlflow",
                "scikit-learn",
                "pandas",
                "numpy",
                "pyarrow",
                "joblib"
            ]
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/movie_recommender_model"

        # Inject model into the shared production repository catalog
        reg_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        version = reg_model.version

        # Initialize the initial rollout deployment stage
        client.transition_model_version_stage(REGISTERED_MODEL_NAME, version, "Staging")

        # -------------------------------------------------------------------------
        # 5. AUTOMATED CD STAGE PROMOTION GATE
        # -------------------------------------------------------------------------
        # Automated baseline validation checks: if structural silhouette checks pass the 
        # validation gate threshold, the candidate model is promoted to Live Production instantly.
        MIN_THRESHOLD = 0.05
        current_score = silhouette_scores.get('k_best', 0)

        if current_score >= MIN_THRESHOLD:
            client.transition_model_version_stage(
                REGISTERED_MODEL_NAME, version, "Production", archive_existing_versions=True
            )
            status = "Production"
            logger.info(
                f"Promotion Success: Silhouette Score ({current_score:.4f}) is above threshold ({MIN_THRESHOLD})."
            )
        else:
            status = "Staging"
            logger.warning(
                f"Promotion Denied: Silhouette Score ({current_score:.4f}) is too low. "
                f"Model kept in Staging for tuning."
            )

        # Generate telemetry performance summaries
        report = (
            f"\n{'═' * 45}\n"
            f" Run Completed Successfully!\n"
            f" Model: {REGISTERED_MODEL_NAME} | Version: {version}\n"
            f" Final Stage: {status}\n"
            f" Run ID: {run_id}\n"
            f"{'═' * 45}"
        )
        logger.info(report)

        return run_id
