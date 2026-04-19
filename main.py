from data_pipeline import (
    load_and_merge_data,
    basic_data_overview,
    clean_data
)
import argparse
from EDA import run_eda
from Text_Pre import run_text_preprocessing
from visualization import run_visualization
from model import build_and_train_model
from MLflow_LifeCycle import run_mlflow
import logging
from logger_config import setup_logging

setup_logging()
logger = logging.getLogger("Main")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--n_components", type=int, default=200)
    args = parser.parse_args()

    logger.info(" Movie Recommender System")

    movies_path = r"C:\Users\Hedaya_city\Downloads\tmdb_5000_movies.csv"
    credits_path = r"C:\Users\Hedaya_city\Downloads\tmdb_5000_credits.csv"


    logger.info("\n Loading Data...")
    df_full = load_and_merge_data(
        credits_path=credits_path,
        movies_path=movies_path
    )
    basic_data_overview(df_full)


    logger.info("\n🧹 Cleaning Data...")
    df_full = clean_data(df_full)


    logger.info("\n Running EDA...")
    (
        actor_counts,
        actor_genre_counts,
        avg_cast,
        famous_actor_counts,
        job_counts,
        writer_counts
    ) = run_eda(df_full)


    logger.info("\n Text Preprocessing...")
    df_full = run_text_preprocessing(df_full)


    logger.info("\n Visualization...")
    run_visualization(
        df_full=df_full,
        actor_counts=actor_counts,
        actor_genre_counts=actor_genre_counts,
        avg_cast=avg_cast,
        famous_actor_counts=famous_actor_counts,
        job_counts=job_counts,
        writer_counts=writer_counts
    )


    logger.info("\nTraining Model (Clustering + SVD)...")
    (
        df_full,
        tfidf,
        svd,
        mlb,
        scaler,
        cluster_labels,
        silhouette_scores,
        text_reduced,
        cos_sim, train_params
    ) = build_and_train_model(df_full, args.max_features, args.n_components)


    required_cols = ["final_text", "genres_list", "title_x"]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f" Missing required columns: {missing}")


    logger.info("\nLogging Model to MLflow (PyFunc + Registry)...")

    run_id = run_mlflow(
        df_full, tfidf, svd, mlb, scaler, cluster_labels,
        silhouette_scores, train_params
    )

    logger.info(f"\n MLflow Run ID: {run_id}")
    logger.info("\n Pipeline Finished Successfully")


if __name__ == "__main__":
    main()
