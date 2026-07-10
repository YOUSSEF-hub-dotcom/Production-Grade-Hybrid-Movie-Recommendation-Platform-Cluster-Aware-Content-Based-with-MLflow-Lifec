import pandas as pd
import logging

# Initialize logger for tracking data pipeline telemetry
logger = logging.getLogger("Data Pipeline")

def load_and_merge_data(credits_path, movies_path):
    """
    Loads raw metadata from paths and performs an outer join 
    to reconstruct the comprehensive movies dataset.
    """
    logger.info("Loading data from CSV...")
    df = pd.read_csv(credits_path)
    df_1 = pd.read_csv(movies_path)

    # Merging datasets based on relational keys
    df_full = pd.merge(
        df,
        df_1,
        left_on='movie_id',
        right_on='id',
        how='outer'
    )
    return df_full


def basic_data_overview(df_full):
    """
    Executes a comprehensive structural audit on the dataframe 
    to monitor data types, shapes, and summary statistics.
    """
    pd.set_option('display.width', None)
    print(df_full.head(30))
    logger.info("Dataset Loading Successful...")

    logger.info("=================>> Basic Function")
    logger.info("Information about Data:")
    print(df_full.info())

    logger.info("Number of rows and Columns:")
    print(df_full.shape)

    logger.info("Name of Columns:")
    print(df_full.columns)

    logger.info("Statistical Operations:")
    print(df_full.describe(include='object').round(2))

    logger.info("Data Types in Data:")
    print(df_full.dtypes)

    logger.info("Display the index Range:")
    print(df_full.index)

    logger.info("Random Rows in Data:")
    print(df_full.sample(5))


def clean_data(df_full):
    """
    Performs structural cleaning and constant value imputations.
    Statistical imputations (like Median) are omitted here to prevent Data Leakage.
    """
    logger.info("=================>> Cleaning Data")

    # CRITICAL FIX: Explicitly re-assigning to drop duplicates and update memory allocations
    logger.info("Number of Frequent Rows Before Drop:")
    print(df_full.duplicated().sum())
    df_full = df_full.drop_duplicates(keep='first').copy()

    # Rule-based column dropping for high-sparsity features
    missing_pct = df_full['homepage'].isnull().mean() * 100
    if missing_pct > 60:
        logger.warning(f"'homepage' has {missing_pct:.2f}% missing values. Proceeding to drop it.")

    logger.info("Missing Values in Data Before Cleaning:")
    print(df_full.isnull().sum())

    logger.info("Dropping 'homepage' column because it has more than 60% missing values...")
    df_full = df_full.drop(['homepage'], axis=1, errors='ignore')

    # PRODUCTION-SAFE CONSTANT IMPUTATION:
    # Filling text features with hardcoded values does NOT warp data distributions 
    # and is completely safe from Data Leakage prior to Train-Test Splitting.
    logger.info("Filling missing values in 'overview' with 'UnKnown'...")
    df_full['overview'] = df_full['overview'].fillna("UnKnown")

    logger.info("Filling missing values in 'tagline' with empty string...")
    df_full['tagline'] = df_full['tagline'].fillna('')

    logger.info("Filling missing values in 'release_date' with 'Unknown'...")
    df_full['release_date'] = df_full['release_date'].fillna('Unknown')

    # ANTI-DATA LEAKAGE GUARDRAIL:
    # 'runtime' is a numerical feature. Calculating the Median across the global column here 
    # would infect the Training Set with Test Set variance (Data Bleeding). 
    # Therefore, we intentionally leave its null values to be resolved post-Split inside the model pipeline.
    logger.info("Leaving 'runtime' missing values to be safely handled in Model Pipeline after Split.")

    logger.info("Checking remaining missing values in the dataset...")
    print(df_full.isnull().sum())

    return df_full
