import pandas as pd
import logging

logger = logging.getLogger("Data Pipeline")

def load_and_merge_data(credits_path, movies_path):
    logger.info("Loading data from Excel...")
    df = pd.read_csv(credits_path)
    df_1 = pd.read_csv(movies_path)

    df_full = pd.merge(
        df,
        df_1,
        left_on='movie_id',
        right_on='id',
        how='outer'
    )

    return df_full


def basic_data_overview(df_full):
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

    logger.info("Random Rows in Dat:")
    print(df_full.sample(5))


def clean_data(df_full):
    logger.info("=================>> Cleaning Data")

    logger.info("Number of Frequent Rows")
    print(df_full.duplicated().sum())

    missing_pct = df_full['homepage'].isnull().mean() * 100
    if missing_pct > 60:
        logger.warning(f"'homepage' has {missing_pct:.2f}% missing values. Proceeding to drop it.")

    logger.info("Missing Values in Data:")
    print(df_full.isnull().sum())

    logger.info(" Dropping 'homepage' column because it has more than 60% missing values...")
    df_full = df_full.drop(['homepage'], axis=1, errors='ignore')

    logger.info("Filling missing values in 'overview' with empty string...")
    df_full['overview'] = df_full['overview'].fillna(
        "UnKnown"
    )

    logger.info(" Filling missing values in 'tagline' with empty string...")
    df_full['tagline'] = df_full['tagline'].fillna('')

    logger.info(" Filling missing values in 'runtime' with median value...")
    df_full['runtime'] = df_full['runtime'].fillna(
        df_full['runtime'].median()
    )

    logger.info(" Filling missing values in 'release_date' with 'Unknown'...")
    df_full['release_date'] = df_full['release_date'].fillna('Unknown')

    logger.info("Checking remaining missing values in the dataset...")
    print(df_full.isnull().sum())

    #sns.heatmap(df_full.isnull(), annot=True)
    #plt.title("Remaining Missing Values")
    #plt.show()

    df_full.drop_duplicates(keep='first')

    return df_full



