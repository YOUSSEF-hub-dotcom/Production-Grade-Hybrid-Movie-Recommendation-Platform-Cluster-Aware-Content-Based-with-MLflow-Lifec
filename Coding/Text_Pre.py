import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger("Text Preprocessing")

def run_text_preprocessing(df_full):

    logger.info("=================>> Text Preprocessing with People Info")

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    df_full['people'] = (
        df_full['actor_names'].astype(str) + " " +
        df_full['director'].astype(str) + " " +
        df_full['production'].astype(str)
    )

    df_full['combined_text'] = (
        df_full['tagline'].astype(str) + " " +
        df_full['keywords'].astype(str) + " " +
        df_full['overview'].astype(str) + " " +
        df_full['people']
    )

    df_full['lower_col'] = df_full['combined_text'].str.lower()

    df_full['tokenized_message'] = df_full['lower_col'].apply(word_tokenize)

    df_full['clean_tokens'] = df_full['tokenized_message'].apply(
        lambda tokens: [
            re.sub(r'[^a-zA-Z]', '', word)
            for word in tokens
            if word.isalpha()
        ]
    )

    stop_words = set(stopwords.words('english'))

    df_full['no_stopwords'] = df_full['clean_tokens'].apply(
        lambda tokens: [
            word for word in tokens
            if word not in stop_words
        ]
    )

    stemmer = PorterStemmer()

    df_full['stemmed_tokens'] = df_full['no_stopwords'].apply(
        lambda tokens: [
            stemmer.stem(word)
            for word in tokens
        ]
    )

    df_full['final_text'] = df_full['stemmed_tokens'].apply(
        lambda tokens: ' '.join(tokens)
    )

    sns.set_theme(style="whitegrid", palette="muted")

    return df_full
