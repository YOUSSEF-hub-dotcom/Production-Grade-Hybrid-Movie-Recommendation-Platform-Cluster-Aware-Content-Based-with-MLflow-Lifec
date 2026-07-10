import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Initialize logger for text engineering pipeline telemetry
logger = logging.getLogger("Text Preprocessing")

def run_text_preprocessing(df_full):
    logger.info("=================>> Text Preprocessing with People Info")

    # -------------------------------------------------------------------------
    # 0. NLTK RESOURCE DEPENDENCIES
    # -------------------------------------------------------------------------
    # Ensuring required lexical resources are warm and available in the local environment.
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # -------------------------------------------------------------------------
    # 1. ENTITY SQUASHING & METADATA COMBINATION
    # -------------------------------------------------------------------------
    # Critical Fix: Collapsing multi-word entity names (e.g., 'Tom Hanks' -> 'TomHanks')
    # into unified tokens to prevent downstream tokenizers from splitting first/last names, 
    # which injects severe recommendations cross-contamination.
    logger.info("Collapsing multi-word names and merging textual feature spaces...")
    
    def collapse_names(names):
        if isinstance(names, list):
            return " ".join([name.replace(" ", "") for name in names])
        elif isinstance(names, str) and names != 'None':
            return names.replace(" ", "")
        return ""

    df_full['clean_actors'] = df_full['actor_names'].apply(collapse_names)
    df_full['clean_director'] = df_full['director'].apply(collapse_names)
    df_full['clean_production'] = df_full['production'].apply(collapse_names)

    # Aggregating collapsed entity metadata space
    df_full['people'] = (
        df_full['clean_actors'] + " " +
        df_full['clean_director'] + " " +
        df_full['clean_production']
    )

    # Constructing the comprehensive unstructured text baseline
    df_full['combined_text'] = (
        df_full['tagline'].astype(str).fillna('') + " " +
        df_full['keywords'].astype(str).fillna('') + " " +
        df_full['overview'].astype(str).fillna('') + " " +
        df_full['people']
    )

    # -------------------------------------------------------------------------
    # 2. NORMALIZATION & LEVELLING
    # -------------------------------------------------------------------------
    # Purpose: Eliminating case-sensitivity variance by forcing standard lower-case profiles.
    df_full['lower_col'] = df_full['combined_text'].str.lower()

    # -------------------------------------------------------------------------
    # 3. LEXICAL TOKENIZATION
    # -------------------------------------------------------------------------
    # Purpose: Converting flat text streams into native structural lists of words.
    df_full['tokenized_message'] = df_full['lower_col'].apply(word_tokenize)

    # -------------------------------------------------------------------------
    # 4. REGULAR EXPRESSION NOISE CLEANING
    # -------------------------------------------------------------------------
    # Purpose: Stripping punctuation, numeric artifacts, and specialized chars 
    # to enforce strict non-sparse alphanumeric arrays.
    df_full['clean_tokens'] = df_full['tokenized_message'].apply(
        lambda tokens: [
            re.sub(r'[^a-zA-Z]', '', word)
            for word in tokens
            if word.isalpha()
        ]
    )

    # -------------------------------------------------------------------------
    # 5. STOPWORDS INFRASTRUCTURE FILTERING
    # -------------------------------------------------------------------------
    # Purpose: Dropping low-entropy connector words ('the', 'is', 'at') using O(1) 
    # hash-set lookups to optimize performance and vector density.
    stop_words = set(stopwords.words('english'))
    df_full['no_stopwords'] = df_full['clean_tokens'].apply(
        lambda tokens: [
            word for word in tokens
            if word not in stop_words
        ]
    )

    # -------------------------------------------------------------------------
    # 6. PORTER STEMMING MECHANICS
    # -------------------------------------------------------------------------
    # Purpose: Reducing morphological variations down to their base linguistic stem 
    # (e.g., 'running', 'runs' -> 'run') to maximize cosine similarity matches.
    stemmer = PorterStemmer()
    df_full['stemmed_tokens'] = df_full['no_stopwords'].apply(
        lambda tokens: [
            stemmer.stem(word)
            for word in tokens
        ]
    )

    # -------------------------------------------------------------------------
    # 7. FINAL DETOKENIZATION & VISUALIZATION ENVIRONMENT SETTINGS
    # -------------------------------------------------------------------------
    # Re-packing pristine tokens back into space-separated strings ready for Vectorizer engines.
    df_full['final_text'] = df_full['stemmed_tokens'].apply(
        lambda tokens: ' '.join(tokens)
    )

    # Enforcing global rendering aesthetics for subsequent visual diagnostics modules
    sns.set_theme(style="whitegrid", palette="muted")
    
    logger.info("Text Preprocessing Pipeline executed successfully.")
    return df_full
