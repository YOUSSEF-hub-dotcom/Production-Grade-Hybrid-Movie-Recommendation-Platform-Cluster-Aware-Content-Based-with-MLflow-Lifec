import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import ast
from collections import Counter, defaultdict
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ----------------- NLTK -----------------
nltk.download('stopwords')
nltk.download('punkt')


# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Hedaya_city\Downloads\tmdb_5000_credits.csv")
    df_1 = pd.read_csv(r"C:\Users\Hedaya_city\Downloads\tmdb_5000_movies.csv")
    df_full = pd.merge(df, df_1, left_on='movie_id', right_on='id', how='outer')
    return df_full


df_full = load_data()


# ----------------- Preprocessing -----------------
@st.cache_data
def preprocess_data(df):
    df = df.drop(['homepage'], axis=1, errors='ignore')
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    df['release_date'] = df['release_date'].fillna('Unknown')

    df['cast_list'] = df['cast'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['crew_list'] = df['crew'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['num_cast'] = df['cast_list'].apply(len)

    # Director
    df['director'] = df['crew_list'].apply(lambda c: next((m['name'] for m in c if m.get('job') == 'Director'), None))
    # Producer
    df['production'] = df['crew_list'].apply(lambda c: next((m['name'] for m in c if m.get('job') == 'Producer'), None))
    # Actor names
    df['actor_names'] = df['cast_list'].apply(lambda cast: [m['name'] for m in cast if 'name' in m])
    # Genres
    df['genres_list'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notnull(x) else [])

    # Era
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['era'] = df['release_date'].dt.year.apply(lambda x: 'Old' if x < 2000 else 'New')

    # Aggregates
    avg_cast = df.groupby('era')['num_cast'].mean()
    top_directors = dict(Counter(df['director'].dropna()).most_common(10))
    top_producers = dict(Counter(df['production'].dropna()).most_common(10))
    top_actors = dict(Counter([a for actors in df['actor_names'] for a in actors]).most_common(10))

    famous_actors = ["Tom Hanks", "Leonardo DiCaprio", "Brad Pitt", "Robert De Niro", "Johnny Depp"]
    famous_actor_counts = {actor: df[df['actor_names'].apply(lambda actors: actor in actors)].shape[0] for actor in
                           famous_actors}

    actor_genres = defaultdict(list)
    for _, row in df.iterrows():
        for actor in row['actor_names']:
            actor_genres[actor].extend(row['genres_list'])
    leo_genres = dict(Counter(actor_genres['Leonardo DiCaprio']))

    all_jobs = [m['job'] for crew in df['crew_list'] for m in crew if 'job' in m]
    top_jobs = dict(Counter(all_jobs).most_common(10))

    df['writers'] = df['crew_list'].apply(
        lambda c: [m['name'] for m in c if m.get('job') in ['Writer', 'Screenplay', 'Author']])
    all_writers = [w for writers in df['writers'] for w in writers]
    top_writers = dict(Counter(all_writers).most_common(10))

    # WordCloud
    df['people'] = df['actor_names'].astype(str) + " " + df['director'].astype(str) + " " + df['production'].astype(str)
    df['combined_text'] = df['tagline'].astype(str) + " " + df['keywords'].astype(str) + " " + df['overview'].astype(
        str) + " " + df['people']
    df['lower_col'] = df['combined_text'].str.lower()
    df['tokenized_message'] = df['lower_col'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    df['stemmed_tokens'] = df['tokenized_message'].apply(
        lambda tokens: [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words])
    all_text = " ".join(df['stemmed_tokens'].apply(lambda x: " ".join(x)))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)

    return (df, avg_cast, top_directors, top_producers, top_actors, famous_actor_counts,
            leo_genres, top_jobs, top_writers, wordcloud)


(df_full, avg_cast, top_directors, top_producers, top_actors, famous_actor_counts,
 leo_genres, top_jobs, top_writers, wordcloud) = preprocess_data(df_full)

# ----------------- Streamlit Layout -----------------
st.set_page_config(page_title="🎥 Movie Dashboard", layout="wide", page_icon="🎬")
st.title("🎬 Movie Recommendation Dashboard")
st.markdown("---")

tab_option = st.sidebar.selectbox("Select a Tab", [
    "📊 Cast & Crew Overview",
    "🎬 Producers & Actors",
    "🌟 Famous Actors & Genres",
    "📑 Movie Era & Crew Jobs",
    "✍️ Writers & Word Cloud"
])

sns.set_theme(style="whitegrid", palette="muted")

# ----------------- Tabs -----------------
if tab_option == "📊 Cast & Crew Overview":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_full['num_cast'], bins=30, kde=True, color="#36A2EB", ax=ax)
        ax.set_title("Distribution of Cast Members per Movie")
        ax.set_xlabel("Number of Cast Members")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(top_directors.values()), y=list(top_directors.keys()), palette="crest", ax=ax)
        ax.set_title("Top 10 Directors by Number of Movies")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Director")
        st.pyplot(fig)

elif tab_option == "🎬 Producers & Actors":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(top_producers.values()), y=list(top_producers.keys()), palette="magma", ax=ax)
        ax.set_title("Top 10 Producers by Number of Movies")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Producer")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(top_actors.values()), y=list(top_actors.keys()), palette="viridis", ax=ax)
        ax.set_title("Top 10 Actors by Number of Movies")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Actor")
        st.pyplot(fig)

elif tab_option == "🌟 Famous Actors & Genres":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(leo_genres.values(), labels=leo_genres.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        ax.set_title("Genres associated with Leonardo DiCaprio")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(famous_actor_counts.values()), y=list(famous_actor_counts.keys()), palette="coolwarm", ax=ax)
        ax.set_title("Number of Movies with Famous Actors")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Actor")
        st.pyplot(fig)

elif tab_option == "📑 Movie Era & Crew Jobs":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        avg_cast.plot(kind='bar', color=['#007BFF', '#28A745'], ax=ax)
        ax.set_title("Average Cast Members in Old vs New Movies")
        ax.set_xlabel("Era")
        ax.set_ylabel("Average Number of Cast")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(top_jobs.values()), y=list(top_jobs.keys()), palette="cubehelix", ax=ax)
        ax.set_title("Top 10 Crew Jobs Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Job")
        st.pyplot(fig)

elif tab_option == "✍️ Writers & Word Cloud":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(top_writers.values()), y=list(top_writers.keys()), palette="flare", ax=ax)
        ax.set_title("Top 10 Writers by Number of Scripts")
        ax.set_xlabel("Number of Scripts")
        ax.set_ylabel("Writer")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud of Movie Recommender System")
        st.pyplot(fig)
