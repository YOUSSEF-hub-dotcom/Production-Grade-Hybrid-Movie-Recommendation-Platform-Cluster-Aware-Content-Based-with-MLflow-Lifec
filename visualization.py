import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

import logging

logger = logging.getLogger("Visualization")

def run_visualization(
    df_full,
    actor_counts,
    actor_genre_counts,
    avg_cast,
    famous_actor_counts,
    job_counts,
    writer_counts
):

    logger.info("=================>> Visualization")

    plt.figure(figsize=(8, 5))
    sns.histplot(df_full['num_cast'], bins=30, kde=True, color="#36A2EB")
    plt.title("Distribution of Cast Members per Movie", fontsize=13)
    plt.xlabel("Number of Cast Members")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.show()

    top_directors = dict(Counter(df_full['director']).most_common(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(top_directors.values()),
        y=list(top_directors.keys()),
        palette="crest"
    )
    plt.title("Top 10 Directors by Number of Movies", fontsize=13)
    plt.xlabel("Number of Movies")
    plt.ylabel("Director")
    plt.tight_layout()
    plt.show()

    top_producers = dict(Counter(df_full['production']).most_common(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(top_producers.values()),
        y=list(top_producers.keys()),
        palette="magma"
    )
    plt.title("Top 10 Producers by Number of Movies", fontsize=13)
    plt.xlabel("Number of Movies")
    plt.ylabel("Producer")
    plt.tight_layout()
    plt.show()

    top_actors = dict(actor_counts.most_common(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(top_actors.values()),
        y=list(top_actors.keys()),
        palette="viridis"
    )
    plt.title("Top 10 Actors by Number of Movies", fontsize=13)
    plt.xlabel("Number of Movies")
    plt.ylabel("Actor")
    plt.tight_layout()
    plt.show()

    leo_genres = actor_genre_counts['Leonardo DiCaprio']
    plt.figure(figsize=(6, 6))
    plt.pie(
        leo_genres.values(),
        labels=leo_genres.keys(),
        autopct='%1.1f%%',
        colors=sns.color_palette("pastel")
    )
    plt.title("Genres associated with Leonardo DiCaprio", fontsize=13)
    plt.tight_layout()
    plt.show()

    avg_cast.plot(
        kind='bar',
        figsize=(6, 5),
        color=['#007BFF', '#28A745']
    )
    plt.title("Average Cast Members in Old vs New Movies", fontsize=13)
    plt.xlabel("Era")
    plt.ylabel("Average Number of Cast")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(famous_actor_counts.keys()),
        y=list(famous_actor_counts.values()),
        palette="coolwarm"
    )
    plt.title("Number of Movies with Famous Actors", fontsize=13)
    plt.xlabel("Actor")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.show()

    top_jobs = dict(job_counts.most_common(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(top_jobs.values()),
        y=list(top_jobs.keys()),
        palette="cubehelix"
    )
    plt.title("Top 10 Crew Jobs Distribution", fontsize=13)
    plt.xlabel("Count")
    plt.ylabel("Job")
    plt.tight_layout()
    plt.show()

    top_writers = dict(writer_counts.most_common(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(top_writers.values()),
        y=list(top_writers.keys()),
        palette="flare"
    )
    plt.title("Top 10 Writers by Number of Scripts", fontsize=13)
    plt.xlabel("Number of Scripts")
    plt.ylabel("Writer")
    plt.tight_layout()
    plt.show()

    all_text = " ".join(df_full['final_text'])
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(all_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title('Word Cloud of Movie Recommender System', fontsize=14)
    plt.tight_layout(pad=0)
    plt.show()
