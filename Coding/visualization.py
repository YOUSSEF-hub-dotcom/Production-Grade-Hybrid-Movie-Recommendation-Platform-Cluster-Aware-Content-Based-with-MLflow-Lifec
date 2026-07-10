import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import logging

# Initialize logger for tracking visualization pipeline telemetry
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

    # -------------------------------------------------------------------------
    # 1. DISTRIBUTION OF CAST MEMBERS PER MOVIE
    # -------------------------------------------------------------------------
    # Purpose: Visualizing the distribution profile of numerical features ('num_cast')
    # to understand skewness and feature density patterns across the dataset.
    plt.figure(figsize=(8, 5))
    sns.histplot(df_full['num_cast'], bins=30, kde=True, color="#36A2EB")
    plt.title("Distribution of Cast Members per Movie", fontsize=13)
    plt.xlabel("Number of Cast Members")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 2. TOP 10 DIRECTORS BY NUMBER OF MOVIES
    # -------------------------------------------------------------------------
    # Purpose: Extracting categorical frequencies to identify dominant 
    # directional nodes, which serve as heavy influencers in recommendation pooling.
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

    # -------------------------------------------------------------------------
    # 3. TOP 10 PRODUCERS BY NUMBER OF MOVIES
    # -------------------------------------------------------------------------
    # Purpose: Auditing high-frequency production networks and studios 
    # to measure categorical cardinality and identify outperforming industry entities.
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

    # -------------------------------------------------------------------------
    # 4. TOP 10 ACTORS BY NUMBER OF MOVIES
    # -------------------------------------------------------------------------
    # Purpose: Tracking star-power distribution metrics across the corpus 
    # to confirm the representation levels of high-frequency cast actors.
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

    # -------------------------------------------------------------------------
    # 5. GENRE ASSOCIATION PROFILE (CASE STUDY: LEONARDO DICAPRIO)
    # -------------------------------------------------------------------------
    # Purpose: Exploring multi-label categorical distributions within specific subsets
    # to evaluate the implicit correlation between explicit actors and genre targets.
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

    # -------------------------------------------------------------------------
    # 6. AVERAGE CAST MEMBERS IN OLD VS NEW MOVIES
    # -------------------------------------------------------------------------
    # Purpose: Bivariate analysis tracking numerical feature evolution across eras 
    # to inspect chronological variance and behavioral drift in movie structural design.
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

    # -------------------------------------------------------------------------
    # 7. NUMBER OF MOVIES WITH FAMOUS ACTORS
    # -------------------------------------------------------------------------
    # Purpose: Benchmarking specific benchmark cohorts to cross-examine 
    # high-profile label saturation and sparsity within the overall recommendation baseline.
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

    # -------------------------------------------------------------------------
    # 8. TOP 10 CREW JOBS DISTRIBUTION
    # -------------------------------------------------------------------------
    # Purpose: Analyzing crew dataset infrastructure metadata density 
    # to map out systemic workflows and discover non-empty metadata connections.
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

    # -------------------------------------------------------------------------
    # 9. TOP 10 WRITERS BY NUMBER OF SCRIPTS
    # -------------------------------------------------------------------------
    # Purpose: Highlighting top screenplay text architects, as content-based systems 
    # depend heavily on underlying creative text similarities.
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

    # -------------------------------------------------------------------------
    # 10. TEXTUAL SEMANTIC WORD CLOUD
    # -------------------------------------------------------------------------
    # Purpose: Generating a visual corpus vocabulary map to perform textual audits 
    # on dominant tokens, directly profiling what feeds the TF-IDF Vectorization pipeline.
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
