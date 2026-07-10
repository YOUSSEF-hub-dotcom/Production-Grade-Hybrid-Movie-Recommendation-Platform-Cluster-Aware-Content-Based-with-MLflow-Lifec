import pandas as pd
import ast
from collections import Counter, defaultdict
import logging

# Initialize logger for tracking exploratory data analysis milestones
logger = logging.getLogger(__name__)

def run_eda(df_full):
    logger.info("=================>> Exploratory Data Analysis (EDA)")

    # -------------------------------------------------------------------------
    # 0. GLOBAL PARSING PRE-REQUISITE (Optimization Phase)
    # -------------------------------------------------------------------------
    # Parsing literal structural strings into native Python objects once to eliminate 
    # redundant computing overhead across the downstream exploratory functions.
    logger.info("Pre-parsing literal structures (cast, crew, genres) for runtime optimization...")
    df_full['cast_list'] = df_full['cast'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df_full['crew_list'] = df_full['crew'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df_full['genres_list'] = df_full['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    ).apply(lambda x: [genre['name'] for genre in x])

    # -------------------------------------------------------------------------
    # 1. UNIQUE INDEXING & DUPLICATION AUDITING
    # -------------------------------------------------------------------------
    # Purpose: Verifying the cardinality of primary keys ('movie_id') vs visible names
    # to catch semantic indexing errors or duplicate row variants.
    logger.info("Assessing unique entities distribution (movie_id vs title_x):")
    print(df_full[['movie_id', 'title_x']].nunique())

    logger.info("Checking for absolute duplicate movie identity boundaries:")
    print(df_full['movie_id'].duplicated().any())

    logger.info("Isolating duplicate movie title strings alongside their unique IDs:")
    duplicated_titles = df_full[df_full['title_x'].duplicated(keep=False)].sort_values('title_x')
    print(duplicated_titles[['movie_id', 'title_x']])

    # -------------------------------------------------------------------------
    # 2. CO-STAR & CAST MEMBER DENSITY DISTRIBUTION
    # -------------------------------------------------------------------------
    # Purpose: Tracking core numerical densities for cast arrays to map out 
    # expected vector input dimensions for content modeling.
    logger.info("Calculating empirical baseline for cast member volume per movie...")
    df_full['num_cast'] = df_full['cast_list'].apply(len)
    average_cast = df_full['num_cast'].mean()
    logger.info(f"Average number of cast members per movie: {round(average_cast, 2)}")

    # -------------------------------------------------------------------------
    # 3. DIRECTORIAL NODES FREQUENCY
    # -------------------------------------------------------------------------
    # Purpose: Profiling creative directional distribution patterns to spot 
    # high-frequency directors who influence systemic similarity patterns.
    def get_director(crew):
        for member in crew:
            if member.get('job') == 'Director':
                return member['name']
        return None

    df_full['director'] = df_full['crew_list'].apply(get_director)
    director_counts = Counter(df_full['director'].dropna())
    directors_multiple_movies = {
        name: count for name, count in director_counts.items() if count > 1
    }
    
    logger.info("Listing active directors associated with multiple movie titles:")
    for director, count in directors_multiple_movies.items():
        print(f"{director}: {count} movies")

    # -------------------------------------------------------------------------
    # 4. STUDIO & PRODUCTION HOUSES AUDITING
    # -------------------------------------------------------------------------
    # Purpose: Evaluating feature sparse patterns across production entities 
    # to determine their weight relevance inside the recommender framework.
    def get_production(crew):
        for member in crew:
            if member.get('job') == 'Producer':
                return member['name']
        return None

    df_full['production'] = df_full['crew_list'].apply(get_production)
    production_counts = Counter(df_full['production'].dropna())
    productions_multiple_movies = {
        name: count for name, count in production_counts.items() if count > 1
    }

    logger.info("Listing active production houses with multi-title representations:")
    for production, count in productions_multiple_movies.items():
        print(f"{production}: {count} movies")

    # -------------------------------------------------------------------------
    # 5. ACTOR REPRESENTATION & SUB-COHORT GENRE CORRELATION
    # -------------------------------------------------------------------------
    # Purpose: Ranking historical cast representation to discover systemic biases 
    # toward specific popular actors, and analyzing cross-categorical correlations.
    def get_actor_names(cast):
        return [member['name'] for member in cast if 'name' in member]

    df_full['actor_names'] = df_full['cast_list'].apply(get_actor_names)
    all_actors = [actor for actors in df_full['actor_names'] for actor in actors]
    actor_counts = Counter(all_actors)
    
    if all_actors:
        top_actor, top_count = actor_counts.most_common(1)[0]
        print(f"The actor who appeared in the highest number of movies is {top_actor} with {top_count} movies.")

    # Building mapped dictionaries connecting explicit actor nodes to multi-labeled genre pools
    actor_genres = defaultdict(list)
    for _, row in df_full.iterrows():
        for actor in row['actor_names']:
            actor_genres[actor].extend(row['genres_list'])

    actor_genre_counts = {
        actor: Counter(genres) for actor, genres in actor_genres.items()
    }

    logger.info("Validating multi-label category cross-referencing (Case Study: Leonardo DiCaprio):")
    if 'Leonardo DiCaprio' in actor_genre_counts:
        print(actor_genre_counts['Leonardo DiCaprio'].most_common(5))

    # -------------------------------------------------------------------------
    # 6. HISTORICAL ERAS VS CAST DRIFT ANALYSIS
    # -------------------------------------------------------------------------
    # Purpose: Time-series clustering analysis to uncover data drift patterns 
    # and historical evolution across pre-2000 vs post-2000 cinematic structures.
    logger.info("Profiling chronological shifts in average cast configurations...")
    df_full['release_date'] = pd.to_datetime(df_full['release_date'], errors='coerce')
    df_full['era'] = df_full['release_date'].dt.year.apply(
        lambda x: 'Old' if pd.notnull(x) and x < 2000 else 'New'
    )
    avg_cast = df_full.groupby('era')['num_cast'].mean()
    logger.info("Average number of cast members grouped by cinema production era:")
    print(avg_cast)

    # -------------------------------------------------------------------------
    # 7. BENCHMARK COHORT SATURATION TEST
    # -------------------------------------------------------------------------
    # Purpose: Testing structural representation of major actor groups to ensure 
    # adequate data density for feature embedding layers.
    logger.info("Auditing saturation levels for highly recognizable talent anchors:")
    famous_actors = ["Tom Hanks", "Leonardo DiCaprio", "Brad Pitt", "Robert De Niro", "Johnny Depp"]
    famous_actor_counts = {}

    for actor in famous_actors:
        count = df_full[df_full['actor_names'].apply(lambda actors: actor in actors)].shape[0]
        famous_actor_counts[actor] = count
        print(f"{actor}: {count} movies")

    # -------------------------------------------------------------------------
    # 8. TOP-TIER MAX VALUE EXTRACTION
    # -------------------------------------------------------------------------
    # Purpose: Isolating dominant outliers in creative direction to ensure 
    # correct scaling during content-based transformation setups.
    logger.info("Locating the highest-volume director within the current registry matrix:")
    if director_counts:
        most_common_director = director_counts.most_common(1)[0]
        print(most_common_director)

    # -------------------------------------------------------------------------
    # 9. RECURRING COLLABORATION PATTERNS (NETWORK INTEGRITY)
    # -------------------------------------------------------------------------
    # Purpose: Uncovering persistent linkages between directors and actors to understand 
    # underlying thematic clusters that can be leveraged for recommendation affinity.
    logger.info("Detecting recurring professional pairs (Director-Actor collaborations):")
    director_actor_pairs = []
    for _, row in df_full.iterrows():
        director = row['director']
        if director and isinstance(row['actor_names'], list):
            for actor in row['actor_names']:
                director_actor_pairs.append((director, actor))

    pair_counts = Counter(director_actor_pairs)
    if director_actor_pairs:
        most_common_pair = pair_counts.most_common(1)[0]
        print(f"Top recurring partnership: {most_common_pair}")

    # -------------------------------------------------------------------------
    # 10. METADATA STRUCTURAL DENSITY BY JOB ROLE
    # -------------------------------------------------------------------------
    # Purpose: Documenting entire crew dataset density profiles to understand 
    # metadata sparsity levels across non-acting positions.
    logger.info("Mapping systemic metadata coverage across all corporate crew descriptions:")
    all_jobs = [member['job'] for crew in df_full['crew_list'] for member in crew if 'job' in member]
    job_counts = Counter(all_jobs)

    logger.info("Displaying top 10 distributed crew positions:")
    for job, count in job_counts.most_common(10):
        print(f"{job}: {count}")

    # -------------------------------------------------------------------------
    # 11. ANOMALY DETECTION & NULL VECTOR IDENTIFICATION
    # -------------------------------------------------------------------------
    # Purpose: Finding structural holes in crucial features (e.g., missing directors)
    # to implement safety measures before text feature aggregation.
    logger.info("Auditing data matrix for catastrophic feature voids (missing directors):")
    missing_director = df_full[df_full['director'].isnull()]
    if len(missing_director) > 0:
        logger.warning(f"Data Anomaly: Identified {len(missing_director)} movies lacking explicit directional assignments!")
    print(missing_director[['title_x', 'id']].head(10))

    # -------------------------------------------------------------------------
    # 12. SCREENPLAY SOURCE EXTRACTION
    # -------------------------------------------------------------------------
    # Purpose: Profiling script architectures, as text similarities depend 
    # heavily on the writers' narrative styles.
    logger.info("Isolating high-volume screenplay and script architects:")
    def get_writers(crew):
        return [member['name'] for member in crew if member.get('job') in ['Writer', 'Screenplay', 'Author']]

    df_full['writers'] = df_full['crew_list'].apply(get_writers)
    all_writers = [writer for writers in df_full['writers'] for writer in writers]
    writer_counts = Counter(all_writers)
    
    if all_writers:
        most_common_writer = writer_counts.most_common(1)[0]
        print(f"{most_common_writer[0]} wrote {most_common_writer[1]} scripts")

    # Returning frozen statistical counters required to power the Visualization Pipeline
    return (
        actor_counts,
        actor_genre_counts,
        avg_cast,
        famous_actor_counts,
        job_counts,
        writer_counts
    )
