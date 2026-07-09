import pandas as pd
import ast
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)
def run_eda(df_full):

    logger.info("=================>> Exploratory Data Analysis (EDA)")

    logger.info("How many unique movies exist in the dataset (based on movie_id or title)")
    print(df_full[['movie_id', 'title_x']].nunique())

    logger.info("Are there any duplicate movies titles?")
    print(df_full['movie_id'].duplicated().any())

    logger.info("Duplicate Movie Titles and thier IDs:")
    duplicated_titles = df_full[df_full['title_x'].duplicated(keep=False)].sort_values('title_x')
    print(duplicated_titles[['movie_id', 'title_x']])


    logger.info("What is the average number of cast members per movie?")
    df_full['cast_list'] = df_full['cast'].apply(lambda x: ast.literal_eval(x))
    df_full['num_cast'] = df_full['cast_list'].apply(len)
    average_cast = df_full['num_cast'].mean()
    logger.info(f"Average number of cast members per movie: {round(average_cast, 2)}")


    logger.info("Are there directors who directed more than one movie?")
    df_full['crew_list'] = df_full['crew'].apply(lambda x: ast.literal_eval(x))

    def get_director(crew):
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return None

    df_full['director'] = df_full['crew_list'].apply(get_director)
    director_counts = Counter(df_full['director'])
    directors_multiple_movies = {
        name: count for name, count in director_counts.items() if count > 1
    }
    logger.info("Directors with more than one movie:")
    for director, count in directors_multiple_movies.items():
        print(f"{director}: {count} movies")


    logger.info("Are there productions who producted more than one movie?")
    df_full['crew_list'] = df_full['crew'].apply(lambda x: ast.literal_eval(x))

    def get_production(crew):
        for member in crew:
            if member['job'] == 'Producer':
                return member['name']
        return None

    df_full['production'] = df_full['crew_list'].apply(get_production)
    production_counts = Counter(df_full['production'])
    productions_multiple_movies = {
        name: count for name, count in production_counts.items() if count > 1
    }

    logger.info("Productions with more than one movie:")
    for production, count in productions_multiple_movies.items():
        print(f"{production}: {count} movies")


    logger.info('which actor appeared in the highest number of movies')
    df_full['cast_list'] = df_full['cast'].apply(lambda x: ast.literal_eval(x))

    def get_actor_names(cast):
        return [member['name'] for member in cast if 'name' in member]

    df_full['actor_names'] = df_full['cast_list'].apply(get_actor_names)
    all_actors = [actor for actors in df_full['actor_names'] for actor in actors]
    actor_counts = Counter(all_actors)
    top_actor, top_count = actor_counts.most_common(1)[0]
    print(
        f"The actor who appeared in the highest number of movies is "
        f"{top_actor} with {top_count} movies."
    )

    df_full['genres_list'] = df_full['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    df_full['genres_list'] = df_full['genres_list'].apply(
        lambda x: [genre['name'] for genre in x]
    )

    actor_genres = defaultdict(list)

    for _, row in df_full.iterrows():
        for actor in row['actor_names']:
            actor_genres[actor].extend(row['genres_list'])

    actor_genre_counts = {
        actor: Counter(genres) for actor, genres in actor_genres.items()
    }

    logger.info("Genres associated with Leonardo DiCaprio:")
    print(actor_genre_counts['Leonardo DiCaprio'].most_common(5))


    logger.info("How many cast members were there in older movies Vs newer movies?")
    df_full['release_date'] = pd.to_datetime(df_full['release_date'], errors='coerce')
    df_full['num_cast'] = df_full['cast'].apply(
        lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0
    )
    df_full['era'] = df_full['release_date'].dt.year.apply(
        lambda x: 'Old' if x < 2000 else 'New'
    )
    avg_cast = df_full.groupby('era')['num_cast'].mean()

    logger.info("Average number of cast members:")
    print(avg_cast)


    logger.info(
        'Do movies with famous actors '
        '(Tom Hanks, Leonardo DiCaprio, etc.) appear in small or large numbers?'
    )
    famous_actors = [
        "Tom Hanks",
        "Leonardo DiCaprio",
        "Brad Pitt",
        "Robert De Niro",
        "Johnny Depp"
    ]

    famous_actor_counts = {}

    for actor in famous_actors:
        count = df_full[
            df_full['actor_names'].apply(lambda actors: actor in actors)
        ].shape[0]
        famous_actor_counts[actor] = count

    for actor, count in famous_actor_counts.items():
        print(f"{actor}: {count} movies")


    logger.info('Which director directed the most movies in the dataset?')

    def get_director(crew):
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return None

    df_full['director'] = df_full['crew_list'].apply(get_director)
    director_counts = Counter(df_full['director'].dropna())
    most_common_director = director_counts.most_common(1)[0]
    print(most_common_director)


    logger.info('Are there directors who always work with the same actors?')

    df_full['actor_names'] = df_full['cast_list'].apply(get_actor_names)

    director_actor_pairs = []

    for _, row in df_full.iterrows():
        director = row['director']
        if director and isinstance(row['actor_names'], list):
            for actor in row['actor_names']:
                director_actor_pairs.append((director, actor))

    pair_counts = Counter(director_actor_pairs)
    most_common_pair = pair_counts.most_common(1)[0]
    print(most_common_pair)


    logger.info("which actor worked the most with a specific director?")

    director_actor_pairs = []

    for _, row in df_full.iterrows():
        director = row['director']
        if director and isinstance(row['cast_list'], list):
            for actor in row['actor_names']:
                director_actor_pairs.append((director, actor))

    pair_counts = Counter(director_actor_pairs)
    most_common_pair = pair_counts.most_common(1)[0]
    print(most_common_pair)


    logger.info("What is the distribution of crew jobs (how many directors, writers, producers, etc.)?")
    df_full['crew_list'] = df_full['crew'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    all_jobs = []

    for crew in df_full['crew_list']:
        for member in crew:
            if 'job' in member:
                all_jobs.append(member['job'])

    job_counts = Counter(all_jobs)

    logger.info("Distribution of crew jobs:")
    for job, count in job_counts.most_common(10):
        print(f"{job}: {count}")


    logger.info("Are there any movies without a director (missing crew data)?")

    def get_director(crew):
        for member in crew:
            if member.get('job') == 'Director':
                return member['name']
        return None

    df_full['director'] = df_full['crew_list'].apply(get_director)
    missing_director = df_full[df_full['director'].isnull()]

    if len(missing_director) > 0:
        logger.warning(f"Found {len(missing_director)} movies without a director!")

    print(missing_director[['title_x', 'id']].head(10))

    logger.info("Which writer wrote the highest number of scripts?")

    def get_writers(crew):
        writers = []
        for member in crew:
            if member.get('job') in ['Writer', 'Screenplay', 'Author']:
                writers.append(member['name'])
        return writers

    df_full['writers'] = df_full['crew_list'].apply(get_writers)
    all_writers = [writer for writers in df_full['writers'] for writer in writers]
    writer_counts = Counter(all_writers)
    most_common_writer = writer_counts.most_common(1)[0]

    print(f"{most_common_writer[0]} wrote {most_common_writer[1]} scripts")

    return (
        actor_counts,
        actor_genre_counts,
        avg_cast,
        famous_actor_counts,
        job_counts,
        writer_counts
    )
