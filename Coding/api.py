import pandas as pd
import mlflow.pyfunc
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import logging

# Initialize standard application logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Production Gateways & Model Tracking Configurations
TMDB_API_KEY = "698fe165093c8bb05c621472b0c91aac"
MODEL_URI = "models:/MovieRecommenderSystem/Production"

# -------------------------------------------------------------------------
# 1. MODEL REGISTRY INGESTION & COLD-START LOADING
# -------------------------------------------------------------------------
# Pulling the latest production-validated artifact bundle from MLflow Registry.
# Extracting the underlying embedded DataFrame directly from the Custom PyFunc 
# instance to eliminate redundant disk I/O reads.
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    # Extract the custom PythonModel instance variables
    df_full = model._model_impl.python_model.df_full
    logger.info("MLflow Production Model and Embedded Metadata DataFrame loaded successfully!")
    logger.info(f"Available Schema Columns: {df_full.columns.tolist()}")
except Exception as e:
    logger.error(f"Critical System Failure: Failed to load target model from registry: {e}")
    # Fallback to an empty schema to prevent immediate microservice crash on startup
    df_full = pd.DataFrame(columns=['title_x', 'cast', 'crew', 'genres', 'vote_average'])

# Initialize FastAPI application container
app = FastAPI(
    title="🎬 Movie Recommender API v2.0",
    description="High-performance recommendation gateway backed by MLflow and FastAPI clustering.",
    version="2.0"
)

# -------------------------------------------------------------------------
# 2. CORS SECURITY & INGRESS POLICIES
# -------------------------------------------------------------------------
# Permitting explicit Cross-Origin resource sharing protocols exclusively 
# for approved Streamlit UI rendering microservices.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------------------------------------------------------------
# 3. EXTERNAL METADATA ENRICHMENT FUNCTIONS (TMDB INTEGRATION)
# -------------------------------------------------------------------------
def get_poster(movie_title: str) -> str:
    """
    Queries TMDB downstream APIs to dynamically resolve asset poster paths.
    Returns a clean placeholder fallback URL upon latency timeouts or network anomalies.
    """
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie_title}
        res = requests.get(url, params=params, timeout=3.0).json()
        if res.get('results'):
            path = res['results'][0].get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception as e:
        logger.warning(f"Failed to fetch metadata poster for '{movie_title}': {e}")
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# -------------------------------------------------------------------------
# 4. CONTENT FILTERING & STRUCTURAL LOOKUP UTILITIES
# -------------------------------------------------------------------------
def search_movie(query: str):
    """Parses vectorized string sub-matches to identify titles matching the text criteria."""
    results = df_full[df_full['title_x'].str.contains(query, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data

def actor_movies(actor: str):
    """Filters data structures to extract films associated with a specific cast actor member."""
    results = df_full[df_full['cast'].str.contains(actor, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data

def director_movies(director: str):
    """Filters data structures to isolate historical films managed by the specified crew director."""
    results = df_full[df_full['crew'].str.contains(director, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data

# -------------------------------------------------------------------------
# 5. FASTAPI ROUTING ENDPOINTS
# -------------------------------------------------------------------------
@app.get("/recommend/", summary="Generate Similarity-Based Movie Recommendations")
def get_recommendations(title: str, n: int = 5):
    """
    Passes target title request parameters down into the custom MLflow cluster model space,
    evaluates neighbor metrics, maps TMDB assets, and structures the final response array.
    """
    try:
        # Construct input payload conforming to the model signature contracts
        input_df = pd.DataFrame({"title_x": [title]})
        recs = model.predict(input_df)

        # Catch out-of-vocabulary application errors passed from the inner wrapper class
        if isinstance(recs, pd.DataFrame) and "error" in recs.columns:
            raise HTTPException(status_code=404, detail=recs["error"].iloc[0])

        # Serialize results to application/json compliant structures
        recs_cleaned = json.loads(recs.head(n).to_json(orient="records"))

        # Hydrate dynamic poster URLs across payload items asynchronously
        for item in recs_cleaned:
            name = item.get('title_x') or title
            item['poster_url'] = get_poster(name)

        return recs_cleaned

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Inference Error: {str(e)}")

@app.get("/search/", summary="Text Search Database Titles")
def search(query: str):
    """Exposes title lookup filtering microservices."""
    return search_movie(query)

@app.get("/actor/", summary="Get Movies by Actor Name")
def get_actor(name: str):
    """Exposes cast profile database lookups."""
    return actor_movies(name)

@app.get("/director/", summary="Get Movies by Director Name")
def get_director(name: str):
    """Exposes crew director profile lookups."""
    return director_movies(name)

@app.get("/movie_list/", summary="Retrieve Complete Unique Title Inventory")
def get_all_titles():
    """Returns absolute collection array lists to populate frontend auto-complete search inputs."""
    return df_full['title_x'].unique().tolist()

@app.get("/", summary="System Health Check Gate")
def health():
    """Liveness probe indicator verifying endpoint operations status."""
    return {"status": "API is running smoothly", "model_uri": MODEL_URI}
