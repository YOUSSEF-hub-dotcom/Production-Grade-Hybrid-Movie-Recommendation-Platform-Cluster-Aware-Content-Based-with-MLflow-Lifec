import pandas as pd
import mlflow.pyfunc
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

TMDB_API_KEY = "698fe165093c8bb05c621472b0c91aac"
MODEL_URI = "models:/MovieRecommenderSystem/Production"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)

    df_full = model._model_impl.python_model.df_full

    print(" تم تحميل الموديل والبيانات بنجاح!")
    print(f"الأعمدة المتاحة: {df_full.columns.tolist()}")
except Exception as e:
    print(f" فشل تحميل الموديل: {e}")
    df_full = pd.DataFrame(columns=['title_x', 'cast', 'crew', 'genres', 'vote_average'])

app = FastAPI(title=" Movie Recommender API v2.0")

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

def get_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie_title}
        res = requests.get(url, params=params).json()
        if res.get('results'):
            path = res['results'][0].get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

def search_movie(query: str):
    results = df_full[df_full['title_x'].str.contains(query, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data

def actor_movies(actor: str):
    results = df_full[df_full['cast'].str.contains(actor, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data

def director_movies(director: str):
    results = df_full[df_full['crew'].str.contains(director, case=False, na=False)].head(10)
    data = results[['title_x', 'vote_average', 'genres']].to_dict(orient="records")
    for item in data:
        item['poster_url'] = get_poster(item['title_x'])
    return data


import json

@app.get("/recommend/")
def get_recommendations(title: str, n: int = 5):
    try:
        input_df = pd.DataFrame({"title_x": [title]})
        recs = model.predict(input_df)

        if isinstance(recs, pd.DataFrame) and "error" in recs.columns:
            raise HTTPException(status_code=404, detail=recs["error"].iloc[0])

        recs_cleaned = json.loads(recs.head(n).to_json(orient="records"))

        for item in recs_cleaned:
            name = item.get('title_x') or title
            item['poster_url'] = get_poster(name)

        return recs_cleaned

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(query: str):
    return search_movie(query)

@app.get("/actor/")
def get_actor(name: str):
    return actor_movies(name)

@app.get("/director/")
def get_director(name: str):
    return director_movies(name)

@app.get("/movie_list/")
def get_all_titles():
    return df_full['title_x'].unique().tolist()

@app.get("/")
def health():
    return {"status": "API is running "}
