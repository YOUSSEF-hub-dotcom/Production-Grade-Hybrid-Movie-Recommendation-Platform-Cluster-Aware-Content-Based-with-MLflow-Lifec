import streamlit as st
import requests
import pandas as pd

# Gateway URL mapping the backend container orchestration port
API_URL = "http://127.0.0.1:8000"

# Configure global single-page application properties and responsive scales
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

# -------------------------------------------------------------------------
# 1. GRAPHICAL SYSTEM & CLIENT-SIDE STYLESHEET INJECTION
# -------------------------------------------------------------------------
# Custom styles to enforce a cinematic visual theme and resolve text color overlaps.
st.markdown("""
    <style>
    /* Full application context viewport background */
    .stApp {
        background: linear-gradient(to right, #141e30, #243B55);
        color: white !important;
    }

    /* Enforce typography contrast baselines across native web components */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: white !important;
    }

    /* Standardize data ingress components (Text Input & Selectbox wrappers) */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #1f2a38 !important;
        border-radius: 8px;
    }

    /* Input text buffer styling overrides */
    input, div[role="listbox"], div[data-baseweb="select"] div {
        color: white !important;
        background-color: #1f2a38 !important;
    }

    /* Intercept and correct unreadable background options within dropdown viewports */
    ul[role="listbox"] {
        background-color: #1f2a38 !important;
    }
    li[role="option"] {
        color: white !important;
        background-color: #1f2a38 !important;
    }
    li[role="option"]:hover {
        background-color: #3a506b !important;
    }

    /* Interactive button transition definitions */
    div.stButton>button {
        color: white !important;
        background-color: #3a506b !important;
        border: 1px solid #ffffff50;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton>button:hover {
        background-color: #2c3e50 !important;
        color: #FFD700 !important;
        border: 1px solid #FFD700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender System")
st.markdown("Interactive Movie Recommender using **FastAPI + Streamlit + MLflow**")

# Construct categorical navigation sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Search", "🎥 Recommendations", "⭐ Actor Insights", "🎬 Director Insights"]
)

# -------------------------------------------------------------------------
# 2. RESPONSIVE GRID RENDERING ENGINE
# -------------------------------------------------------------------------
def show_results(response):
    """
    Parses structural response payloads from upstream servers and generates 
    a balanced 3-column UI grid rendering movie posters and expanded profiles.
    """
    if response.status_code == 200:
        data = response.json()
        if data:
            # Chunk collection lists into 3-column rows dynamically
            for i in range(0, len(data), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(data):
                        movie = data[i + j]
                        with cols[j]:
                            # Render remote web asset with adaptive edge boundaries
                            st.image(movie['poster_url'], use_container_width=True)
                            st.subheader(movie.get('title_x', 'Unknown'))

                            # Extract and format multi-categorical genre badges
                            genres = movie.get('genres', '[]')
                            st.caption(f"🎭 {genres}")

                            # Interactive localized drawer to view unstructured data extensions
                            with st.expander("View Details"):
                                st.write(f"⭐ Rating: {movie.get('vote_average', 'N/A')}")
                                if 'overview' in movie:
                                    st.write(movie['overview'])
        else:
            st.warning("⚠️ No results found for your search.")

    elif response.status_code == 404:
        error_detail = response.json().get('detail', 'Movie not found')
        st.error(f"🔍 {error_detail}")

    else:
        st.error(f"🚫 Error Code: {response.status_code}. Please check backend logs.")


# -------------------------------------------------------------------------
# 3. HIGH-LATENCY NETWORK CACHING DECLARATIONS
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Caches structural index data for 1 hour to prevent unnecessary network overhead
def fetch_movie_list():
    """
    Hits backend lookups to populate frontend auto-complete forms on cold-starts.
    Gracefully handles empty array states on initial server boots.
    """
    try:
        res = requests.get(f"{API_URL}/movie_list/", timeout=5.0)
        return res.json()
    except Exception as e:
        return []

# Execute cached pull sequence
all_movies = fetch_movie_list()

# -------------------------------------------------------------------------
# 4. VIEW CONTROLLERS & EVENT BINDINGS
# -------------------------------------------------------------------------

# Tab 1: Direct Content Full-Text Pattern Search
with tab1:
    st.subheader("Search Movies")
    query = st.text_input("Enter movie name:", key="search_input")
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                res = requests.get(f"{API_URL}/search/", params={"query": query})
                show_results(res)

# Tab 2: Latent Vector Proximity Recommendation Engine
with tab2:
    st.subheader("Get Recommendations")
    # Using secure selectbox widgets to prevent typographic query syntax injection
    title = st.selectbox("Select a movie you liked:", all_movies, index=None)
    n = st.slider("Number of recommendations:", 3, 12, 6)

    if st.button("Recommend"):
        if title:
            with st.spinner("Finding similar movies..."):
                res = requests.get(f"{API_URL}/recommend/", params={"title": title, "n": n})
                show_results(res)

# Tab 3: Cast-Profile Data Sub-Setting Filter
with tab3:
    st.subheader("Actor Movies")
    actor = st.text_input("Enter actor name:", key="actor_input")
    if st.button("Get Actor Movies"):
        if actor:
            with st.spinner("Fetching..."):
                res = requests.get(f"{API_URL}/actor/", params={"name": actor})
                show_results(res)

# Tab 4: Production Crew Sub-Setting Filter
with tab4:
    st.subheader("Director Movies")
    director = st.text_input("Enter director name:", key="dir_input")
    if st.button("Get Director Movies"):
        if director:
            with st.spinner("Fetching..."):
                res = requests.get(f"{API_URL}/director/", params={"name": director})
                show_results(res)
