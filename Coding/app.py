import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    /* خلفية التطبيق كاملة */
    .stApp {
        background: linear-gradient(to right, #141e30, #243B55);
        color: white !important;
    }

    /* توحيد لون العناوين والنصوص */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: white !important;
    }

    /* تعديل صناديق الإدخال (Text Input & Selectbox) */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #1f2a38 !important;
        border-radius: 8px;
    }

    /* لون النص جوه الصناديق */
    input, div[role="listbox"], div[data-baseweb="select"] div {
        color: white !important;
        background-color: #1f2a38 !important;
    }

    /* حل مشكلة اللون الأبيض في القائمة المنسدلة (Selectbox Dropdown) */
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

    /* تعديل أزرار ستريمليت */
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Search", "🎥 Recommendations", "⭐ Actor Insights", "🎬 Director Insights"]
)


# تعديل دالة show_results لإضافة تفاصيل أكثر
def show_results(response):
    if response.status_code == 200:
        data = response.json()
        if data:
            for i in range(0, len(data), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(data):
                        movie = data[i + j]
                        with cols[j]:
                            # تصميم يشبه الكارت
                            st.image(movie['poster_url'], use_container_width=True)
                            st.subheader(movie.get('title_x', 'Unknown'))

                            # عرض الأنواع (Genres) بشكل لطيف
                            genres = movie.get('genres', '[]')
                            st.caption(f"🎭 {genres}")

                            # زرار "Show More" لرؤية التفاصيل
                            with st.expander("View Details"):
                                st.write(f"⭐ Rating: {movie.get('vote_average', 'N/A')}")
                                # إذا كان الموديل يرجع الـ Overview
                                if 'overview' in movie:
                                    st.write(movie['overview'])
        else:
            st.warning("⚠️ No results found for your search.")

    elif response.status_code == 404:
        error_detail = response.json().get('detail', 'Movie not found')
        st.error(f"🔍 {error_detail}")

    else:
        st.error(f"🚫 Error Code: {response.status_code}. Please check backend logs.")


with tab1:
    st.subheader("Search Movies")
    query = st.text_input("Enter movie name:", key="search_input")
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                res = requests.get(f"{API_URL}/search/", params={"query": query})
                show_results(res)


# --- جلب قائمة الأفلام مرة واحدة عند تشغيل الـ App ---
@st.cache_data
def fetch_movie_list():
    try:
        res = requests.get(f"{API_URL}/movie_list/")
        return res.json()
    except:
        return []


all_movies = fetch_movie_list()

with tab2:
    st.subheader("Get Recommendations")
    # استخدام selectbox بدل text_input لمنع أخطاء الكتابة
    title = st.selectbox("Select a movie you liked:", all_movies, index=None)
    n = st.slider("Number of recommendations:", 3, 12, 6)

    if st.button("Recommend"):
        if title:
            with st.spinner("Finding similar movies..."):
                res = requests.get(f"{API_URL}/recommend/", params={"title": title, "n": n})
                show_results(res)

with tab3:
    st.subheader("Actor Movies")
    actor = st.text_input("Enter actor name:", key="actor_input")
    if st.button("Get Actor Movies"):
        if actor:
            with st.spinner("Fetching..."):
                res = requests.get(f"{API_URL}/actor/", params={"name": actor})
                show_results(res)

with tab4:
    st.subheader("Director Movies")
    director = st.text_input("Enter director name:", key="dir_input")
    if st.button("Get Director Movies"):
        if director:
            with st.spinner("Fetching..."):
                res = requests.get(f"{API_URL}/director/", params={"name": director})
                show_results(res)