# 🎬 Movie Recommendation Platform

> **Hybrid Content-Based Recommendation Engine powered by NLP, Hierarchical Clustering, MLflow, FastAPI, and Streamlit**

An end-to-end **production-oriented Movie Recommendation Platform** built from a **Data Science + ML Engineering** perspective. Rather than relying on plain cosine similarity, the platform combines **Natural Language Processing, metadata engineering, dimensionality reduction, hierarchical clustering, MLOps, API serving, and an interactive dashboard** to deliver scalable and explainable movie recommendations.

---

# 📌 Project Overview

Modern streaming services contain thousands of movies, making manual content discovery increasingly difficult.

This project builds a hybrid recommendation engine capable of discovering semantically similar movies using movie plots, genres, actors, directors, and numerical metadata.

The platform emphasizes:

- Explainable recommendations
- Scalable architecture
- Production-ready deployment
- Reproducible ML workflow
- Modular engineering design

---

# 🎯 Objectives

- Build an intelligent recommendation engine.
- Combine NLP with metadata.
- Improve recommendation relevance using clustering.
- Deploy the model through FastAPI.
- Track experiments using MLflow.
- Provide an interactive Streamlit interface.

---

# 🧠 End-to-End Architecture

```text
Raw Movie Metadata
        ↓
Data Cleaning & Validation
        ↓
EDA & Industry Insights
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Genre Encoding
        ↓
Numerical Feature Engineering
        ↓
Weighted Feature Fusion
        ↓
Truncated SVD
        ↓
Hierarchical Clustering
        ↓
Cluster-aware Cosine Similarity
        ↓
MLflow Lifecycle
        ↓
FastAPI
        ↓
Streamlit Dashboard
```

---

# 📦 Dataset

- Movies: **4,803**
- Sources include textual, categorical and numerical metadata.
- Text: Overview, Tagline, Keywords, Cast & Crew
- Categories: Genres
- Numerical: Budget, Revenue, Popularity, Runtime

---

# 📊 Exploratory Data Analysis

The project investigated:

- Genre distribution
- Duplicate movies
- Director productivity
- Actor specialization
- Collaboration networks
- Cast statistics
- Movie era comparison

These insights directly influenced feature engineering and weighting.

---

# 🧪 Feature Engineering Pipeline

- Text Cleaning
- Tokenization
- Stopword Removal
- Porter Stemming
- TF-IDF
- MultiLabelBinarizer
- StandardScaler
- Truncated SVD

## Feature Weighting

| Feature | Weight |
|---|---:|
| Genres | 2.0 |
| Text | 1.5 |
| Numerical | 1.0 |

---

# 🤖 Recommendation Strategy

1. TF-IDF
2. Truncated SVD
3. Hierarchical Clustering
4. Cluster-aware Cosine Similarity

This minimizes noisy global comparisons while improving semantic relevance.

---

# 🔬 Why Hierarchical Clustering?

- Deterministic
- Better for irregular clusters
- Dendrogram visualization
- More interpretable than K-Means
- Produces coherent recommendation groups

---

# 📈 Cluster Evaluation

| Clusters | Silhouette |
|---:|---:|
|3|0.1661|
|5|0.1545|
|10|0.0836|

Although 10 clusters produced a lower Silhouette Score, they generated more meaningful thematic movie groups and superior recommendation quality.

---

# 📊 Final Model Performance

| Metric | Value |
|---|---:|
|Movies|4803|
|Feature Matrix|(4803,224)|
|Explained Variance|25.58%|
|Production Version|15|
|Promotion Threshold|0.05|

---

# ✅ Recommendation Validation

Example query:

**Four Rooms**

Top recommendations included:

- The Big Lebowski
- Lock, Stock and Two Smoking Barrels
- Seven Psychopaths
- 8 Heads in a Duffel Bag

Similarity exceeded **90%**, demonstrating strong semantic alignment.

---

# 🔄 MLflow Lifecycle

- Experiment Tracking
- Parameter Logging
- Metrics
- Artifacts
- PyFunc Model
- Model Registry
- Quality Gates
- Automatic Promotion

---

# 🌐 Production FastAPI Layer

- Recommendation endpoint
- Search endpoint
- Actor exploration
- Director exploration
- TMDB poster integration
- MLflow model loading

---

# 🖥 Streamlit Dashboard

- Movie search
- Recommendation cards
- Posters
- Ratings
- Metadata visualization

---

# 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| NLP | TF-IDF | Semantic Encoding |
| Feature Engineering | MultiLabelBinarizer | Genre Encoding |
| Dimensionality Reduction | Truncated SVD | Compression |
| Clustering | Hierarchical Clustering | Movie Grouping |
| Similarity | Cosine Similarity | Recommendation |
| MLOps | MLflow | Lifecycle |
| API | FastAPI | Serving |
| UI | Streamlit | Dashboard |

---

# 📁 Project Structure

```text
project/
├── data_pipeline.py
├── text_preprocessing.py
├── EDA.py
├── model.py
├── mlflow_lifecycle.py
├── api.py
├── app.py
├── MLproject
├── conda.yaml
└── README.md
```

---

# 🚀 How to Run

```bash
mlflow run .
uvicorn api:app --reload
streamlit run app.py
```

---

# 💡 Why This Project Is Strong

- End-to-end Data Science workflow
- Explainable recommendation pipeline
- Hybrid feature engineering
- Production-ready MLOps
- API deployment
- Interactive dashboard

---

# 🌱 Future Improvements

- FAISS ANN search
- User-personalized recommendations
- Feedback loop
- CI/CD
- Vector database integration
- RAG-assisted movie search

---

# 👨‍💻 Author

**Youssef Mahmoud**

Aspiring AI Engineer | Data Scientist
