# 🎬 Movie Recommendation Platform

> **Hybrid Content-Based Recommendation Engine powered by NLP, Hierarchical Clustering, MLflow, FastAPI, and Streamlit.**

An end-to-end **production-oriented Movie Recommendation Platform** built from a combined **Data Science + Machine Learning Engineering** perspective. Rather than relying solely on global cosine similarity, the platform integrates **Natural Language Processing, advanced metadata engineering, latent vector dimensionality reduction, hierarchical clustering, MLOps lifecycle tracking, API microservices, and an interactive frontend dashboard** to deliver highly scalable, cluster-aware, and explainable movie recommendations.

---

## 📌 Project Overview

Modern streaming infrastructure scales to tens of thousands of media assets, causing intense content discovery friction for end-users. 

This platform resolves this by engineering a hybrid mathematical representation of films—fusing semantic plot contexts, keyword bags, and cast/crew metadata with numerical statistics (budget, popularity, runtime). By forcing recommendations to filter through localized hierarchical clusters, the engine minimizes global vector noise and yields mathematically and thematically superior recommendations.

### 🎯 Key Engineering Objectives
*   **Intelligent Hybrid Vectorization:** Seamlessly merge textual semantic embeddings with weighted categorical variables and normalized numerical features.
*   **Cluster-Aware Relevance Tuning:** Mitigate global high-dimensional noise by grouping similar assets prior to similarity computation.
*   **Production-Grade Serving Infrastructure:** Expose recommendation matrices via a low-latency REST API constructed with FastAPI.
*   **Rigorous MLOps Governance:** Leverage MLflow for systemic hyperparameter tracking, model artifact serialization, and automated metric-gate validation pipelines.

---

## 🧠 End-to-End Architectural Pipeline

```text
       [ Raw Movie Metadata Ingestion ]
                      │
        [ Data Cleaning & Drop Fields ]
                      │
       [ Categorical & JSON Extraction ]
                      │
   ┌──────────────────┴──────────────────┐
   ▼                                     ▼
[ Text Mining Pipeline ]       [ Metadata Feature Engineering ]
   │                                     │
   ├─► Tokenization                      ├─► MultiLabelBinarizer (Genres)
   ├─► Stopword Removal                  └─► StandardScaler (Numerical Data)
   ├─► Porter Stemming                   
   └─► TF-IDF Vectorization              
   │                                     │
   └──────────────────┬──────────────────┘
                      ▼
         [ Weighted Feature Fusion ]
                      ▼
          [ Latent SVD Compression ]
                      ▼
        [ Hierarchical Clustering ]
                      ▼
    [ Cluster-Aware Cosine Similarity ]
                      ▼
       [ MLflow Experiment Tracking ]
                      ▼
         [ FastAPI Microservice ]
                      ▼
        [ Streamlit Web Application ]
📦 Dataset TopologyTotal Corpus Size: 4,803 structured feature records.Textual Dimensions: Movie Overviews, Taglines, Raw Keyword Tokens, Cast Arrays, and Production Crew Hierarchies.Categorical Dimensions: Multi-genre indexing classifications.Numerical Dimensions: Production Budget, Box-Office Revenue, Global Popularity Indexes, and Asset Runtime.📊 Exploratory Data Analysis & Feature EngineeringExploratory analysis on the TMDB dataset unmasked significant data traits (e.g., strong actor specialization in specific genres, highly localized director collaboration networks, and distinct structural eras pre/post the year 2000). These discoveries directly dictated our custom feature fusion logic.Multi-Modal Feature Weighting StrategyTo ensure that explicit thematic traits dominate implicit numerical trends, a custom weighting factor matrix was engineered prior to dimension reduction:Metadata Feature DomainMatrix Weight MappingTechnical Extraction TargetGenres2.0MultiLabelBinarized Sparse MatrixTextual Semantics1.5Stemmed Token TF-IDF MatrixNumerical Dynamics1.0StandardScaler Scaled MetricsDimensionality ReductionThe engineered high-dimensional fused space was compressed using Truncated SVD (Singular Value Decomposition) down to 224 components, striking an optimal balance between runtime latency and matrix information preservation.🔬 Unsupervised Machine Learning StrategyWhy Agglomerative Hierarchical Clustering?Unlike K-Means which enforces strict spherical boundaries and requires random centroid initializations, Agglomerative Hierarchical Clustering was deployed because:Deterministic Execution: Eliminates random state variances across model training runs.Arbitrary Geometric Structures: Adapts organically to irregular, multi-density data distributions.Thematic Interpretability: Tree structures align accurately with real-world genre taxonomies.Cluster Evaluation TopologyThe cluster space was evaluated across multiple partitions to isolate the optimal hyperparameter cut-off:Target Cluster Partitions (K)Silhouette Coefficient ScoreQualitative Thematic Coherence30.1661Too broad; mixed disparate genres.50.1545Standard macro grouping behavior.100.0836Selected: Generated tightly-focused, hyper-relevant thematic pools.💡 Engineering Trade-off Note: While a lower $K$ yielded a superior mathematical silhouette score, setting $K=10$ generated dramatically higher contextual relevance and completely eliminated anomalous recommendations during testing.📈 Model Performance & Validation ProfilesProduction Deployment BenchmarksTotal Vectorized Records: 4,803 active profiles.Compressed Latent Shape: (4803, 224)Cumulative Explained Variance: 25.58%Current Production Artifact Build: v15Deterministic Recommendation ValidationInput Anchor Query: Four Rooms (Thematic Profile: Dark Comedy, Anthology, Indicated Ensemble Cast).Generated Results Portfolio:The Big LebowskiLock, Stock and Two Smoking BarrelsSeven Psychopaths8 Heads in a Duffel BagStatistical Threshold: Vector similarity proximity crossed > 90%, confirming precise semantic and stylistic convergence.🔄 Automated MLOps Governance & DeploymentMLflow Lifecycle ArchitectureThe platform leverages an automated MLOps wrapper to enforce continuous deployment gates:Parameters Logged: TF-IDF max features, SVD components, clustering linkage, and distance metrics.Artifact Logging: Graph tracking of Silhouette scores, Dendrogram trees, and pickled pipeline layers.Automated Quality Gate: Models crossing the promotion threshold (0.05 Silhouette Delta compared to the baseline) are programmatically tagged as Production within the MLflow Model Registry.Serving & Interface InfrastructureMicroservice Tier (FastAPI): Exposes secure endpoints (/recommend/, /search/, /actor/, /director/) parsing the system-cached MLflow serialization pipeline. Includes native TMDB API handlers for real-time poster hotlinking.Consumer Interface (Streamlit): A customized, dark-themed responsive single-page dashboard executing clean asynchronous data calls to the FastAPI backend microservice.🛠️ Technology Stack ArchitectureNLP & Vector Processing: NLTK, Scikit-Learn (TF-IDF, MultiLabelBinarizer, Truncated SVD).Unsupervised Modeling: Scikit-Learn (AgglomerativeClustering, Cosine Similarity).MLOps & Lifecycles: MLflow, Conda Virtualization Environments.Backend Engineering: FastAPI, Uvicorn ASGI, Requests.Frontend Dashboard: Streamlit Web Server, Asynchronous UI Caching, CSS Injection.🚀 Execution & Deployment ProtocolTo initialize the end-to-end environment pipeline locally, execute the following CLI commands:Bash# 1. Execute the MLflow pipeline to train, validate, and register the artifact
mlflow run .

# 2. Boot up the low-latency FastAPI production serving layer
uvicorn api:app --host 127.0.0.1 --port 8000 --reload

# 3. Launch the responsive Streamlit analytical interface
streamlit run app.py
🌱 Future Engineering Roadmap[ ] Integrate FAISS (Facebook AI Similarity Search) to achieve Sub-millisecond Approximate Nearest Neighbor (ANN) vector processing.[ ] Refactor the architecture into a fully decoupled distributed infrastructure using Docker Containerization and GitHub Actions CI/CD.[ ] Implement Vector Database Ingestion (Pinecone or Milvus) to manage dynamic real-time data streaming.[ ] Build a hybrid user-feedback storage loop to capture clickstream rewards and personalize cluster weights dynamically.👨‍💻 Author ProfileYoussef MahmoudAspiring AI Engineer | Machine Learning Operations & Data Scientist💼 LinkedIn Profile: Connect on LinkedIn
