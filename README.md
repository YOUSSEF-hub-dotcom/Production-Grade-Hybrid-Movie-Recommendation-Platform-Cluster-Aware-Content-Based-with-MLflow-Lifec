# 🎬 Movie Recommendation Platform

> **Hybrid Content-Based Recommendation Engine powered by NLP, Hierarchical Clustering, MLflow, FastAPI, and Streamlit**

An end-to-end **Production-Ready Movie Recommendation Platform** built from a **Machine Learning Engineering** perspective.

Unlike traditional recommendation systems that rely solely on cosine similarity, this platform combines **Natural Language Processing (NLP), advanced feature engineering, dimensionality reduction, hierarchical clustering, MLOps, REST APIs, and an interactive dashboard** to generate highly relevant, scalable, and explainable movie recommendations.

The project demonstrates the complete machine learning lifecycle—from raw movie metadata to production deployment—following modern Data Science and MLOps best practices.

---

# 🌟 Project Overview

With thousands of movies available across modern streaming platforms, finding relevant content has become increasingly challenging.

Traditional recommendation systems frequently suffer from:

- Limited semantic understanding
- Noisy similarity calculations
- Poor scalability
- Lack of explainability
- Cold recommendation quality

This project addresses these challenges by building a **Hybrid Content-Based Recommendation Platform** capable of understanding movie semantics using textual information, metadata, clustering, and machine learning techniques.

Instead of comparing every movie against the entire dataset, recommendations are generated **within semantically similar clusters**, significantly improving recommendation quality while reducing unnecessary comparisons.

---

# 🎯 Project Objectives

The primary objectives of this project are:

✅ Build an intelligent movie recommendation engine

✅ Combine textual and structured movie metadata

✅ Improve recommendation relevance through clustering

✅ Reduce feature dimensionality while preserving semantic meaning

✅ Build a scalable recommendation pipeline

✅ Track experiments using MLflow

✅ Deploy recommendation services using FastAPI

✅ Develop an interactive Streamlit dashboard

✅ Follow production-ready MLOps practices

---

# 🚀 Why This Project?

Most recommendation projects stop after computing cosine similarity over TF-IDF vectors.

This project goes far beyond that by implementing a complete production workflow:

```text
Business Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Metadata Engineering
        ↓
Feature Fusion
        ↓
Dimensionality Reduction
        ↓
Hierarchical Clustering
        ↓
Cluster-aware Recommendation
        ↓
MLflow Lifecycle
        ↓
FastAPI Deployment
        ↓
Streamlit Dashboard
```

The result is a recommendation engine that is both **more explainable and more production-oriented** than traditional content-based systems.

---

# 📚 Table of Contents

- Project Overview
- Business Problem
- Project Objectives
- End-to-End Architecture
- Dataset Overview
- Exploratory Data Analysis
- Text Processing Pipeline
- Feature Engineering
- Recommendation Strategy
- Clustering Analysis
- Model Evaluation
- Recommendation Validation
- MLflow Lifecycle
- FastAPI Deployment
- Streamlit Dashboard
- Technology Stack
- Project Structure
- Installation
- Future Improvements
- Author

---

# 💼 Business Problem

Streaming platforms provide access to thousands of movies across multiple genres, making manual content discovery increasingly difficult.

Traditional recommendation systems often struggle with:

- Weak semantic understanding
- Limited personalization
- High computational complexity
- Poor recommendation explainability

This project aims to solve these problems by creating a hybrid recommendation engine that combines textual understanding with structured movie metadata to deliver more meaningful recommendations.

---

# ⭐ Key Features

### 🤖 Natural Language Processing

- Text Cleaning
- Tokenization
- Stopword Removal
- Porter Stemming
- TF-IDF Vectorization

---

### 📊 Feature Engineering

- Genre Encoding
- Metadata Fusion
- Numerical Feature Scaling
- Weighted Feature Combination

---

### 🧠 Machine Learning

- Truncated SVD
- Hierarchical Clustering
- Cluster-aware Cosine Similarity

---

### ⚙️ MLOps

- MLflow Experiment Tracking
- Model Registry
- Versioning
- Automatic Promotion
- Quality Gates

---

### 🚀 Deployment

- FastAPI REST API
- Streamlit Dashboard
- TMDB Poster Integration

---

# 🏗 End-to-End Architecture

```text
Raw Movie Metadata
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Genre Encoding
        │
        ▼
Numerical Feature Engineering
        │
        ▼
Weighted Feature Fusion
        │
        ▼
Truncated SVD
        │
        ▼
Hierarchical Clustering
        │
        ▼
Cluster-aware Cosine Similarity
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
FastAPI REST API
        │
        ▼
Streamlit Dashboard
```

---

# 📦 Dataset Overview

The recommendation engine is built using the **TMDB 5000 Movies Dataset**, containing rich movie metadata collected from The Movie Database (TMDB).

### Dataset Summary

| Property | Value |
|-----------|-------|
| Movies | **4,803** |
| Problem Type | Recommendation System |
| Recommendation Style | Content-Based Hybrid |
| Dataset Source | TMDB |
| Data Types | Text, Categorical, Numerical |

### Available Features

#### Text Features

- Overview
- Tagline
- Keywords
- Cast
- Crew

#### Categorical Features

- Genres

#### Numerical Features

- Budget
- Revenue
- Popularity
- Runtime


## 👨‍💻 Author

**Youssef Mahmoud**
AI / Data Science Student

[LinkedIn](https://www.linkedin.com/in/youssef-mahmoud-63b243361)

The diversity of these features enables the recommendation engine to capture both semantic similarity and structured movie characteristics.

---
