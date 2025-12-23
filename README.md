# 🎓 UBC Clustering-Based Course Recommendation System

This project implements a **course recommendation system using unsupervised learning**.  
Instead of predicting explicit ratings, the system **groups similar courses and/or users using clustering algorithms** and generates recommendations based on cluster membership.

The full analysis and implementation are provided in the Jupyter Notebook:
`recommendation.ipynb`.

---

## 📌 Project Overview

Traditional recommender systems often rely on collaborative filtering or matrix factorization.  
In this project, we explore an alternative approach using **clustering techniques** to discover structure in course data and recommend courses within similar groups.

The project compares three clustering methods:

- **K-Means Clustering**
- **Agglomerative Hierarchical Clustering**
- **Spectral Clustering**

---

## 🧠 Recommendation Strategy

The recommendation pipeline follows these steps:

1. **Feature Extraction**
   - Course features are transformed into numerical representations suitable for clustering.
   - Features may include course ratings, enrollment patterns, or topic-related attributes.

2. **Clustering**
   - Courses are grouped into clusters based on similarity.
   - Each clustering algorithm assigns courses to distinct groups.

3. **Recommendation**
   - Given a course or user preference, recommendations are generated from the **same cluster**.
   - Courses within the same cluster are assumed to be similar in content or appeal.

---

## 🔍 Clustering Algorithms Used

### 1️⃣ K-Means Clustering
- Partitions courses into `k` clusters
- Optimizes intra-cluster similarity
- Efficient and scalable for larger datasets

### 2️⃣ Agglomerative Hierarchical Clustering
- Builds clusters bottom-up by merging similar courses
- Does not require pre-selecting the number of clusters
- Provides interpretability via dendrogram structure

### 3️⃣ Spectral Clustering
- Uses graph-based similarity representation
- Captures complex, non-linear relationships
- Effective when cluster boundaries are not spherical

---

## 📂 Repository Structure
```
course_recommender/
│
├── recommendation.ipynb # Clustering-based recommendation system
└── README.md # Project documentation
```

## ⚙️ Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**
