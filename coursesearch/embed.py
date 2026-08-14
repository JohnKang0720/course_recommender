"""Embed course descriptions, then search and cluster in that vector space."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

MODEL = "all-MiniLM-L6-v2"


def embed(texts):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL).encode(list(texts), normalize_embeddings=True, batch_size=64)


def search(query_vec, vectors, k=6):
    scores = vectors @ query_vec
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


def cluster(vectors, k=14):
    return KMeans(k, n_init=10, random_state=0).fit_predict(vectors)


def compare_clusterings(vectors, k=14):
    labels = {
        "K-Means": KMeans(k, n_init=10, random_state=0).fit_predict(vectors),
        "Hierarchical": AgglomerativeClustering(k).fit_predict(vectors),
        "Spectral": SpectralClustering(k, random_state=0, affinity="nearest_neighbors").fit_predict(vectors),
    }
    return {name: silhouette_score(vectors, lab) for name, lab in labels.items()}
