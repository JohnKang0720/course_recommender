"""The UMAP topic map and the clustering comparison."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

INK, MUTE, GRID = "#141414", "#8a8a8a", "#e6e6e6"


def umap_2d(vectors):
    return umap.UMAP(n_neighbors=30, min_dist=0.5, spread=1.5, metric="cosine",
                     random_state=42).fit_transform(vectors)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def topic_map(coords, clusters, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab20", s=9, alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title("The course catalog as a topic map", color=INK, fontsize=13, loc="left")
    ax.set_xlabel("Each point is a course; nearby courses are semantically similar", color=MUTE, fontsize=9)
    save(fig, path)


def cluster_comparison(scores, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    names, values = list(scores), list(scores.values())
    ax.bar(names, values, color=INK, width=0.6)
    for i, v in enumerate(values):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", color=INK, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Clustering the embeddings is weak — which is why search wins",
                 color=INK, fontsize=11.5, loc="left")
    save(fig, path)
