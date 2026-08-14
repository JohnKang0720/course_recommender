"""Scrape → embed → cluster → figures → export the static demo.

Caches the scrape and embeddings so re-runs are fast.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coursesearch import embed, scrape, viz

REPO = Path(__file__).resolve().parent.parent
DATA, FIG, DOCS = REPO / "data", REPO / "figures", REPO / "docs"


def load_courses():
    cache = DATA / "courses.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("Scraping UBC catalog…")
    df = scrape.scrape_all()
    DATA.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load_vectors(df):
    cache = DATA / "embeddings.npy"
    if cache.exists():
        return np.load(cache)
    print(f"Embedding {len(df)} courses…")
    vectors = embed.embed(df["name"] + ". " + df["description"])
    np.save(cache, vectors)
    return vectors


def main():
    df = load_courses()
    vectors = load_vectors(df).astype(np.float32)
    print(f"{len(df)} courses · {vectors.shape[1]}-d embeddings")

    clusters = embed.cluster(vectors)
    scores = embed.compare_clusterings(vectors)
    coords = viz.umap_2d(vectors)
    print("Silhouette: " + " · ".join(f"{k} {v:.3f}" for k, v in scores.items()))

    viz.topic_map(coords, clusters, FIG / "topic_map.png")
    viz.cluster_comparison(scores, FIG / "clustering.png")
    export(df, vectors, clusters, coords, scores)


def export(df, vectors, clusters, coords, scores):
    DOCS.mkdir(exist_ok=True)
    courses = [{"code": r.code, "name": r.name, "desc": r.description[:220],
                "cluster": int(clusters[i]), "x": round(float(coords[i, 0]), 2), "y": round(float(coords[i, 1]), 2)}
               for i, r in enumerate(df.itertuples())]
    (DOCS / "courses.json").write_text(json.dumps(courses, separators=(",", ":")))
    vectors.tofile(DOCS / "vectors.bin")
    silhouette = {k: round(float(v), 3) for k, v in scores.items()}
    (DOCS / "meta.json").write_text(json.dumps({"n": len(df), "dim": vectors.shape[1], "silhouette": silhouette}))
    for f in FIG.glob("*.png"):
        (DOCS / "figures").mkdir(exist_ok=True)
        (DOCS / "figures" / f.name).write_bytes(f.read_bytes())
    print(f"Exported demo -> docs/ ({(DOCS / 'vectors.bin').stat().st_size / 1e6:.1f} MB vectors)")


if __name__ == "__main__":
    main()
