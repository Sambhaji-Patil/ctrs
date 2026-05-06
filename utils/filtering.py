import requests
import numpy as np
from typing import List

from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
import os

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("REQUESTY_API_KEY")
API_URL = "https://router.requesty.ai/v1/embeddings"



def get_embeddings(texts: List[str]) -> np.ndarray:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input": texts,
            "model": "openai/text-embedding-3-small",
            "encoding_format": "float"
        },
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")

    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings)


def batched_embeddings(texts: List[str], batch_size=50):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = get_embeddings(batch)
        all_embeddings.append(emb)

    return np.vstack(all_embeddings)

def get_representatives(texts: List[str], eps: float, min_samples: int):
    embeddings = batched_embeddings(texts)

    distance_matrix = cosine_distances(embeddings)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    ).fit(distance_matrix)

    labels = clustering.labels_
    clusters = {}

    for idx, label in enumerate(labels):
        if label == -1:
            clusters[f"noise_{idx}"] = [idx]
        else:
            clusters.setdefault(label, []).append(idx)

    representatives = []

    for _, indices in clusters.items():
        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        distances = cosine_distances(
            cluster_embeddings, centroid.reshape(1, -1)
        ).flatten()

        best_idx = indices[np.argmin(distances)]
        representatives.append(texts[best_idx])

    return representatives