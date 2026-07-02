

from pathlib import Path

import json
import numpy as np

from tool.embed_query import Embedder 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "catalog preprocesing"
EMBEDDINGS_FILE = BASE_DIR / "catalog_embeddings.npy"
CATALOG_FILE = BASE_DIR / "shl_product_catalog.json"
TOP_K = 10


class SHLRetriever:
    def __init__(self):
        self.embedder = Embedder()

        print("Loading embeddings...")
        self.embeddings = np.load(EMBEDDINGS_FILE)

        print("Loading catalog...")
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        print(f"Loaded {len(self.catalog)} assessments.")

    def search(self, query: str, top_k: int = TOP_K):
        query_embedding = self.embedder.embed(query)

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            assessment = self.catalog[idx].copy()
            assessment["similarity_score"] = float(scores[idx])
            results.append(assessment)

        return results


if __name__ == "__main__":
    retriever = SHLRetriever()

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        results = retriever.search(query)

        print("\nTop Results:\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}")
            print(f"   Score : {result['similarity_score']:.4f}")
            print(f"   Link  : {result.get('link', 'N/A')}")
            print()