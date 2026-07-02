

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path(__file__).parent / "catalog_texts.json"
OUTPUT_FILE = Path(__file__).parent / "catalog_embeddings.npy"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def main():
    # Load the model once
    model = SentenceTransformer(MODEL_NAME)
    # Load input texts
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    embeddings = []
    total = len(items)
    for idx, item in enumerate(items, 1):
        text = item["text"]
        embedding = model.encode(text, normalize_embeddings=True)
        embeddings.append(embedding)
        print(f"[{idx}/{total}] Embedded: {item.get('name', '<unknown assessment>')}")
    embeddings = np.stack(embeddings)
    np.save(OUTPUT_FILE, embeddings)
    print(f"Saved embeddings of shape {embeddings.shape} to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()