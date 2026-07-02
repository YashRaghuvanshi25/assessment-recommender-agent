from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"


class Embedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, text: str):
        """
        Returns a normalized embedding vector.
        """
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )


if __name__ == "__main__":

    embedder = Embedder()

    while True:

        query = input("\nEnter text (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        embedding = embedder.embed(query)

        print(f"\nEmbedding Dimension: {len(embedding)}")
        print(f"First 10 Values:\n{embedding[:10]}")