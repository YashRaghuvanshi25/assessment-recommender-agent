import time
from tool.query_decomposer import QueryDecomposer
from tool.retriever import SHLRetriever


class RecommendationEngine:
    def __init__(self):
        print("Initializing Recommendation Engine...")
        self.decomposer = QueryDecomposer()
        self.retriever = SHLRetriever()

    def recommend(self, user_query: str, top_k: int = 5):
        overall_start = time.perf_counter()

        t = time.perf_counter()
        decomposition = self.decomposer.decompose(user_query)
        print(f"[TIMING] Query decomposition: {time.perf_counter() - t:.3f}s")

        retrieval_queries = decomposition.get("retrieval_queries", [user_query])

        retrieval_buckets = {}

        retrieval_start = time.perf_counter()
        for query in retrieval_queries:
            q_start = time.perf_counter()
            results = self.retriever.search(query, top_k=top_k)
            print(f"[TIMING] Retrieval for '{query}': {time.perf_counter() - q_start:.3f}s")

            best_results = {}
            for result in results:
                assessment_id = result["entity_id"]

                if (
                    assessment_id not in best_results
                    or result["similarity_score"] > best_results[assessment_id]["similarity_score"]
                ):
                    best_results[assessment_id] = result

            bucket = list(best_results.values())
            bucket.sort(
                key=lambda x: x["similarity_score"],
                reverse=True,
            )

            retrieval_buckets[query] = bucket

        print(f"[TIMING] Total retrieval: {time.perf_counter() - retrieval_start:.3f}s")

        final_recommendations = []
        selected_ids = set()
        coverage = {}

        for retrieval_query, bucket in retrieval_buckets.items():
            if not bucket:
                continue

            top_result = bucket[0]
            assessment_id = top_result["entity_id"]

            if assessment_id in selected_ids:
                coverage[retrieval_query] = top_result["name"]
            else:
                selected_ids.add(assessment_id)
                final_recommendations.append(top_result)
                coverage[retrieval_query] = top_result["name"]

        print(f"[TIMING] Total recommend(): {time.perf_counter() - overall_start:.3f}s")
        return retrieval_buckets, final_recommendations, coverage


if __name__ == "__main__":
    engine = RecommendationEngine()

    while True:
        user_query = input("\nEnter query (or 'exit'): ").strip()

        if user_query.lower() == "exit":
            break

        buckets, recommendations, coverage = engine.recommend(user_query)

        for retrieval_query, results in buckets.items():
            print(f"\n=== {retrieval_query} ===\n")

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['name']}")
                print(f"   Score : {result['similarity_score']:.4f}")
                print(f"   Link  : {result.get('link', 'N/A')}")
                print()

        print("\n========== Final Recommendations ==========")
        for i, result in enumerate(recommendations, 1):
            print(f"{i}. {result['name']}")
            print(f"   Score : {result['similarity_score']:.4f}")
            print(f"   Link  : {result.get('link', 'N/A')}")
            print()

        print("========== Coverage ==========")
        for retrieval_query, assessment in coverage.items():
            print(f"{retrieval_query} -> {assessment}")