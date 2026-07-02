import time
from tool.recommendation_engine import RecommendationEngine


TEST_CASES = [
    {
        "query": "Graduate management trainee battery covering cognitive, personality and situational judgement",
        "expected": [
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "query": "Java assessment for entry-level developers",
        "expected": [
            "Java",
        ],
    },
    {
        "query": "Senior leadership selection for CXOs and directors",
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "query": "Sales assessment for graduates",
        "expected": [
            "Graduate Scenarios",
        ],
    },
]


def evaluate():
    engine = RecommendationEngine()

    total_recall = 0.0
    total_precision = 0.0
    total_grounded = 0
    total_latency = 0.0

    print("=" * 80)
    print("SHL Recommendation Evaluation")
    print("=" * 80)

    for idx, case in enumerate(TEST_CASES, 1):

        start = time.perf_counter()

        # Only evaluation requests Top-10
        _, recommendations, _ = engine.recommend(
            case["query"],
            max_recommendations=10,
        )

        latency = time.perf_counter() - start
        total_latency += latency

        returned = [r["name"] for r in recommendations[:10]]
        expected = case["expected"]

        # Recall@10
        hits = sum(
            any(exp.lower() in rec.lower() for rec in returned)
            for exp in expected
        )

        recall = hits / len(expected)

        # Precision@10
        precision = hits / max(len(returned), 1)

        total_recall += recall
        total_precision += precision

        # Groundedness
        grounded = all(
            "name" in r and "url" in r
            for r in recommendations
        )

        if grounded:
            total_grounded += 1

        print(f"\nTest {idx}")
        print("-" * 80)
        print(f"Query              : {case['query']}")
        print(f"Expected           : {expected}")
        print(f"Returned           : {returned}")
        print(f"Recall@10          : {recall:.2f}")
        print(f"Precision@10       : {precision:.2f}")
        print(f"Grounded           : {grounded}")
        print(f"Latency            : {latency:.2f}s")

    avg_recall = total_recall / len(TEST_CASES)
    avg_precision = total_precision / len(TEST_CASES)
    avg_grounded = total_grounded / len(TEST_CASES)
    avg_latency = total_latency / len(TEST_CASES)

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Retrieval Quality (Recall@10)          : {avg_recall:.2f}")
    print(f"Recommendation Relevance (Precision)   : {avg_precision:.2f}")
    print(f"Groundedness                           : {avg_grounded:.2f}")
    print(f"Average Response Latency               : {avg_latency:.2f}s")
    print(f"Overall Recommendation Accuracy        : {(avg_recall + avg_precision) / 2:.2f}")


if __name__ == "__main__":
    evaluate()