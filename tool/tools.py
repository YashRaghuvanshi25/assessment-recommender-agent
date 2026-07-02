from langchain.tools import tool

from tool.recommendation_engine import RecommendationEngine

# Initialize once and reuse across tool calls
_engine = RecommendationEngine()


@tool
def recommend_assessments(query: str) -> dict:
    """
    Recommend the most suitable SHL assessments for a hiring request.

    Args:
        query: Natural language hiring requirement.

    Returns:
        {
            "query": "original standalone hiring request",
            "recommendations": [
                {
                    "id": "...",
                    "name": "...",
                    "link": "...",
                    "score": 0.91,
                    "test_type": "...",
                    "duration": "...",
                    "languages": "..."
                },
                ...
            ],
            "coverage": {
                "<retrieval phrase>": "<assessment name>"
            }
        }
    """

    _, recommendations, coverage = _engine.recommend(query)

    results = []

    for assessment in recommendations:
        results.append(
            {
                "id": assessment["entity_id"],
                "name": assessment["name"],
                "link": assessment["link"],
                "score": round(assessment["similarity_score"], 4),
                "keys": ", ".join(assessment.get("keys", [])) if assessment.get("keys") else None,
                "duration": assessment.get("duration", None),
                "languages": ", ".join(assessment.get("languages", [])) if assessment.get("languages") else None,
            }
        )

    return {
        "query": query,
        "recommendations": results,
        "coverage": coverage,
    }


TOOLS = [
    recommend_assessments,
]