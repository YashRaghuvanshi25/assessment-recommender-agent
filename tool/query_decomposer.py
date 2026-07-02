import json
import os
import re
import requests

from dotenv import load_dotenv

load_dotenv()


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")

    if base_url.endswith("/chat/completions"):
        return base_url

    return f"{base_url}/chat/completions"


def _supports_response_format(model: str) -> bool:
    return not model.endswith(":free") and model != "openrouter/free"


def _parse_json_content(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

SYSTEM_PROMPT = """
You are a semantic query decomposition assistant.

Your task is to extract the smallest meaningful semantic retrieval phrases from a hiring request while preserving the original skills, technologies, and competencies.

Rules:
- Return JSON only.
- Keep each retrieval query short and focused.
- Group closely related technologies and skills into a single retrieval query.
- Preserve the original technologies, skills, and domains whenever possible.
- Do not replace technologies with broader concepts (e.g. "Backend Engineering").
- Do not invent or infer technologies, skills, or experience that are not explicitly mentioned.
- Do not repeat the same context across multiple queries unless it materially changes retrieval.
- Generate one or more retrieval queries whenever multiple semantic retrieval intents would improve retrieval quality.
- Retrieval queries may represent roles, competencies, assessment categories, technologies, domains, seniority, or hiring objectives.
- Do not generate multiple queries that are simple rephrasings of one another.
- Each retrieval query should capture one distinct semantic retrieval intent.
- Even if the request describes a single hiring need, generate multiple retrieval queries whenever different semantic facets (role, competency, assessment category, seniority, hiring objective, etc.) would retrieve different relevant assessments.
Example:
User:
We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.

Output:
{
  "retrieval_queries": [
    "Graduate management trainee",
    "Cognitive ability",
    "Personality",
    "Situational judgement"
  ]
}
"""


class QueryDecomposer:
    def __init__(self, model: str | None = None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")
        self.model = model or os.getenv("OPENROUTER_MODEL")

        missing = [
            name
            for name, value in {
                "OPENROUTER_API_KEY": self.api_key,
                "OPENROUTER_BASE_URL": self.base_url,
                "OPENROUTER_MODEL": self.model,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

        self.chat_completions_url = _chat_completions_url(self.base_url)

    def decompose(self, query: str):
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        }
        if _supports_response_format(self.model):
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "SHL Recommender",
        }

        try:
            response = requests.post(
                self.chat_completions_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"OpenRouter request failed: {error}") from error

        if not response.ok:
            model_hint = f" Check OPENROUTER_MODEL; currently set to '{self.model}'."
            raise RuntimeError(
                f"OpenRouter request failed ({response.status_code}): {response.text}{model_hint}"
            )

        data = response.json()

        message = data["choices"][0].get("message", {})
        content = message.get("content")

        if not content:
            raise RuntimeError(
                "OpenRouter returned an empty message.content. "
                f"Model: '{self.model}'. Response: {json.dumps(data)}"
            )

        try:
            return _parse_json_content(content)
        except Exception:
            print("[WARN] Failed to parse decomposer output:")
            print(content)
            return {"retrieval_queries": [query]}


if __name__ == "__main__":
    decomposer = QueryDecomposer()

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        result = decomposer.decompose(query)
        print(json.dumps(result, indent=2))
