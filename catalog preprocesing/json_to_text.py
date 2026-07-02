import json
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "shl_product_catalog.json"
OUTPUT_FILE = Path(__file__).parent / "catalog_texts.json"

# Fields that don't help semantic retrieval
IGNORE_FIELDS = {
    "entity_id",
    "scraped_at",
    "status",
    "job_levels_raw",
    "languages_raw",
    "duration_raw",
    "link",
}

LABELS = {
    "name": "Assessment Name",
    "description": "Description",
    "job_levels": "Suitable Job Levels",
    "languages": "Available Languages",
    "duration": "Duration",
    "remote": "Remote Testing",
    "adaptive": "Adaptive Testing",
    "keys": "Assessment Categories",
}


def format_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, list):
        if not value:
            return None
        return "\n".join(str(v) for v in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)

    return str(value)


def assessment_to_text(assessment):
    parts = []

    for key, value in assessment.items():

        if key in IGNORE_FIELDS:
            continue

        value = format_value(value)

        if value is None:
            continue

        label = LABELS.get(key, key.replace("_", " ").title())


        if key in {"remote", "adaptive"}:
            value = value.title()

        parts.append(f"{label}:\n{value}")

    return "\n\n".join(parts)


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    output = []

    for assessment in catalog:

        output.append(
            {
                "id": assessment.get("entity_id"),
                "name": assessment.get("name"),
                "text": assessment_to_text(assessment),
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(output)} assessments.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()