
import re


CITATION_PATTERN = re.compile(
    r"\bSource\s+(\d+)\s*,\s*Page\s+(\d+)\b",
    re.IGNORECASE,
)


def build_sources(results: list[dict], answer: str) -> list[dict]:
    """Return only sources cited in the AI answer."""

    sources = []
    seen = set()

    for match in CITATION_PATTERN.finditer(answer):
        index = int(match.group(1)) - 1
        cited_page = int(match.group(2))

        if not (0 <= index < len(results)):
            continue

        result = results[index]
        metadata = result.get("metadata", {})
        actual_page = metadata.get("page")

        # Reject a citation if its page doesn't match metadata.
        if str(actual_page) != str(cited_page):
            continue

        source = metadata.get("source", "Unknown")
        key = (source, str(actual_page))

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "page": actual_page,
            "similarity": result.get("similarity"),
        })

    return sources
