
def build_context(
    results: list[dict],
) -> str:
    """Build grounded context from retrieved chunks."""

    if not results:
        return ""

    sections = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        text = result["text"]

        section = (
            f"SOURCE {index}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

        sections.append(section)

    return "\n\n".join(sections)