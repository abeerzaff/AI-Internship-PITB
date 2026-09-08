
import time
from google import genai
from google.genai import errors as genai_errors

import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

MAX_RETRIES = 3
BACKOFF_SECONDS = 2


def _call_gemini(prompt: str) -> str:
    """Call Gemini with retry/backoff for rate-limit or transient errors."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except genai_errors.APIError as e:
            last_error = e
            # Retry only on rate-limit (429) or service-unavailable (503)
            if getattr(e, "code", None) in (429, 503):
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} retries: {last_error}")


def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize arbitrary input text (no retrieval involved)."""
    prompt = (
        f"Summarize the following text in no more than {max_length} words. "
        f"Be concise and preserve the key points:\n\n{text}"
    )
    return _call_gemini(prompt)


def answer_question(question: str, context: str) -> dict:
  
    if not context.strip():
        prompt = (
            "No documents have been uploaded, so there is no document context "
            "available. Answer the question below using your own general "
            "knowledge.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        prompt = (
            "You are answering a question. Context retrieved from an uploaded "
            "document is provided below.\n\n"
            "- If the context contains information that answers the question, "
            "use it, and start your reply with the exact tag [SOURCE: DOCUMENT].\n"
            "- If the context does NOT contain relevant information for this "
            "question, ignore the context and answer using your own general "
            "knowledge instead, starting your reply with the exact tag "
            "[SOURCE: GENERAL_KNOWLEDGE].\n"
            "Always include exactly one of those two tags at the very start, "
            "then a newline, then your answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

    raw = _call_gemini(prompt)
    return _parse_sourced_answer(raw, has_context=bool(context.strip()))


def _parse_sourced_answer(raw: str, has_context: bool) -> dict:
    """Split the [SOURCE: ...] tag from the model's reply, if present."""
    if not has_context:
        return {"answer": raw.strip(), "answer_source": "general_knowledge"}

    text = raw.strip()
    if text.startswith("[SOURCE: DOCUMENT]"):
        return {"answer": text[len("[SOURCE: DOCUMENT]"):].strip(), "answer_source": "document"}
    if text.startswith("[SOURCE: GENERAL_KNOWLEDGE]"):
        return {"answer": text[len("[SOURCE: GENERAL_KNOWLEDGE]"):].strip(), "answer_source": "general_knowledge"}

    return {"answer": text, "answer_source": "document"}
