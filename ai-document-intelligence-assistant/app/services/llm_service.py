
import logging
import time

from google import genai
from google.genai import errors

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None

MAX_ATTEMPTS = 3


class LLMUnavailableError(RuntimeError):
    """Gemini stayed unavailable after retries."""


class LLMQuotaExceededError(RuntimeError):
    """Gemini API quota or rate limit has been exceeded."""


def get_gemini_client():
    global _client

    if _client is None:
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        _client = genai.Client(
            api_key=settings.gemini_api_key
        )

        logger.info("Gemini client initialized.")

    return _client


def generate_answer(prompt: str) -> str:

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = get_gemini_client()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=settings.model_name,
                contents=prompt,
            )
            break

        # Temporary Gemini/server problem
        # Retry this.
        except errors.ServerError as exc:
            logger.warning(
                "Gemini unavailable (attempt %d/%d): %s",
                attempt,
                MAX_ATTEMPTS,
                exc,
            )

            if attempt == MAX_ATTEMPTS:
                raise LLMUnavailableError(
                    str(exc)
                ) from exc

            time.sleep(2 * attempt)

        # Client-side API problem such as quota/rate limit
        except errors.ClientError as exc:

            if exc.code == 429:
                logger.warning(
                    "Gemini API quota exceeded: %s",
                    exc,
                )

                raise LLMQuotaExceededError(
                    "Gemini API quota has been exceeded."
                ) from exc

            # Other 4xx errors should not be hidden
            raise

    answer = (response.text or "").strip()

    if not answer:
        raise ValueError(
            "The AI service returned an empty response."
        )

    logger.info("Gemini response generated.")

    return answer

