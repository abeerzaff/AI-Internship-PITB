from google import genai

from config import (
    GEMINI_API_KEY,
    MODEL_NAME
)


def generate_ai_insights(data_summary):

    if not GEMINI_API_KEY:

        return (
            "⚠️ AI insights are unavailable. "
            "Please configure GEMINI_API_KEY "
            "in the .env file."
        )

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a professional data analyst.

Analyze the following dataset information.

{data_summary}

Provide the response using these sections:

## Overall Summary

Give a short summary of the dataset.

## Key Findings

Provide 3 to 5 important findings.

## Trends and Patterns

Explain important trends or relationships.

## Data Quality Concerns

Mention missing values, duplicates, or other
data-quality concerns if present.

## Recommendations

Provide 3 practical recommendations
that could help with decision-making.

Important:
- Do not invent information.
- Only use values and patterns present
  in the supplied dataset information.
- Keep the language clear and professional.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            f"⚠️ Unable to generate AI insights: {e}"
        )