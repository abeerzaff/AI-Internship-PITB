
import os

import gradio as gr
import requests

from app.core.config import settings

API_URL = settings.api_url.rstrip("/")

#API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

BACKEND_DOWN = "**Error:** cannot reach the backend. Is uvicorn running?"


def _error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    return f"**Error:** {detail}"


def _sources_md(sources: list[dict]) -> str:
    if not sources:
        return ""

    pages_by_file: dict[str, set] = {}

    for item in sources:
        name = item["source"].split("_", 1)[-1]
        pages_by_file.setdefault(name, set()).add(item["page"])

    lines = ["**Sources**"]

    for name, pages in pages_by_file.items():
        page_list = ", ".join(
            str(p) for p in sorted(pages, key=str)
        )
        lines.append(f"- {name} (page {page_list})")

    return "\n".join(lines)


def upload(file):
    if file is None:
        return "Please choose a PDF first.", None

    path = getattr(file, "name", file)

    try:
        with open(path, "rb") as handle:
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={
                    "file": (
                        os.path.basename(path),
                        handle,
                        "application/pdf",
                    )
                },
                timeout=600,
            )
    except requests.exceptions.RequestException:
        return BACKEND_DOWN, None

    if response.status_code != 200:
        return _error(response), None

    data = response.json()

    return (
        f"**{data['filename']}** was uploaded and indexed.\n\n"
        f"- Pages: {data['page_count']}\n"
        f"- Chunks: {data['chunk_count']}\n"
        f"- Text extraction: {data['extraction_method']}\n\n"
        "You can now ask questions or open the Summary tab."
    ), data["document_id"]


def ask(question, document_id):
    if not document_id:
        return "Please upload a PDF first.", ""

    if not question or len(question.strip()) < 3:
        return "Please type a question (at least 3 characters).", ""

    try:
        response = requests.post(
            f"{API_URL}/chat/ask",
            json={
                "question": question.strip(),
                "document_id": document_id,
            },
            timeout=180,
        )
    except requests.exceptions.RequestException:
        return BACKEND_DOWN, ""

    if response.status_code != 200:
        return _error(response), ""

    data = response.json()

    status = (
        "*Answer found in the document.*"
        if data["grounded"]
        else "*Not found in the document.*"
    )

    return (
        data["answer"],
        f"{status}\n\n{_sources_md(data['sources'])}",
    )


def analyze(kind: str, document_id: str | None):
    if not document_id:
        return "Please upload a PDF first.", ""

    try:
        response = requests.post(
            f"{API_URL}/documents/{kind}",
            params={"document_id": document_id},
            timeout=300,
        )
    except requests.exceptions.RequestException:
        return BACKEND_DOWN, ""

    if response.status_code != 200:
        return _error(response), ""

    data = response.json()

    if kind == "summary":
        text = data["summary"]
    else:
        text = "\n".join(
            f"{i}. {item}"
            for i, item in enumerate(data["items"], 1)
        )
        if not text:
            text = "No explicit items were identified."

    return text, _sources_md(data["sources"])


with gr.Blocks(title="Document Intelligence Assistant") as demo:

    # Automatically stores the uploaded PDF's document ID.
    document_id_state = gr.State(value=None)

    gr.Markdown(
        "# AI-Powered Document Intelligence Assistant\n"
        "Upload a PDF, ask questions about it, and generate a summary, "
        "key points and action items. Answers come only from your document."
    )

    with gr.Tab("1. Upload"):
        file_input = gr.File(
            label="PDF document",
            file_types=[".pdf"]
        )
        upload_button = gr.Button(
            "Upload and index",
            variant="primary"
        )
        upload_status = gr.Markdown()

        upload_button.click(
            upload,
            inputs=[file_input],
            outputs=[upload_status, document_id_state],
        )

    with gr.Tab("2. Ask a question"):
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the objective of this document?",
            lines=2,
        )

        ask_button = gr.Button("Ask", variant="primary")
        answer_box = gr.Markdown()
        answer_sources = gr.Markdown()

        ask_outputs = [answer_box, answer_sources]

        ask_button.click(
            ask,
            inputs=[question_box, document_id_state],
            outputs=ask_outputs,
        )

        question_box.submit(
            ask,
            inputs=[question_box, document_id_state],
            outputs=ask_outputs,
        )

    with gr.Tab("3. Summary and insights"):
        with gr.Row():
            summary_button = gr.Button(
                "Summary",
                variant="primary"
            )
            keypoints_button = gr.Button("Key points")
            actions_button = gr.Button("Action items")

        analysis_box = gr.Markdown()
        analysis_sources = gr.Markdown()
        analysis_outputs = [analysis_box, analysis_sources]

        summary_button.click(
            lambda doc_id: analyze("summary", doc_id),
            inputs=[document_id_state],
            outputs=analysis_outputs,
        )

        keypoints_button.click(
            lambda doc_id: analyze("key-points", doc_id),
            inputs=[document_id_state],
            outputs=analysis_outputs,
        )

        actions_button.click(
            lambda doc_id: analyze("action-items", doc_id),
            inputs=[document_id_state],
            outputs=analysis_outputs,
        )


if __name__ == "__main__":
    demo.launch()
