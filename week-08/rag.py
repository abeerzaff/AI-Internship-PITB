import os
import re
import time
from pathlib import Path

import gradio as gr
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Make sure your .env file exists."
    )

# Gemini model that you already tested successfully
GEMINI_MODEL = "gemini-3.7-flash"

# Embedding model from the assignment
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks retrieved for each question
TOP_K = 8

# Retry settings for transient Gemini errors (e.g. 503 UNAVAILABLE)
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2


# ============================================================
# 2. INITIALIZE MODELS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# 3. INITIALIZE CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

COLLECTION_NAME = "pdf_rag_collection"

collection = None

current_pdf_name = None

# Opening chunk of the currently loaded PDF, used to ground
# document-level / meta questions (e.g. "what is this file about?")
doc_overview_chunk = None


# ============================================================
# 4. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 6. TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=800,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 7. PROCESS PDF
# ============================================================

def process_pdf(pdf_path):
    """
    Processes a PDF into chunks, embeddings, and a fresh ChromaDB
    collection. Returns (status_message, success_bool) so the chat
    layer can decide what to show and whether questions can proceed.
    """

    global collection
    global current_pdf_name
    global doc_overview_chunk

    if pdf_path is None:
        return "❌ Please upload a PDF first.", False

    try:

        current_pdf_name = Path(pdf_path).name

        # ----------------------------------------------------
        # Extract pages
        # ----------------------------------------------------

        pages = extract_pdf_pages(pdf_path)

        if not pages:
            return (
                "❌ Could not extract text from this PDF. "
                "The PDF may be scanned/image-based.",
                False
            )

        # ----------------------------------------------------
        # Clean pages
        # ----------------------------------------------------

        for page in pages:
            page["text"] = clean_text(page["text"])

        # ----------------------------------------------------
        # Create page-aware chunks
        # ----------------------------------------------------

        chunks = []

        for page in pages:

            page_chunks = chunk_text(page["text"])

            for chunk_index, chunk in enumerate(page_chunks):

                chunks.append({
                    "chunk_id": f"page_{page['page']}_chunk_{chunk_index}",
                    "source": current_pdf_name,
                    "page": page["page"],
                    "text": chunk
                })

        if not chunks:
            return "❌ No usable text chunks were created.", False

        # Save the opening chunk so meta/document-level questions
        # (e.g. "what is this document about?") have grounding even
        # if semantic retrieval doesn't surface the intro chunk.
        doc_overview_chunk = chunks[0]

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = embedding_model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # ----------------------------------------------------
        # Reset previous collection
        # ----------------------------------------------------

        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # ----------------------------------------------------
        # Create new collection
        # ----------------------------------------------------

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # ----------------------------------------------------
        # Prepare metadata
        # ----------------------------------------------------

        ids = [chunk["chunk_id"] for chunk in chunks]

        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        # ----------------------------------------------------
        # Store in ChromaDB
        # ----------------------------------------------------

        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        status_message = (
            f"✅ **{current_pdf_name}** processed successfully — "
            f"you can ask questions now.\n\n"
            f"*{len(pages)} pages extracted · {len(chunks)} chunks stored*"
        )

        return status_message, True

    except Exception as e:
        return f"❌ Error while processing PDF:\n\n`{str(e)}`", False


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(query, top_k=TOP_K):

    if collection is None:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    if not results["documents"]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):
        retrieved.append({
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        })

    return retrieved


# ============================================================
# 9. BUILD DOCUMENT CONTEXT
# ============================================================

def build_context(results, include_overview=True):

    context_parts = []

    # Always ground the model with the document's opening section,
    # so broad/meta questions have something to reason from even if
    # semantic search doesn't retrieve the intro chunk on its own.
    if include_overview and doc_overview_chunk is not None:
        context_parts.append(f"""
DOCUMENT OVERVIEW (opening section of the file)

File: {doc_overview_chunk["source"]}

Page: {doc_overview_chunk["page"]}

Content:
{doc_overview_chunk["text"]}
""")

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        context_parts.append(f"""
SOURCE {i}

File: {metadata["source"]}

Page: {metadata["page"]}

Content:
{result["text"]}
""")

    return "\n\n".join(context_parts)


# ============================================================
# 10. GENERATE GEMINI ANSWER
# ============================================================

def generate_answer(question, context):

    system_instruction = """
You are a PDF question-answering assistant.

Answer the user's question ONLY using the document
context provided to you.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make assumptions.
4. If the answer is not supported by the document
   context, say:

"I could not find this information in the uploaded document."

5. Keep the answer clear and concise.
6. Use only information from the uploaded document.
"""

    prompt = f"""
{system_instruction}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            return response.text

        except genai_errors.ServerError as e:

            # Covers transient issues like 503 UNAVAILABLE / overloaded model.
            last_error = e

            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_time)

            continue

    raise RuntimeError(
        f"Gemini did not respond after {MAX_RETRIES} attempts "
        f"(last error: {last_error})"
    )


# ============================================================
# 11. MAIN RAG FUNCTION
# ============================================================

def answer_question(question):

    if collection is None:
        return "❌ Please upload a PDF first.", ""

    if not question or not question.strip():
        return "❌ Please enter a question.", ""

    try:

        results = retrieve_chunks(question, TOP_K)

        if not results:
            return "I could not find this information in the uploaded document.", ""

        context = build_context(results)

        answer = generate_answer(question, context)

        source_lines = []
        seen_sources = set()

        for result in results:

            metadata = result["metadata"]
            source_key = (metadata["source"], metadata["page"])

            if source_key not in seen_sources:
                source_lines.append(f"- **{metadata['source']}** — Page {metadata['page']}")
                seen_sources.add(source_key)

        sources = "\n".join(source_lines)

        return answer, sources

    except Exception as e:
        return f"❌ Error generating answer:\n\n`{str(e)}`", ""


# ============================================================
# 12. CHAT ORCHESTRATION
# ============================================================

def respond(message, history):
    """
    Handles a single turn from the combined upload + ask input bar.
    `message` is a dict: {"text": str, "files": [paths]} from
    gr.MultimodalTextbox. Appends user + assistant turns to history
    in Gradio's messages format.
    """

    history = history or []

    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    # ------------------------------------------------------------
    # Step 1: process an uploaded PDF, if one came with this turn
    # ------------------------------------------------------------

    if files:

        pdf_path = files[0]

        history.append({
            "role": "user",
            "content": f"📄 Uploaded: **{Path(pdf_path).name}**"
        })

        status_message, success = process_pdf(pdf_path)

        history.append({
            "role": "assistant",
            "content": status_message
        })

        if not success:
            return history, gr.MultimodalTextbox(value=None, interactive=True)

    # ------------------------------------------------------------
    # Step 2: answer a question, if text was provided this turn
    # ------------------------------------------------------------

    if text:

        history.append({"role": "user", "content": text})

        if collection is None:
            history.append({
                "role": "assistant",
                "content": "⚠️ Please upload a PDF first — I need a document to search before I can answer."
            })
        else:
            answer, sources = answer_question(text)

            reply = answer

            if sources:
                reply += f"\n\n---\n📚 **Sources**\n{sources}"

            history.append({"role": "assistant", "content": reply})

    return history, gr.MultimodalTextbox(value=None, interactive=True)


def new_chat():
    """Resets the visible chat. Does not delete the ChromaDB collection,
    so the user can keep asking questions about the same PDF in a fresh
    conversation view."""

    return [], gr.MultimodalTextbox(value=None, interactive=True)


# ============================================================
# 13. GRADIO CHAT INTERFACE
# ============================================================

CUSTOM_CSS = """
:root, .dark {
    --body-background-fill: #0f0f11 !important;
    --background-fill-primary: #0f0f11 !important;
    --background-fill-secondary: #17171a !important;
    --border-color-primary: #2a2a2e !important;
}

.gradio-container {
    background: #0f0f11 !important;
    max-width: 860px !important;
    margin: auto !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 18px 0 6px;
}

.app-title {
    font-size: 26px;
    font-weight: 700;
    color: #ececf1;
}

.app-subtitle {
    font-size: 14px;
    color: #9a9aa2;
    margin-top: 4px;
}

/* Chat window */
#chatbot {
    background: #0f0f11 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 16px !important;
    min-height: 520px;
}

/* Assistant messages: plain text on dark background, like ChatGPT */
.message.bot, .message-wrap .bot, [data-testid="bot"] {
    background: transparent !important;
    color: #ececf1 !important;
}

/* User messages: rounded bubble, right aligned */
.message.user, .message-wrap .user, [data-testid="user"] {
    background: #2a2b32 !important;
    color: #ececf1 !important;
    border-radius: 18px !important;
}

/* Bottom input bar */
#chat-input textarea {
    background: #17171a !important;
    color: #ececf1 !important;
    border: 1px solid #33343a !important;
    border-radius: 22px !important;
    font-size: 15px !important;
}

#chat-input {
    border-radius: 22px !important;
}

/* New chat button */
.new-chat-btn {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

footer { display: none !important; }
"""

with gr.Blocks(
    title="Professional RAG PDF Assistant"
) as demo:

    gr.HTML("""
        <div class="app-header">
            <div class="app-title">🤖 Professional RAG PDF Assistant</div>
            <div class="app-subtitle">Upload a PDF, then ask anything about it — grounded, source-cited answers only.</div>
        </div>
    """)

    chatbot = gr.Chatbot(
        value=[{
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your PDF assistant.\n\n"
                "Attach a PDF below using the 📎 icon, then ask me anything "
                "about it. I'll only answer from what's actually in the "
                "document, and I'll always show you the source pages."
            )
        }],
        elem_id="chatbot",
        avatar_images=(None, "🤖"),
        show_label=False
    )

    msg = gr.MultimodalTextbox(
        interactive=True,
        file_types=[".pdf"],
        placeholder="Attach a PDF and/or ask a question...",
        show_label=False,
        elem_id="chat-input",
        sources=["upload"]
    )

    new_chat_btn = gr.Button("🗑️ New Chat", elem_classes="new-chat-btn", size="sm")

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    new_chat_btn.click(
        fn=new_chat,
        inputs=None,
        outputs=[chatbot, msg]
    )


# ============================================================
# 14. LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="gray")
    )