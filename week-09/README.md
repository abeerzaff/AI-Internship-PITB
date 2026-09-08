# AI Assistant Backend API

Assignment 9 — AI Product Development and Final Delivery (AI Backend and API Integration)

A FastAPI backend that exposes AI capabilities — text summarization and
RAG-based question answering over uploaded PDF documents — as REST endpoints.
Built on top of the RAG pipeline from Assignment 8 (ChromaDB + Gemini).

## Features

Home endpoint — confirms the API is live
Health check endpoint — for monitoring/testing
Text summarization endpoint— summarizes any input text via Gemini
Question-answering endpoint— RAG-based answers grounded in uploaded documents
Document upload endpoint— ingests PDFs (extract → chunk → embed → store in ChromaDB), with automatic OCR fallback for scanned/image-based PDFs


## Setup

1. Clone the repo and enter the project folder
   git clone <your-repo-url>
   cd week-09
   

2. create and activate a virtual environment**

   python3 -m venv .venv
   source .venv/bin/activate      # Mac/Linux
   .venv\Scripts\activate         # Windows
  

3. Install dependencies
   pip install -r requirements.txt
   

4. Install dependencies**
  
   pip install -r requirements.txt
 
   OCR runs on pymupdf (renders PDF pages) and easyocr (reads the text out of them), 

5. Set up your .env file
  
   cp  .env
  
   Then open .env and add your real Gemini API key:

   GEMINI_API_KEY=your_actual_key_here



## Running the app

uvicorn main:app --reload


The API will be available at `http://127.0.0.1:8000`.
Interactive docs (Swagger UI) are auto-generated at `http://127.0.0.1:8000/docs`
— use this for both testing and as part of your API documentation screenshots.

## Sample Requests

Health check
GET http://127.0.0.1:8000/health

Summarize text
 POST http://127.0.0.1:8000/summarize \

Upload a document
 POST http://127.0.0.1:8000/upload \
 
Question answering endpoint
POST http://127.0.0.1:8000/ask \
 

## /ask: Document Answers vs. General Knowledge

1. `/ask` no longer refuses when a question isn't covered by the uploaded
document(s), or when no document has been uploaded at all. Instead if relevant chunks exist for the question → answers from the document,answer_source: document, sources lists the file(s) used.If no document was uploaded, or the uploaded document doesn't cover the question (e.g. asking "What is machine learning?" against an unrelated PDF) , answers from the LLM's own general knowledge, answer_source: general_knowledge, sources is empty.

## Expected Output

1. /summarize= {"summary": "...", "status": "success"}
2. /upload={"filename": "sample.pdf", "chunks_stored": 12, "status": "success"}
3. /ask={"answer": "...", "sources": [sample.pdf], "status": "success"}

## Scanned/Image-Based PDFs (OCR)

The /upload endpoint can also handle scanned PDFs automatically. 
Here is how it works:

1. First, the system tries to extract text from the PDF using pypdf. This is fast and works well for normal text-based PDFs.
2. If the PDF has very little or no text, the system assumes it may be a scanned or image-based PDF. It then uses OCR.
3. For OCR, pymupdf converts each PDF page into an image. 
4. Then easyocr  reads the text from those images. 
5. The /upload response tells you which method was used:
   .extraction_method: native= normal PDF text extraction was used.
   .extraction_method: ocr= OCR was used for a scanned PDF.
6. A 422 error is returned only when the system cannot find any text using either method. For example, the PDF may be completely blank or corrupted.

## Testing the API with Postman
You can test all the API endpoints using Postman.

### 1. Set the Base URL

Use:
http://127.0.0.1:8000
You can save this as your Postman environment base URL.

### 2. Test the Basic Endpoints
First test:
GET /
GET /health

These should return a successful response. This confirms that your FastAPI server is running correctly.

### 3. Test /summarize and /ask

For both endpoints:
Method: POST
Go to Body
Select raw
Select JSON
Enter the required JSON data

### 4. Test /upload with a Normal PDF

For a normal text-based PDF:

Method: POST
Go to Body
Select form-data
Add a key named file
Change its type from Textto File
Select a normal PDF file

The response should contain:
extraction_method: "native"
This means the system successfully extracted the text directly from the PDF.

### 5. Test /upload with a Scanned PDF

Now test the same endpoint with a scanned PDF or a PDF created from photos of pages.
Use the same settings:

The response should contain:
extraction_method: "ocr"`
This proves that the OCR system successfully read the scanned PDF and stored the extracted text in the vector database.


#### Case 1 — Ask Before Uploading a Document
Ask a general question, for example:
What is machine learning?
The expected response should contain:
answer_source: "general_knowledge"

#### Case 2 — Ask a Question About the Uploaded Document

Then ask a question that is actually answered by the document.

The expected response should contain:
"answer_source": "document"

#### Case 3 — Ask an Unrelated Question

For example, if the uploaded PDF is about a completely different topic, ask:
What is machine learning?
The expected response should again contain:
answer_source: "general_knowledge"




