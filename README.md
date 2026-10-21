# AI Knowledge Inbox — Backend

A FastAPI backend for saving notes and web pages, generating embeddings, and answering questions using semantic search and retrieval augmented generation (RAG).

## Tech Stack

- Python 3.12
- FastAPI
- SQLite
- Google Gemini API
- `google-genai`
- HTTPX and BeautifulSoup
- `uv` for dependency management

## Features

- Save plain-text notes.
- Ingest readable content from URLs.
- Split content into overlapping chunks.
- Generate embeddings using Gemini.
- Store items, chunks, and embeddings in SQLite.
- Search saved knowledge using cosine similarity.
- Generate answers grounded in retrieved content.
- Return source excerpts and similarity scores.
- Validate incoming requests and handle API errors.

## Prerequisites

- Python 3.12 or compatible version
- `uv`
- A Google Gemini API key

## Setup

Clone or open the backend repository and navigate to its root directory.

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
DATABASE_PATH=./knowledge.db
```

Install dependencies:

```bash
uv sync
```

If the project does not have a `pyproject.toml` with dependencies configured, install the required packages:

```bash
uv add fastapi "uvicorn[standard]" google-genai python-dotenv httpx beautifulsoup4
```

## Run the Server

From the backend repository root:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## API Endpoints

### `GET /health`

Checks whether the API is running.

### `POST /ingest`

Saves a note or ingests content from a URL.

Example note request:

```json
{
  "source_type": "note",
  "content": "TCP is a connection-oriented transport layer protocol."
}
```

Example URL request:

```json
{
  "source_type": "url",
  "url": "https://example.com"
}
```

### `GET /items`

Returns the saved knowledge items.

### `POST /query`

Searches saved knowledge and generates an answer from relevant excerpts.

Example request:

```json
{
  "question": "What does TCP do?"
}
```

The response includes an answer and source excerpts with similarity scores. If no sufficiently relevant chunks are found, the API should return an appropriate fallback answer with an empty source list.

## How It Works

1. The user submits a note or URL.
2. For URLs, the backend fetches and extracts readable page content.
3. The content is normalized and split into overlapping chunks.
4. Gemini generates an embedding for each chunk.
5. The item, chunks, and embeddings are saved in SQLite.
6. When a question arrives, the backend embeds the question.
7. It compares the question embedding with stored chunk embeddings using cosine similarity.
8. Relevant chunks are selected and passed to Gemini as context.
9. Gemini generates a grounded answer, and the API returns the answer and sources.

## Data Storage

SQLite stores the application's data locally.

- `items`: source type, URL, raw content, and creation time.
- `chunks`: chunk content, parent item ID, chunk index, and embedding.

The `knowledge.db` file is created locally when the database is initialized.

## Configuration and Security

- Keep your Gemini API key in `.env`.
- Never commit `.env` or expose the API key in frontend code.
- Keep the local database out of version control.
- URL ingestion should be hardened against server-side request forgery (SSRF) before public deployment. Validate destinations and revalidate redirect targets.
- Similarity thresholds are heuristic values. Test relevant and unrelated queries before choosing a production threshold.
- Configure CORS appropriately for the frontend origin.

## Troubleshooting

**Missing API key:** Check that `.env` exists in the backend root and contains a valid `GEMINI_API_KEY`.

**Import errors:** Confirm the virtual environment and dependencies are set up correctly.

**URL ingestion fails:** The website may block automated requests, require JavaScript, or return unsupported content. Redirects and network timeouts can also cause failures.

**No relevant sources:** Confirm that content was ingested successfully and adjust the retrieval threshold based on test queries.

**Database issues:** Stop the server before resetting the local database. Resetting it deletes saved knowledge.

## Current Scope

This is a lightweight prototype intended to demonstrate document ingestion, embedding-based retrieval, and grounded question answering. It uses local SQLite storage and a straightforward similarity-search approach rather than a dedicated vector database.