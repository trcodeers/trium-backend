from contextlib import asynccontextmanager
import json
from fastapi.middleware.cors import CORSMiddleware
from app.ai import cosine_similarity, embed_text, generate_answer
from app.db import get_db
from app.schemas import QueryRequest, QueryResponse


from app.db import get_db
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ai import embed_text
from app.db import init_db
from app.ingestion import ingest_item
from app.schemas import IngestRequest, IngestResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Database tables initialized successfully")
    yield


app = FastAPI(
    title="AI Knowledge Inbox",
    description="Save, search, and ask questions about your knowledge.",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/test-embedding")
def test_embedding(request: EmbeddingRequest):
    try:
        vector = embed_text(
            request.text,
            task_type="RETRIEVAL_DOCUMENT",
        )

        return {
            "success": True,
            "text": request.text,
            "dimensions": len(vector),
            "embedding_preview": vector[:5],
        }
    except Exception as exc:
        print(f"Embedding error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Embedding generation failed. Check the server logs.",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
       return ingest_item(
            source_type=request.source_type,
            content=request.content,
            url=request.url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Ingestion error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Ingestion failed. Check the server logs.",
        ) from exc





@app.get("/items")
def list_items():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, source_type, source_url, raw_content, created_at
            FROM items
            ORDER BY id DESC
            """
        ).fetchall()

    return {
        "items": [
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "content": row["raw_content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
        "total": len(rows),
    }





@app.post("/query", response_model=QueryResponse)
def query_knowledge(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        # 1. Embed the question for semantic retrieval.
        question_embedding = embed_text(
            question,
            task_type="RETRIEVAL_QUERY",
        )

        # 2. Load saved chunks and their parent item metadata.
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    chunks.item_id,
                    chunks.chunk_index,
                    chunks.content,
                    chunks.embedding,
                    items.source_url
                FROM chunks
                JOIN items ON items.id = chunks.item_id
                """
            ).fetchall()

        # 3. Rank chunks by cosine similarity.
        ranked_chunks = []

        for row in rows:
            stored_embedding = json.loads(row["embedding"])

            similarity = cosine_similarity(
                question_embedding,
                stored_embedding,
            )

            ranked_chunks.append({
                "item_id": row["item_id"],
                "source_url": row["source_url"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "similarity": similarity,
            })

        ranked_chunks.sort(
            key=lambda chunk: chunk["similarity"],
            reverse=True,
        )
        
        # Reject questions when no saved chunk is sufficiently relevant.
        MIN_SIMILARITY = 0.45

        if not ranked_chunks or ranked_chunks[0]["similarity"] < MIN_SIMILARITY:
            return {
                "answer": (
                    "I couldn't find sufficiently relevant information "
                    "in your saved knowledge to answer that question."
                ),
                "sources": [],
            }

        # 4. Keep the five most relevant chunks.
        selected = ranked_chunks[:5]

        # 5. Generate an answer using only the retrieved excerpts.
        contexts = [
            {
                "title": f"Item {chunk['item_id']}",
                "content": chunk["content"],
            }
            for chunk in selected
        ]

        answer = generate_answer(question, contexts)

        return {
            "answer": answer,
            "sources": [
                {
                    "item_id": chunk["item_id"],
                    "source_url": chunk["source_url"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "similarity": round(chunk["similarity"], 4),
                }
                for chunk in selected
            ],
        }

    except Exception as exc:
        print(f"Query error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Query failed. Check the server logs.",
        ) from exc






