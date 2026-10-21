
from contextlib import asynccontextmanager

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
