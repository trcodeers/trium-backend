
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs automatically when the application starts
    init_db()
    print("Database tables initialized successfully")

    yield

    # Runs when the application shuts down


app = FastAPI(
    title="AI Knowledge Inbox",
    description="Save, search, and ask questions about your knowledge.",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
