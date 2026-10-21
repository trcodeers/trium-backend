
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.ai import embed_text
from app.chunking import chunk_text
from app.db import get_db


def fetch_url(url: str) -> tuple[str, str]:
    """Fetch a public webpage and extract readable text."""

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provide a valid HTTP or HTTPS URL.")

    # Basic protection against localhost and private-network targets.
    host = parsed.hostname.lower()

    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host == "metadata.google.internal"
    ):
        raise ValueError("Local and internal URLs are not allowed.")

    try:
        with httpx.Client(
            timeout=15,
            follow_redirects=False,
            headers={"User-Agent": "AI-Knowledge-Inbox/1.0"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError("Only HTML and plain-text URLs are supported.")

    if len(response.content) > 2_000_000:
        raise ValueError("The webpage is too large. Maximum size is 2 MB.")

    if "text/plain" in content_type:
        text = response.text
        title = parsed.hostname
    else:
        soup = BeautifulSoup(response.text, "html.parser")

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else parsed.hostname
        )

        for element in soup(
            ["script", "style", "nav", "footer", "header", "noscript"]
        ):
            element.decompose()

        text = soup.get_text(" ", strip=True)

    if not text.strip():
        raise ValueError("No readable text was found at this URL.")

    return title[:300], text[:100000]



def ingest_item(
    source_type: str,
    content: str | None,
    url: str | None,
) -> dict:
    """Extract, chunk, embed, and save an item."""

    if source_type == "url":
        if not url:
            raise ValueError("A URL is required.")

        _, extracted_content = fetch_url(url)
        source_url = url

    else:
        if not content or not content.strip():
            raise ValueError("Note content is required.")

        extracted_content = content.strip()
        source_url = None

    chunks = chunk_text(extracted_content)

    if not chunks:
        raise ValueError("No text could be processed.")

    # Generate embeddings before opening the database transaction.
    embedded_chunks = [
        (index, chunk, embed_text(chunk, "RETRIEVAL_DOCUMENT"))
        for index, chunk in enumerate(chunks)
    ]

    import json

    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO items (source_type, source_url, raw_content)
            VALUES (?, ?, ?)
            """,
            (source_type, source_url, extracted_content),
        )

        item_id = cursor.lastrowid

        conn.executemany(
            """
            INSERT INTO chunks (item_id, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?)
            """,
            [
                (item_id, index, chunk, json.dumps(vector))
                for index, chunk, vector in embedded_chunks
            ],
        )

    return {
        "id": item_id,
        "source_type": source_type,
        "chunks_created": len(chunks),
        "message": "Item ingested successfully.",
    }
