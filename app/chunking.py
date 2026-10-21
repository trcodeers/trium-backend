
def chunk_text(
    text: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[str]:
    """Split text into overlapping chunks using character counts."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split()).strip()

    if not normalized:
        return []

    chunks = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])

        if end == len(normalized):
            break

        start = end - overlap

    return chunks
