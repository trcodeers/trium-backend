
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    source_type: Literal["note", "url"]
    content: str | None = Field(default=None, max_length=100000)
    url: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type == "note":
            if not self.content or not self.content.strip():
                raise ValueError("Notes require non-empty content.")

        if self.source_type == "url":
            if not self.url or not self.url.strip():
                raise ValueError("URLs require a URL.")

        return self


class IngestResponse(BaseModel):
    id: int
    source_type: str
    chunks_created: int
    message: str
