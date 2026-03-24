import base64
from tempfile import NamedTemporaryFile
from datauri import DataURI
from typing import Sequence
from openai.types import (
    EmbeddingCreateParams,
    CreateEmbeddingResponse,
    Embedding,
)
from openai.types.create_embedding_response import Usage, CreateEmbeddingResponse
from fastapi import APIRouter
from src.embeddings import service as embedding_service

router = APIRouter()


@router.post("/embeddings")
async def embeddings(params: EmbeddingCreateParams) -> CreateEmbeddingResponse:
    if isinstance(params["input"], str):
        uri = DataURI(params["input"])
        if uri.mimetype != "audio/mpeg":
            raise Exception("Not supported!")

        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            temp_file.write(uri.data)
            temp_file.flush()

            result = embedding_service.embedding_audio(temp_file.name)
            return CreateEmbeddingResponse(
                data=[
                    Embedding(
                        embedding=result,
                        index=0,
                        object="embedding",
                    )
                ],
                model="marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
                object="list",
                usage=Usage(
                    prompt_tokens=0,
                    total_tokens=0,
                ),
            )

    if isinstance(params["input"], Sequence) and not isinstance(params["input"], str):
        raise Exception("Not implemented yet")

    raise Exception("Not implemented yet")
