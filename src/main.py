from fastapi import FastAPI

from src.embeddings import router as embedding_router

app = FastAPI()

app.include_router(embedding_router.router)
