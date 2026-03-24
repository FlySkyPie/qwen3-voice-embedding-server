from fastapi import FastAPI

from src.embeddings import router as router_service

app = FastAPI()

app.include_router(router_service)
