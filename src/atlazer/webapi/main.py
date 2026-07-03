from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from atlazer.celery_app.tasks.webapi import generate_embeddings
from atlazer.utils.embedder import get_embedder, BaseEmbedder, chunks_to_vector
from atlazer.webapi.schemas import (
    EmbedChunksRequest,
    EmbedChunksResponse,
    HealthResponse,
)

embedder_service: Optional[BaseEmbedder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder_service
    # Initialize and cache the embedder on startup
    embedder_service = get_embedder()
    yield


app = FastAPI(
    title="Chunk Embedding Service",
    description="Microservice to embed chunks of text.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if embedder_service is not None else "loading",
        model=embedder_service.model_name if embedder_service else "unknown",
    )


@app.post("/embed", response_model=EmbedChunksResponse)
def embed(payload: EmbedChunksRequest):
    if embedder_service is None:
        raise HTTPException(status_code=503, detail="Service is not ready yet.")
    
    try:
        embedded_chunks = chunks_to_vector(payload.chunks)
        return EmbedChunksResponse(chunks=embedded_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-parallel", response_model=EmbedChunksResponse)
def embed_parallel(payload: EmbedChunksRequest):
    if embedder_service is None:
        raise HTTPException(status_code=503, detail="Service is not ready yet.")
    
    try:
        #generate embeddings in parallel using Celery
        job = generate_embeddings.delay(
            kwargs={"chunks": payload.chunks},
            queue="webapi",
        )
        return {"job_id": job.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))