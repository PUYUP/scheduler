import structlog
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request

from atlazer.celery_app.tasks.webapi import generate_embeddings
from atlazer.celery_app.tasks.matcher import single_user
from atlazer.utils.embedder import get_embedder, BaseEmbedder, chunks_to_vector
from atlazer.webapi.schemas import (
    EmbedChunksRequest,
    EmbedChunksResponse,
    EmbedParallelResponse,
    HealthResponse,
    PaperMatcherRequest,
)

log = structlog.get_logger(__name__)
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
        log.info("webapi.embed.start", chunks_count=len(payload.chunks))
        embedded_chunks = chunks_to_vector(payload.chunks)
        log.info("webapi.embed.success", chunks_count=len(embedded_chunks))
        return EmbedChunksResponse(chunks=embedded_chunks)
    except Exception as e:
        log.error("webapi.embed.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-parallel", response_model=EmbedParallelResponse)
def embed_parallel(payload: EmbedChunksRequest):
    if embedder_service is None:
        raise HTTPException(status_code=503, detail="Service is not ready yet.")
    
    try:
        #generate embeddings in parallel using Celery
        log.info("webapi.embed-parallel.start", chunks_count=len(payload.chunks))
        job = generate_embeddings.apply_async(
            kwargs={"chunks": payload.chunks, "provision": payload.provision},
            queue="webapi",
        )
        log.info("webapi.embed-parallel.success", task_id=job.id)
        return EmbedParallelResponse(task_id=job.id)
    except Exception as e:
        log.error("webapi.embed-parallel.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper-matcher", response_model=EmbedParallelResponse)
def paper_matcher(payload: PaperMatcherRequest):
    if embedder_service is None:
        raise HTTPException(status_code=503, detail="Service is not ready yet.")
    
    try:
        log.info("webapi.paper-matcher.start", user_id=payload.user_id)
        job = single_user.apply_async(
            kwargs={
                "metadata": {
                    "user_id": payload.user_id,
                    "language_code": payload.language_code
                }
            },
            queue="webapi",
        )
        log.info("webapi.paper-matcher.success", task_id=job.id)
        return EmbedParallelResponse(task_id=job.id)
    except Exception as e:
        log.error("webapi.paper-matcher.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gemini-batch-webhook")
async def gemini_batch_webhook(request: Request):
    body = await request.json()
    log.info("webapi.gemini-batch-webhook.start", body=body)
    return {"ok": True}
