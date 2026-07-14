import structlog
from contextlib import asynccontextmanager
from typing import Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request

from atlazer.celery_app.main import db_pool
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
from atlazer.utils.gemini_batch import get_batch_results
from atlazer.storage.challenge import ChallengeDepot

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

    data = body.get("data")
    user_metadata = body.get("user_metadata")
    batch_type = body.get("type")

    if data and user_metadata and batch_type == "batch.succeeded":
        # getting batch result
        log.info(
            "webapi.gemini-batch-webhook.succeeded",
            data=data,
            user_metadata=user_metadata
        )

        batch_id = data.get("id")
        user_id = user_metadata.get("user_id")
        challenge_paper_id = user_metadata.get("challenge_paper_id")
        challenge_paper_summary_id = user_metadata.get("challenge_paper_summary_id")
        paper_id = user_metadata.get("paper_id")
        challenge_id = user_metadata.get("challenge_id")

        if batch_id and user_id and challenge_paper_id:
            try:
                result = get_batch_results(batch_id)
                log.info("webapi.gemini-batch-webhook.result", batch_id=batch_id)

                # store result in database
                depot = ChallengeDepot(db_pool)
                depot.update_challenge_paper_summary(
                    challenge_paper_summary_id=challenge_paper_summary_id,
                    update_data={
                        "result": result,
                        "tool": "google-gemini",
                        "model": "gemini-3.1-flash-lite",
                        "status": "completed",
                        "job_id": batch_id,
                        "finished_at": datetime.now(),
                    }
                )
            except Exception as e:
                log.error("webapi.gemini-batch-webhook.error", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
            
    return {"ok": True}
