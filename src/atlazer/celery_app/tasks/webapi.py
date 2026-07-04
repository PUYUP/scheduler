import structlog

from typing import List, Dict, Any
from atlazer.celery_app.main import app
from atlazer.utils.embedder import chunks_to_vector
from atlazer.celery_app.main import db_pool
from atlazer.storage.user import UserDepot
from atlazer.models.user import ProfileUpdate

log = structlog.get_logger(__name__)


@app.task(
    name="atlazer.celery_app.tasks.webapi.generate_embeddings",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="webapi",
    ignore_result=False,
)
def generate_embeddings(
    self,
    chunks: List[Dict[str, Any]],
    provision: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generates embedding vectors for a list of texts.

    Kwargs:
        texts (List[Dict[str, Any]]): List of texts to generate embeddings for.
        provision (Dict[str, Any] | None): Provision metadata.

    Returns:
        List[Dict[str, Any]]: List of texts with embeddings.
    
    Input:
        {
            "chunks": [
                {
                    "text": "The Role of Artificial Intelligence in Healthcare",
                },
            ]
        }
    
    Output:
        {
            "chunks": [
                {
                    "text": "The Role of Artificial Intelligence in Healthcare",
                    "embedding": [0.1, 0.2, 0.3, ...]
                }
            ]
        }
    """

    log.info("webapi.generate_embeddings.start", chunks_count=len(chunks), provision=provision)

    # generate embeddings
    embedded_chunks = chunks_to_vector(chunks)

    # store chunks for profile interest embedding
    if provision is not None:
        profile_id = provision.get("profile_id")
        if profile_id is not None:
            result = embedded_chunks[0]
            embedding = result["embedding"]
            log.info("webapi.generate_embeddings.profile_interest", embedding=embedding)
            
            try:
                user_depot = UserDepot(db_pool)
                user_depot.update_profile(profile_id, ProfileUpdate(interest_embedding=embedding))
                log.info("webapi.generate_embeddings.profile_interest.success", profile_id=profile_id, embedding=embedding)
            except Exception as e:
                log.error("webapi.generate_embeddings.profile_interest.failed", error=str(e))

    return {
        "chunks": embedded_chunks
    }
