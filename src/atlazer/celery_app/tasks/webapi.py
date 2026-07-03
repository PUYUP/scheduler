from typing import Dict, Any
from atlazer.celery_app.main import app
from atlazer.utils.embedder import chunks_to_vector


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
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generates embedding vectors for a list of texts.

    Kwargs:
        texts (List[Dict[str, Any]]): List of texts to generate embeddings for.

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

    embedded_chunks = chunks_to_vector(metadata["chunks"])
    return {
        "chunks": embedded_chunks
    }
