import numpy as np
from typing import List, Dict, Any
from atlazer.storage.paper import PaperDepot
from atlazer.celery_app.main import db_pool
from atlazer.storage.user import UserDepot
from atlazer.storage.challenge import ChallengeDepot


def getting_paper_chunks(paper_id: str) -> List[Dict[str, Any]]:
    """
    Returns the embedding of a paper as a list of lists of floats.

    Args:
        paper_id: The ID of the paper to retrieve the embedding for
    Returns:
        A list of embeddings, where each embedding is a list of floats
        e.g., [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        If the paper is not found, it will raise a ValueError
        If a chunk is missing an embedding, it will be skipped
    """
    depot = PaperDepot(db_pool)
    chunks = depot.get_chunks_by_paper_id(paper_id)
    if chunks is None:
        raise ValueError(f"Paper {paper_id} not found")
    
    # Extract embeddings and filter out None values
    embeddings = []
    for chunk in chunks:
        # Check if embedding is not None and has the correct dimension (1024)
        if chunk.embedding is not None and len(chunk.embedding) == 1024:
            embeddings.append({
                "id": str(chunk.id),
                "embedding": chunk.embedding,
                "content": chunk.content,
            })
        else:
            # Optional: Log a warning if a chunk is skipped
            print(f"Skipping chunk {chunk.id} in paper {paper_id} due to invalid/missing embedding")
    
    return embeddings

def getting_profile_vectors(user_id: str) -> List[List[float]]:
    depot = UserDepot(db_pool)
    profile = depot.get_profile_by_user_id(user_id)
    if profile is None:
        raise ValueError(f"Profile for user {user_id} not found")
    if profile.interest_embedding is None or len(profile.interest_embedding) != 1024:
        raise ValueError(f"Profile {user_id} has invalid/missing embedding")
    return [profile.interest_embedding]

def getting_answer_chunks(answer_id: str) -> List[Dict[str, Any]]:
    depot = ChallengeDepot(db_pool)
    chunks = depot.get_chunks_by_answer_id(answer_id)
    if chunks is None:
        raise ValueError(f"Answer {answer_id} not found")

    # Extract embeddings and filter out None values
    embeddings = []
    for chunk in chunks:
        # Check if embedding is not None and has the correct dimension (1024)
        if chunk.embedding is not None and len(chunk.embedding) == 1024:
            embeddings.append({
                "id": str(chunk.id),
                "embedding": chunk.embedding,
                "content": chunk.content,
            })
        else:
            # Optional: Log a warning if a chunk is skipped
            print(f"Skipping chunk {chunk.id} in answer {answer_id} due to invalid/missing embedding")
    
    return embeddings

def embedding_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    Args:
        embedding1: numpy array of the first embedding
        embedding2: numpy array of the second embedding
    Returns:
        cosine similarity score
    """
    # Ensure embeddings are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(similarity)

def embedding_similarity_matrix(embeddings_A, embeddings_B):
    """
    Calculate the cosine similarity matrix between two sets of embeddings.
    Args:
        embeddings_A: list of numpy arrays
        embeddings_B: list of numpy arrays
    Returns:
        similarity_matrix: numpy array where element (i, j) is the cosine similarity
                         between embeddings_A[i] and embeddings_B[j]
    """
    # Convert lists of embeddings to numpy matrices
    matrix_A = np.array(embeddings_A)
    matrix_B = np.array(embeddings_B)
    
    # Calculate dot products between each pair of embeddings
    dot_products = np.dot(matrix_A, matrix_B.T)
    
    # Calculate magnitudes of each embedding
    magnitudes_A = np.linalg.norm(matrix_A, axis=1)
    magnitudes_B = np.linalg.norm(matrix_B, axis=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = dot_products / (magnitudes_A[:, np.newaxis] * magnitudes_B)
    
    return similarity_matrix
    