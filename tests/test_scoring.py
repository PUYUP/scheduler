from sklearn.metrics.pairwise import cosine_similarity
from atlazer.utils.answer_scoring import (
    getting_paper_vector, 
    getting_profile_vector,
    embedding_similarity_matrix,
    embedding_similarity
)


def main():
    paper_id = '00d4437f-baf3-4f68-a567-dd708a76a7df'
    user_id = 'b2a99b27-6d85-4be1-8a59-0f04a959b18c'
    paper_vectors = getting_paper_vector(paper_id)
    profile_vectors = getting_profile_vector(user_id)
    
    similarity_score = cosine_similarity(paper_vectors, profile_vectors)
    print(similarity_score)


if __name__ == "__main__":
    main()
