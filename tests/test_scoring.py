import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from atlazer.utils.answer_scoring import (
    getting_paper_chunks, 
    getting_profile_vectors,
    getting_answer_chunks,
    embedding_similarity_matrix,
    embedding_similarity
)


def scoring(paper_id: str):
    answer_id = 'e93db9e3-fe76-451e-adbf-46af16d59d50'
    
    paper_vectors = getting_paper_chunks(paper_id)
    paper_embeddings = [x["embedding"] for x in paper_vectors]
    paper_contents = [x["content"] for x in paper_vectors]

    answer_vectors = getting_answer_chunks(answer_id)
    answer_embeddings = [x["embedding"] for x in answer_vectors]
    answer_contents = [x["content"] for x in answer_vectors]
    similarity_matrix = cosine_similarity(answer_embeddings, paper_embeddings)
    data_to_insert = []

    print('paper_embeddings length', len(paper_embeddings))
    print('similarity_matrix length', len(similarity_matrix))
    print('answer embedding length', len(answer_embeddings))

    for i, answer_chunk_content in enumerate(answer_contents):
        scores_for_c = similarity_matrix[i]

        best_match_index = np.argmax(scores_for_c)
        highest_score = scores_for_c[best_match_index]
        best_match_paper_content = paper_contents[best_match_index]

        # print(f"\nChunk Content: {answer_chunk_content}")
        # print(f"Chunk Scores: {scores_for_c}")
        # print(f"Best Match Content: {best_match_content}")
        # print(f"Highest Score: {highest_score}")

        if highest_score > 0.0000000:
            data_to_insert.append({
                "answer_chunk_content": answer_chunk_content,
                "paper_chunk_content": best_match_paper_content,
                "similarity_score": highest_score,
                "classification_tag": paper_id
            })

    return data_to_insert


def main():
    # ax = scoring('00d4437f-baf3-4f68-a567-dd708a76a7df')
    # print('--------------------------------------')
    bx = scoring('c9c98571-aac5-4d4d-99f5-72af67e9fcb1')

    # print("Ax:", ax)
    print("Bx:", bx)


if __name__ == "__main__":
    main()
