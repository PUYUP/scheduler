from atlazer.celery_app.tasks.workspace.context import (
    chunking,
    embedding,
    save_embedding,
    find_relevant_papers,
)


def main():
    user_id = "a1ffa462-1595-4373-92ff-2d422cbef153"
    workspace_id = "91439e8d-7858-4ccc-8c87-3a160b904678"
    context_id = "bcf794a4-3427-476d-8fcb-f5ef81e708f8"
    content = "di era 2000-an saat internet menjadi sorotan dunia bermunculan situs web yang kemudian merubah cara orang berkirim data seperti melakukan pembelian daring. AI juga sama, apakah jauh dalam 20 tahun kedepan polanya akan terulang? afafa"
    language_code = "cu"

    metadata = {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "context_id": context_id,
        "content": content,
        "language_code": language_code
    }

    # chunk_result = chunking(metadata=metadata)
    # metadata.update({"chunks": chunk_result["chunks"]})

    # embedded_result = embedding(metadata=metadata)
    # metadata.update({"chunks": embedded_result["chunks"]})

    # save_embedding(metadata=metadata)
    res = find_relevant_papers(metadata=metadata)
    print(res)

if __name__ == "__main__":
    main()
