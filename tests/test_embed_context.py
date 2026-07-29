import uuid

from atlazer.celery_app.tasks.workspace import chunk_context


def main():
    user_id = "a1ffa462-1595-4373-92ff-2d422cbef153"
    workspace_id = "91439e8d-7858-4ccc-8c87-3a160b904678"
    context_id = "2cba7e7b-e061-4bd2-9986-283f6ee32e20"
    content = "di era 2000-an saat internet menjadi sorotan dunia bermunculan situs web yang kemudian merubah cara orang berkirim data seperti melakukan pembelian daring. AI juga sama, apakah jauh dalam 20 tahun kedepan polanya akan terulang? afafa"
    language_code = "cu"
    chunks = chunk_context(
        metadata={
            "user_id": user_id,
            "workspace_id": workspace_id,
            "context_id": context_id,
            "content": content,
            "language_code": language_code
        }
    )
    print(chunks)


if __name__ == "__main__":
    main()
