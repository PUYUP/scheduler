from atlazer.celery_app.tasks.evaluation import generate_jsonl


def main():
    challenge_id = '37312e0d-bf6c-4a42-86c8-480d3bc23475'
    answer_id = 'e93db9e3-fe76-451e-adbf-46af16d59d50'
    paper_id = 'e89aeba7-23a8-4e53-95f7-2bd3681aeab7'
    user_id = 'a1ffa462-1595-4373-92ff-2d422cbef153'

    generate_jsonl({
        "challenge_id": challenge_id,
        "answer_id": answer_id,
        "paper_id": paper_id,
        "user_id": user_id,
    })


if __name__ == '__main__':
    main()