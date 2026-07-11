from atlazer.celery_app.main import db_pool
from atlazer.storage.challenge import ChallengeDepot
from atlazer.celery_app.tasks.matcher import single_user
from datetime import date


def main():
    try:
        user_id = "a1ffa462-1595-4373-92ff-2d422cbef153"
        user_papers = single_user({"user_id": user_id})
        target_date = date(2022, 1, 1)
        depot = ChallengeDepot(db_pool)
        challenge = depot.insert_challenge(user_id=user_id, target_date=target_date, papers=user_papers)
        print(f"Challenge created with ID: {challenge.id}")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()