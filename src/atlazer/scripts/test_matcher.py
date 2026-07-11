from datetime import timedelta
from celery.utils.time import timezone
from datetime import datetime
from atlazer.models.user import ProfileUpdate
from atlazer.celery_app.main import db_pool
from atlazer.celery_app.tasks.matcher import single_user, batch_user
from atlazer.storage.user import UserDepot
from atlazer.storage.matcher import MatcherDepot

import random


def generate_placeholder_embedding(dim=1024, low=-1.0, high=1.0):
    return [random.uniform(low, high) for _ in range(dim)]


def main():
    # batch_user()
    papers = single_user({'user_id': 'a1ffa462-1595-4373-92ff-2d422cbef153'})
    print(papers)
    
    # user_a = results[0]
    # embedding_a = user_a.intereset_embedding

    # print(embedding_a)

    # embedding = generate_placeholder_embedding()
    # profile_id = "6461c0b8-aede-49ee-a4db-93ae6de988ce"

    # try:
    #     user_depot = UserDepot(db_pool)
    #     # Set next processed at to 48 hours from now, to prevent updating
    #     # frequently
    #     next_processed_at = datetime.now(timezone.utc) + timedelta(hours=48)
    #     user_depot.update_profile(
    #         profile_id, 
    #         ProfileUpdate(
    #             interest_embedding=embedding, 
    #             next_processed_at=next_processed_at
    #         )
    #     )
    #     print('OK')
    # except Exception as e:
    #     print('failed')
    #     print(e)


if __name__ == '__main__':
    main()
