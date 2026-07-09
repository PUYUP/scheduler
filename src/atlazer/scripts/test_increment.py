from atlazer.celery_app.tasks.scrape import scrape_topic_incremental
from atlazer.config.settings import settings

def main():
    result = scrape_topic_incremental.apply_async(
        args=[
            "cs.AI",
            "arxiv",
            1,
        ],
        queue="scrape"
    )
    print(f"Task ID: {result}")


if __name__ == '__main__':
    main()
