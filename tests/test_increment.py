from atlazer.celery_app.tasks.scrape import scrape_topic_increment
from atlazer.config.settings import settings

def main():
    # result = scrape_topic_increment.apply_async(
    #     args=[
    #         "arxiv",
    #         1,
    #     ],
    #     queue="scrape"
    # )
    # print(f"Task ID: {result}")

    result = scrape_topic_increment(
        "arxiv",
        1
    )
    print(f"Task ID: {result}")


if __name__ == '__main__':
    main()
