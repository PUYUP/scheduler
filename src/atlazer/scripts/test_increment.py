from atlazer.celery_app.tasks.scrape import scrape_topic_incremental


def main():
    result = scrape_topic_incremental(
        topic="cs.AI",
        repository="arxiv",
        start=1,
    )
    print(f"Task ID: {result}")


if __name__ == '__main__':
    main()
