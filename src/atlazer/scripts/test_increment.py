from atlazer.celery_app.tasks.scrape import scrape_topic_incremental


def main():
    scrape_topic_incremental('cs.AI')


if __name__ == '__main__':
    main()
