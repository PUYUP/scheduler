from atlazer.celery_app.tasks.ingestion.extractors.arxiv import fetch_page


def main():
    fetch_page.apply_async()


if __name__ == '__main__':
    main()
