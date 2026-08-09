from atlazer.celery_app.tasks.ingestion.extractors.iaescore import fetch_page


def main():
    # fetch_page.s("ijai").set(queue="iaescore").apply_async()
    fetch_page("ijai")


if __name__ == '__main__':
    main()
