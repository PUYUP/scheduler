from atlazer.celery_app.tasks.matcher import batch_user

def main():
    batch_user()


if __name__ == '__main__':
    main()
