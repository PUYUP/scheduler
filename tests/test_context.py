from atlazer.celery_app.tasks.workspace.context import summarize_similarities


def main():
    metadata = {
        "context_id": "37ef8517-adca-4273-afc3-3dafbe76c885",
        "workspace_id": "91439e8d-7858-4ccc-8c87-3a160b904678",
        "language_code": "en"
    }

    summarize_similarities(metadata)


if __name__ == "__main__":
    main()
