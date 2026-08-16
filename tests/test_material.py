from atlazer.celery_app.tasks.workspace.material import upload_material, generate_jsonl, documents_deduplication


def main():
    workspace_id = "91439e8d-7858-4ccc-8c87-3a160b904678"
    
    # metadata = documents_deduplication(metadata={
    #     "workspace_id": workspace_id,
    #     "processing_date": "2026-08-16",
    #     "language_code": "en"
    # })

    # print(metadata)
    
    # chunks = generate_jsonl(metadata=metadata)
    upload_material(metadata={
        "workspace_id": workspace_id,
        "material_id": "7c3424c1-cbb5-4a22-aa7e-ed42ac4fda65",
    })


if __name__ == "__main__":
    main()