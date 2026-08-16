import numpy as np

from atlazer.celery_app.main import db_pool
from atlazer.storage.workspace.notes import WorkspaceNoteDepot
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.tasks.workspace.notes import (
    deduplicate_notes,
    save_enriched_notes,
    process_workspaces,
    chunk_enriched_notes,
)
from atlazer.celery_app.tasks.workspace.material import find_relevant_papers, documents_deduplication


def main():
    workspace_id = "91439e8d-7858-4ccc-8c87-3a160b904678"
    # note_depot = WorkspaceNoteDepot(db_pool)
    # today_chunks = note_depot.get_chunks_by_workspace(workspace_id)
    # embeddings = []
    # notes_ids = []

    # for c in today_chunks:
    #     notes_ids.append(str(c.id))
    #     embeddings.append(c.embedding)

    # embeddings = np.array(embeddings)

    # ncs = NotesClusteringService()
    # result = ncs.find_duplicates(notes_ids=notes_ids, embeddings=embeddings)
    # print(result)

    # deduplicate_notes(metadata={"workspace_id": workspace_id})
    # print(result)

    # result = save_enriched_notes(
    #     metadata={
    #         "key": f"notes/91439e8d-7858-4ccc-8c87-3a160b904678/2026/08/14",
    #         "job_id": "batches/nk22ufgsj1q8vwunlvl0naqluu3laxz0sf33",
    #         "workspace_id": workspace_id
    #     }
    # )
    # print(result)

    job = process_workspaces.apply_async()
    print(job.id)

    # chunks = chunk_enriched_notes(metadata={
    #     "workspace_id": workspace_id,
    #     "processing_date": "2026-08-14"
    # })
    # print(chunks)

    # job = chunk_enriched_notes.s(
    #     metadata={
    #         "workspace_id": workspace_id,
    #         "processing_date": "2026-08-14"
    #     }
    # ).apply_async()

    # job = find_relevant_papers.s(
    #     metadata={
    #         "workspace_id": workspace_id,
    #         "material_note_id": "b1e9b9b6-ddaf-40c7-ae54-56afbff39f0f"
    #     }
    # ).apply_async()

    # print(job.id)

    # res = find_relevant_papers(metadata={
    #     "workspace_id": workspace_id,
    #     "material_note_id": "b1e9b9b6-ddaf-40c7-ae54-56afbff39f0f"
    # })

    # print(res["matched_result"]["similar_chunks"])

    # save_similarities(metadata=res)

    # xx = documents_deduplication(metadata={
    #     "workspace_id": workspace_id,
    #     "processing_date": "2026-08-16"
    # })

    # print(xx)


if __name__ == "__main__":
    main()
