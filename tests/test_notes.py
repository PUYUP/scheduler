import numpy as np

from atlazer.celery_app.main import db_pool
from atlazer.storage.workspace_notes import WorkspaceNoteDepot
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.tasks.workspace.notes import dedup_daily_notes, save_enrichments


def main():
    workspace_id = "91439e8d-7858-4ccc-8c87-3a160b904678"
    # note_depot = WorkspaceNoteDepot(db_pool)
    # today_chunks = note_depot.get_chunks_daily(workspace_id)
    # embeddings = []
    # notes_ids = []

    # for c in today_chunks:
    #     notes_ids.append(str(c.id))
    #     embeddings.append(c.embedding)

    # embeddings = np.array(embeddings)

    # ncs = NotesClusteringService()
    # result = ncs.find_duplicates(notes_ids=notes_ids, embeddings=embeddings)
    # print(result)

    # dedup_daily_notes(metadata={"workspace_id": workspace_id})
    # print(result)

    result = save_enrichments(
        metadata={
            "key": f"notes/91439e8d-7858-4ccc-8c87-3a160b904678/2026/08/14",
            "job_id": "batches/nk22ufgsj1q8vwunlvl0naqluu3laxz0sf33",
            "workspace_id": workspace_id
        }
    )
    print(result)


if __name__ == "__main__":
    main()
