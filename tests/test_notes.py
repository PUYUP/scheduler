import numpy as np

from atlazer.celery_app.main import db_pool
from atlazer.storage.workspace_notes import WorkspaceNoteDepot
from atlazer.utils.notes_clustering import NotesClusteringService


def main():
    note_depot = WorkspaceNoteDepot(db_pool)
    today_chunks = note_depot.get_chunks_daily()
    embeddings = []
    notes_ids = []

    for c in today_chunks:
        notes_ids.append(str(c.id))
        embeddings.append(c.embedding)

    embeddings = np.array(embeddings)
    print(embeddings.shape)

    ncs = NotesClusteringService()
    result = ncs.find_duplicates(notes_ids=notes_ids, embeddings=embeddings)
    print(result)


if __name__ == "__main__":
    main()
