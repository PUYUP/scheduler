"""
Skrip untuk merevoke (menghentikan) task Celery 'summarize_paper'
yang sedang berjalan (active) maupun yang masih mengantre (reserved)
di semua worker.

Cara pakai:
    python revoke_summarize_paper.py
"""

import json

from atlazer.celery_app.main import app

TASK_NAME = "atlazer.celery_app.tasks.matcher.summarize_paper"

# Nama queue yang mau dibersihkan. Kalau task ini di-routing ke queue
# khusus (bukan default "celery"), sesuaikan daftar ini.
QUEUE_NAMES = ["matcher"]


def revoke_active_tasks(inspector):
    """Revoke task yang sedang berjalan (active) dengan SIGKILL."""
    active_tasks = inspector.active() or {}
    revoked = []
    for worker, tasks in active_tasks.items():
        for task in tasks:
            if task["name"] == TASK_NAME:
                app.control.revoke(task["id"], terminate=True, signal="SIGKILL")
                revoked.append((worker, task["id"]))
                print(f"[ACTIVE]   Revoked task {task['id']} di worker {worker}")
    return revoked


def revoke_reserved_tasks(inspector):
    """Revoke task yang masih mengantre (reserved), tanpa terminate."""
    reserved_tasks = inspector.reserved() or {}
    revoked = []
    for worker, tasks in reserved_tasks.items():
        for task in tasks:
            if task["name"] == TASK_NAME:
                app.control.revoke(task["id"])
                revoked.append((worker, task["id"]))
                print(f"[RESERVED] Revoked task {task['id']} di worker {worker}")
    return revoked


def revoke_scheduled_tasks(inspector):
    """Revoke task yang dijadwalkan (punya ETA/countdown), belum dieksekusi.

    Struktur item scheduled() berbeda dari active()/reserved(): info task
    (id, name, dll) ada di dalam key 'request', bukan langsung di root dict.
    """
    scheduled_tasks = inspector.scheduled() or {}
    revoked = []
    for worker, tasks in scheduled_tasks.items():
        for task in tasks:
            request = task.get("request", {})
            if request.get("name") == TASK_NAME:
                app.control.revoke(request["id"])
                revoked.append((worker, request["id"]))
                print(f"[SCHEDULED] Revoked task {request['id']} di worker {worker}")
    return revoked


def purge_task_from_redis_queue(queue_names=QUEUE_NAMES):
    """Hapus pesan mentah task ini langsung dari list Redis yang jadi antrean.

    Ini beda dari revoke(): revoke() cuma menyuruh worker ABAIKAN task id
    tertentu (disimpan di memori worker). Kalau pesannya masih ada di
    Redis (belum di-ACK, misal karena worker di-SIGKILL saat acks_late=True),
    Redis akan MENGIRIM ULANG pesan itu ke worker lain setelah visibility
    timeout lewat -> task seolah "hidup lagi".

    Fungsi ini membaca seluruh isi list queue, membuang pesan yang task
    name-nya cocok, lalu menulis ulang list tanpa pesan tersebut. Dengan
    begitu pesannya benar-benar hilang, tidak akan di-requeue lagi.
    """
    import redis

    redis_client = app.connection().default_channel.client
    total_removed = 0

    for queue_name in queue_names:
        raw_messages = redis_client.lrange(queue_name, 0, -1)
        kept_messages = []
        removed_here = 0

        for raw in raw_messages:
            try:
                envelope = json.loads(raw)
                # Celery protocol v2: nama task ada di headers['task']
                task_name = (envelope.get("headers") or {}).get("task")
            except (json.JSONDecodeError, AttributeError):
                task_name = None

            if task_name == TASK_NAME:
                removed_here += 1
            else:
                kept_messages.append(raw)

        if removed_here:
            # Tulis ulang queue: hapus dulu, lalu push balik pesan yang
            # disimpan (urutan asli list Redis: index 0 = kepala antrean,
            # jadi push ulang pakai rpush + urutan yang sama).
            pipe = redis_client.pipeline()
            pipe.delete(queue_name)
            if kept_messages:
                pipe.rpush(queue_name, *kept_messages)
            pipe.execute()
            print(f"[PURGE]    Hapus {removed_here} pesan dari queue '{queue_name}'")

        total_removed += removed_here

    return total_removed


def main():
    inspector = app.control.inspect()

    print(f"Mencari task '{TASK_NAME}' di seluruh worker...\n")

    active_revoked = revoke_active_tasks(inspector)
    reserved_revoked = revoke_reserved_tasks(inspector)
    scheduled_revoked = revoke_scheduled_tasks(inspector)
    purged = purge_task_from_redis_queue()

    total = (
        len(active_revoked)
        + len(reserved_revoked)
        + len(scheduled_revoked)
        + purged
    )
    if total == 0:
        print("Tidak ada task yang ditemukan/direvoke/dihapus.")
    else:
        print(
            f"\nSelesai. Total {total} task ditangani "
            f"({len(active_revoked)} active, "
            f"{len(reserved_revoked)} reserved, "
            f"{len(scheduled_revoked)} scheduled, "
            f"{purged} dihapus langsung dari Redis queue)."
        )


if __name__ == "__main__":
    main()