from __future__ import annotations

from atlazer.celery_app.main import app, db_pool


@app.task(
    name="atlazer.celery_app.tasks.challenge.generate_challenge",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_challenge(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    pass
