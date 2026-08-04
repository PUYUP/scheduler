import redis
from atlazer.config.settings import settings

def get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[return-value]
