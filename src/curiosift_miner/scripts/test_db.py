from curiosift_miner.storage import db
from curiosift_miner.config.settings import settings

import asyncio

async def main():
    config = db.DatabaseConfig()

    async with db.DatabasePool(config) as pool:
        # await pool.ensure_schema()
        async with pool.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM papers LIMIT 2")
            print(rows)

if __name__ == "__main__":
    asyncio.run(main())
