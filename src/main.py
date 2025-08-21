import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info(f"Gandalf started at {datetime.now()}")
    while True:
        logger.info("Gandalf is running...")
        await asyncio.sleep(60)  # Run every minute

if __name__ == "__main__":
    asyncio.run(main())