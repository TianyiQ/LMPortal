import asyncio
from typing import Optional

from tqdm import tqdm

# Optional event loop for use throughout the program
CUR_LOOP = None


def start_loop() -> asyncio.BaseEventLoop:
    "Start an event loop for all future coroutines to run in. Necessary if executing multiple reasoning runs."
    global CUR_LOOP
    CUR_LOOP = asyncio.get_event_loop()
    print(f"New event loop: {CUR_LOOP}")

    return CUR_LOOP


async def await_with_retry(coroutine, max_retries: int = 5, retry_delay: float = 1.0, retry_exp_base: float = 2.0):
    "Run an async function from a sync environment, with a retry mechanism."
    from utils.io_utils import logger

    for retry_idx in range(max_retries):
        try:
            return await coroutine
        except Exception as e:
            logger.minor(f"Error: {e}", dedup="message_stem", max_count=10)
            await asyncio.sleep(retry_delay * (retry_exp_base**retry_idx))
    raise Exception(f"Failed after {max_retries} retries")


async def await_with_pbar(coroutine, pbar: Optional[tqdm] = None, update_amount: int = 1):
    "Run an async function from a sync environment, with a progress bar."
    result = await coroutine
    if pbar is not None:
        pbar.update(update_amount)
    return result


def run_coroutine(coroutine):
    "Run an async function from a sync environment, without starting multiple event loops."
    global CUR_LOOP

    try:
        # If start_loop is called before, use the loop created then
        if CUR_LOOP is not None:
            loop = CUR_LOOP
        else:
            loop = asyncio.get_running_loop()

    except RuntimeError:
        # No loop is running, so we can safely use asyncio.run()
        return asyncio.run(coroutine)

    # Schedule the coroutine and get the result
    return loop.run_until_complete(coroutine)
