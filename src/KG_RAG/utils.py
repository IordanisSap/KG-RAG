import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

def call_in_batches(func, array, batch_size=1, log_every=10):
    result = []
    total = len(array)

    for i in range(0, total, batch_size):
        batch = array[i:i + batch_size]
        tmp = func(batch)
        result.extend(tmp)
        if (i // batch_size) % log_every == 0 or i + batch_size >= total:
            logging.info(f"Processed {min(i + batch_size, total)}/{total}")
    return result