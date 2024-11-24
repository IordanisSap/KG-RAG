

def log(text: str, level: str = "INFO"):
    """
    Log a message to the console.

    Args:
        text: The message to log.
        level: The level of the log message. Default is "INFO".
    """
    print(f"[{level}] {text}")
    
    
    
def call_in_batches(func, array, batch_size=1):
    result = []
    for i in range(0, len(array), batch_size):
        batch = array[i:i + batch_size]
        tmp = func(batch)
        result.extend(tmp)
        log(f"({min((i+batch_size), len(array))}/{len(array)})")
    return result