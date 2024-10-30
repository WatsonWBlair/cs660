import time

def run_timer(func, *args):
    start_time = time.time()
    results = func(*args)
    end_time = time.time()
    return results, end_time - start_time