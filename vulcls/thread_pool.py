from typing import *
from concurrent.futures import ThreadPoolExecutor


_global_thread_pool: Optional[ThreadPoolExecutor] = None


def init_thread_pool(max_threads: int = 10):
    global _global_thread_pool
    _global_thread_pool = ThreadPoolExecutor(max_workers=max_threads)


def get_thread_pool() -> ThreadPoolExecutor:
    return _global_thread_pool


def shutdown_thread_pool():
    _global_thread_pool.shutdown()


__all__ = ['init_thread_pool', 'get_thread_pool', 'shutdown_thread_pool']
