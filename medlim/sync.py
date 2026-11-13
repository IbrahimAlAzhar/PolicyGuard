"""Coordination between the three per-model PBS jobs that share files (no flock needed: O_EXCL create is atomic
on Lustre/NFS). One job builds a shared file; the others wait for its 'done' marker."""
import os
import time


def try_lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} pid={os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def wait_for(predicate, what, timeout_min, every_s=60, log=print, lock=None, stale_min=240):
    """poll predicate() until True. Returns True/False. If `lock` exists but is older than stale_min, it is removed."""
    t0 = time.time()
    while True:
        if predicate():
            return True
        if lock and os.path.exists(lock) and time.time() - os.path.getmtime(lock) > stale_min * 60:
            log(f"removing stale lock {lock}")
            release(lock)
        waited = (time.time() - t0) / 60
        if waited > timeout_min:
            log(f"gave up waiting for {what} after {waited:.0f} min")
            return False
        log(f"waiting for {what} ({waited:.0f}/{timeout_min} min)")
        time.sleep(every_s)
