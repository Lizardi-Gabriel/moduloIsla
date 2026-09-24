"""Control de repeticion para logs locales y remotos, sin ocultar el primer fallo."""
from config import LOG_REPEAT_INTERVAL, LOG_REPEAT_CAPACITY
import logging
import threading
import time
from collections import OrderedDict


class LogLimiter:
    def __init__(self, interval=LOG_REPEAT_INTERVAL, capacity=LOG_REPEAT_CAPACITY):
        self.interval = interval
        self.capacity = capacity
        self._seen = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            previous = self._seen.get(key)
            if previous is not None and now - previous < self.interval:
                return False
            self._seen[key] = now
            self._seen.move_to_end(key)
            if len(self._seen) > self.capacity:
                self._seen.popitem(last=False)
            return True


class RepeatedWarningFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.limiter = LogLimiter()

    def filter(self, record):
        return getattr(record, "state_transition", False) or record.levelno < logging.WARNING or self.limiter.allow(
            (record.name, record.levelno, record.getMessage())
        )
