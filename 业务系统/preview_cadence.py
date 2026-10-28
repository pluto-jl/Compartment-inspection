"""Throttle display encoding only; detection and evidence bypass this helper."""
import time


class PreviewCadence:
    def __init__(self, publish, interval=.25, clock=time.monotonic):
        self.publish = publish
        self.interval = interval
        self.clock = clock
        self.last = None
        self.pending = None
        self.encoded = 0

    def submit(self, frame):
        self.pending = frame
        current = self.clock()
        if self.last is None or current - self.last >= self.interval:
            self.flush()

    def flush(self):
        if self.pending is not None:
            self.publish(self.pending)
            self.pending = None
            self.last = self.clock()
            self.encoded += 1
