"""Conservative camera-view change hint for the *presentation layer*.

This module never changes model predictions, confidence scores, or event counts.
A substantial scene difference disables short-lived region holdovers.
"""
from __future__ import annotations
import cv2
import numpy as np

class ViewStability:
    def __init__(self, max_mean_change=14.0, max_gap_seconds=1.0):
        self.previous = None
        self.timestamp = None
        self.max_mean_change = float(max_mean_change)
        self.max_gap_seconds = float(max_gap_seconds)

    def add(self, frame, stamp):
        if frame is None or len(frame.shape) not in (2, 3):
            raise ValueError('画面格式无效')
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        thumb = cv2.resize(gray, (64, 40), interpolation=cv2.INTER_AREA)
        stamp = float(stamp)
        if self.previous is None:
            stable, reason, diff = False, 'first_frame', None
        elif self.timestamp is not None and (stamp < self.timestamp or stamp - self.timestamp > self.max_gap_seconds):
            stable, reason, diff = False, 'time_gap', None
        else:
            diff = float(np.mean(cv2.absdiff(thumb, self.previous)))
            stable = diff <= self.max_mean_change
            reason = 'stable' if stable else 'view_changed'
        self.previous = thumb
        self.timestamp = stamp
        return {'stable': bool(stable), 'reason': reason, 'mean_change': diff}
