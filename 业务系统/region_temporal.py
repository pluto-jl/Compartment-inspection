"""Visual-only microdroplet region aggregation.

A held region is explicitly marked as *recent evidence*, never a detection
on the current frame. Raw bounding boxes remain untouched. This algorithm
must not be used as a safety decision, tracking accuracy metric, or fault
classification mechanism. No holdovers survive camera movement or gaps.
"""
from __future__ import annotations


def _clip(xyxy, width, height):
    if len(xyxy) != 4: return None
    try: x1, y1, x2, y2 = (float(v) for v in xyxy)
    except (TypeError, ValueError): return None
    from math import isfinite
    if not all(isfinite(v) for v in (x1,y1,x2,y2)): return None
    x1, x2 = sorted((max(0.0, min(x1, width - 1)), max(0.0, min(x2, width - 1))))
    y1, y2 = sorted((max(0.0, min(y1, height - 1)), max(0.0, min(y2, height - 1))))
    if x2 - x1 < 1 or y2 - y1 < 1: return None
    return [x1, y1, x2, y2]


def _overlap(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter=x*y
    area=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/area if area > 0 else 0.


class RegionFusion:
    def __init__(self, hold_seconds=.4, min_hits=2):
        self.hold_seconds=float(hold_seconds)
        self.min_hits=max(2,int(min_hits))
        self.tracks=[]
        self.last_stamp=None

    def add(self, stamp, boxes, width, height, stable):
        stamp=float(stamp)
        if not stable or (self.last_stamp is not None and (stamp < self.last_stamp or stamp-self.last_stamp>1.0)):
            self.tracks=[]
        self.last_stamp=stamp
        current=[_clip(b.get('xyxy',[]),width,height) for b in boxes]
        for rect in current:
            if rect is None: continue
            existing=next((t for t in self.tracks if _overlap(rect,t['xyxy'])>=.25),None)
            if existing:
                existing.update(xyxy=rect,last=stamp,hits=existing['hits']+1)
            else:
                self.tracks.append(dict(xyxy=rect,last=stamp,hits=1))
        self.tracks=[t for t in self.tracks if stamp-t['last'] <= self.hold_seconds]
        return [dict(xyxy=t['xyxy'],held=(t['last']<stamp),supported=t['hits']>=self.min_hits)
                for t in self.tracks if t['hits']>=self.min_hits]


def visible_regions(boxes, confirmed, width, height):
    current=[]
    for b in boxes:
        rect=_clip(b.get('xyxy',[]),width,height)
        # Preserve one drawable record for every current detection (even invalid); never hide a detection.
        if rect is None:
            rect=[0.,0.,0.,0.]
        current.append({'xyxy':rect,'held':False,'supported':any(_overlap(rect,c['xyxy'])>=.25 for c in confirmed)})
    for c in confirmed:
        rect=_clip(c.get('xyxy',[]),width,height)
        if rect is None or not c.get('held'): continue
        if any(_overlap(rect,b['xyxy'])>=.25 for b in current): continue
        current.append({'xyxy':rect,'held':True,'supported':True})
    return current
