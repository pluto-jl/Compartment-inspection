"""Fixed local-zoom experiment: geometry-only crops, two-view support, no training."""
import math
from tiled_detection import merge_nms

def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def windows(width,height):
    if width<2 or height<2:raise ValueError('Image too small')
    w=min(width,max(2,math.ceil(width*.6)));h=min(height,max(2,math.ceil(height*.6)))
    return list(dict.fromkeys((x,y,x+w,y+h) for y in [0,height-h] for x in [0,width-w]))

def supported_candidates(predictions):
    """Each support must come from a different crop; clustered around fixed anchor."""
    todo=sorted(predictions,key=lambda p:-p['conf']);out=[]
    while todo:
        anchor=todo.pop(0);cluster=[anchor];rest=[]
        for p in todo:
            if overlap(anchor['xyxy'],p['xyxy'])>=.5:cluster.append(p)
            else:rest.append(p)
        todo=rest;views={p['view'] for p in cluster}
        if len(views)<2:continue
        best_by_view={v:max((p for p in cluster if p['view']==v),key=lambda p:p['conf']) for v in views}
        out.append(dict(xyxy=anchor['xyxy'],conf=min(p['conf'] for p in best_by_view.values()),support_views=sorted(views)))
    return merge_nms(out,.5)

def smaller_overlap(a,b):
    """Intersection over smaller area detects part/body duplicate boxes."""
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    if min(aa,bb)<=0:return 0.0
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/min(aa,bb)


def is_redundant(candidate,existing):
    if overlap(candidate,existing)>=.5 or smaller_overlap(candidate,existing)>=.8:return True
    area_a=max(0,candidate[2]-candidate[0])*max(0,candidate[3]-candidate[1])
    area_b=max(0,existing[2]-existing[0])*max(0,existing[3]-existing[1])
    small,large=(candidate,existing) if area_a<area_b else (existing,candidate)
    cx=(small[0]+small[2])/2;cy=(small[1]+small[3])/2
    return smaller_overlap(candidate,existing)>=.5 and large[0]<=cx<=large[2] and large[1]<=cy<=large[3]


def combine(full,candidates):
    # Existing full-image detections remain unchanged and in the original order.
    kept=list(full)
    for p in sorted(candidates,key=lambda p:-p['conf']):
        if not any(is_redundant(p['xyxy'],b['xyxy']) for b in kept):kept.append(p)
    return kept


def not_clipped(local_box,width,height,margin=2):
    """Only crop-derived proposals are filtered. The full-image branch is preserved."""
    x1,y1,x2,y2=local_box
    return x1>margin and y1>margin and x2<width-margin and y2<height-margin


def expanded_windows(width,height):
    """Four corners plus one distinct center crop, independent of detections/labels."""
    base=windows(width,height);w=base[0][2]-base[0][0];h=base[0][3]-base[0][1]
    x=(width-w)//2;y=(height-h)//2
    return list(dict.fromkeys(base+[(x,y,x+w,y+h)]))
