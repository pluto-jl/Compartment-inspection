"""Experimental full-image plus overlapping tiles; no SAHI dependency.

Reimplements the published slice/detect/merge idea, not a new trained module.
"""
import time
import numpy as np


def tile_windows(width,height,size=640,overlap=.2):
    if width<1 or height<1 or size<1 or not 0<=overlap<1:
        raise ValueError('Invalid image or slicing settings')
    if max(width,height)<=size:return []  # Avoid redundant duplicate inference.
    def positions(length):
        if length<=size:return [0]
        stride=max(1,round(size*(1-overlap)))
        values=list(range(0,length-size+1,stride))
        if values[-1]!=length-size:values.append(length-size)
        return values
    return [(x,y,min(width,x+size),min(height,y+size)) for y in positions(height) for x in positions(width)]


def merge_nms(predictions,threshold=.5,max_det=100):
    pending=sorted(predictions,key=lambda p:-p['conf'])
    kept=[]
    while pending and len(kept)<max_det:
        best=pending.pop(0);kept.append(best)
        a=np.array(best['xyxy'],dtype=float)
        if not pending:break
        b=np.array([p['xyxy'] for p in pending],dtype=float)
        intersection=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(axis=1)
        area_a=np.maximum(0,a[2:]-a[:2]).prod()
        area_b=np.maximum(0,b[:,2:]-b[:,:2]).prod(axis=1)
        overlaps=intersection/np.maximum(1e-9,area_a+area_b-intersection)
        pending=[p for p,o in zip(pending,overlaps) if o<=threshold]
    return kept


def _records(result,offset=(0,0),image_shape=None):
    boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
    records=[]
    for box,score in zip(boxes,scores):
        box=[v+offset[i%2] for i,v in enumerate(box)]
        if image_shape is not None:
            h,w=image_shape
            box=[max(0,min(w if i%2==0 else h,v)) for i,v in enumerate(box)]
        if box[2]>box[0] and box[3]>box[1]:records.append(dict(xyxy=box,conf=score))
    return records


def predict_image(model,image,tiled=False):
    height,width=image.shape[:2]
    started=time.perf_counter()
    options=dict(imgsz=640,conf=.25,iou=.5,device='0',quantize=32,verbose=False,max_det=100)
    predictions=_records(model.predict(image,**options)[0],image_shape=(height,width))
    if not str(model.predictor.device).startswith('cuda'):
        raise RuntimeError('Tiled experiment requires actual CUDA inference')
    windows=tile_windows(width,height) if tiled else []
    for start in range(0,len(windows),8):
        batch=windows[start:start+8]
        crops=[np.ascontiguousarray(image[y1:y2,x1:x2]) for x1,y1,x2,y2 in batch]
        results=model.predict(crops,**options)
        assert len(results)==len(batch),'Inference returned an incomplete tile batch'
        for result,window in zip(results,batch):
            predictions.extend(_records(result,window[:2],(height,width)))
    if tiled and windows:predictions=merge_nms(predictions)
    return predictions,dict(milliseconds=(time.perf_counter()-started)*1000,tiles=len(windows),
                            model_calls=1+(len(windows)+7)//8,actual_device=str(model.predictor.device))
