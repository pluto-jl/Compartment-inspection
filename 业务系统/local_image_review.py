"""Additional local candidates for manual image review only."""

def crop_candidates(model,image,threshold,device):
    import numpy as np
    from local_zoom_review import windows,supported_candidates,not_clipped
    height,width=image.shape[:2];areas=windows(width,height)
    crops=[np.ascontiguousarray(image[y1:y2,x1:x2]) for x1,y1,x2,y2 in areas]
    results=model.predict(crops,imgsz=640,conf=threshold,iou=.5,device=device,quantize=32,verbose=False,max_det=100)
    if len(results)!=len(areas):raise RuntimeError('局部复核未返回完整裁片结果')
    if device=='0' and not str(model.predictor.device).startswith('cuda'):
        raise RuntimeError('局部复核未运行在GPU上')
    proposals=[]
    for view,(result,area) in enumerate(zip(results,areas)):
        for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            if not not_clipped(box,area[2]-area[0],area[3]-area[1]):continue
            mapped=[value+area[i%2] for i,value in enumerate(box)]
            proposals.append(dict(xyxy=mapped,conf=score,view=view))
    return supported_candidates(proposals)


def append_candidates(boxes,candidates):
    from local_zoom_review import is_redundant
    added=[]
    for p in sorted(candidates,key=lambda p:-p['conf']):
        if not any(is_redundant(p['xyxy'],b['xyxy']) for b in boxes+added):
            added.append(dict(xyxy=[round(v,2) for v in p['xyxy']],confidence=round(float(p['conf']),5),label='droplet',source='local_zoom_candidate',support_views=p['support_views']))
    return boxes+added
