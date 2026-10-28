"""Experimental approximate localization; never suppress current detections."""
import cv2
from contextlib import nullcontext
from region_temporal import RegionFusion,visible_regions
from view_stability import ViewStability


class RegionPresentation:
    def __init__(self):
        self.fusion=RegionFusion()
        self.view=ViewStability()

    def render(self, frame, boxes, stamp, profiler=None):
        height,width=frame.shape[:2]
        with profiler.measure('view_stability') if profiler else nullcontext():
            state=self.view.add(frame,stamp)
        with profiler.measure('region_fusion_and_drawing') if profiler else nullcontext():
            confirmed=self.fusion.add(stamp,boxes,width,height,state['stable'])
            regions=visible_regions(boxes,confirmed,width,height)
            assert sum(not r['held'] for r in regions)==len(boxes)
            # With no visible regions, downstream publish/record treat the frame as read-only.
            output=frame.copy() if regions else frame
            scale=max(.5,width/1600)
            for r in regions:
                color=(0,210,255) if r['held'] else ((70,240,100) if r['supported'] else (255,210,50))
                x1,y1,x2,y2=map(round,r['xyxy'])
                cv2.rectangle(output,(x1,y1),(x2,y2),color,max(2,width//600))
                label='recent evidence' if r['held'] else ('supported candidate' if r['supported'] else 'candidate region')
                cv2.putText(output,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,max(1,width//1000),cv2.LINE_AA)
        return output,dict(view=state,regions=regions,current_count=len(boxes),
                           held_count=sum(r['held'] for r in regions))
