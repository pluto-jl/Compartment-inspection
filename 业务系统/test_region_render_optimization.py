import unittest,tempfile
from types import SimpleNamespace
import numpy as np
from patrol import App,detection_options

try:
    from region_overlay import RegionPresentation
    REGION_OK=True
except Exception:
    REGION_OK=False

@unittest.skipUnless(REGION_OK,'轻量包未包含区域时序模块')
class RegionRenderOptimizationTests(unittest.TestCase):
 def test_region_detection_does_not_draw_or_copy_discarded_image(self):
  class Values:
   def __init__(self,v):self.v=v
   def cpu(self):return self
   def tolist(self):return self.v
  with tempfile.TemporaryDirectory() as td:
   app=App(td,device='cpu');app.job=dict(threshold=.25,**detection_options({'detail':'region'}));app.model_meta=dict(device='cpu')
   app.detector=SimpleNamespace(predict=lambda *a,**k:[SimpleNamespace(boxes=SimpleNamespace(xyxy=Values([[8,8,20,25]]),conf=Values([.8])))])
   frame=np.zeros((64,64,3),np.uint8);out,boxes,ms=app.detect(frame)
   self.assertIs(out,frame);self.assertFalse(frame.any());self.assertEqual(len(boxes),1)
 def test_empty_presentation_reuses_read_only_frame(self):
  frame=np.zeros((64,64,3),np.uint8);out,hint=RegionPresentation().render(frame,[],0)
  self.assertIs(out,frame);self.assertFalse(frame.any());self.assertEqual(hint['regions'],[])
 def test_nonempty_presentation_owns_output_and_preserves_input(self):
  frame=np.zeros((64,64,3),np.uint8);out,hint=RegionPresentation().render(frame,[dict(xyxy=[8,8,20,25],confidence=.8)],0)
  self.assertIsNot(out,frame);self.assertFalse(frame.any());self.assertTrue(out.any());self.assertEqual(hint['current_count'],1)
if __name__=='__main__':unittest.main()
