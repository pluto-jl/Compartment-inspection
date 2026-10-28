"""Regression tests: UI persistence and region overlays never upgrade model detections."""
import unittest
import numpy as np
from region_temporal import RegionFusion, visible_regions
from view_stability import ViewStability

class RegionTemporalSafetyTests(unittest.TestCase):
    def test_current_detection_is_never_suppressed(self):
        fusion=RegionFusion();boxes=[{'xyxy':[10,10,30,30],'confidence':.8}]
        confirmed=fusion.add(0.,boxes,80,80,False)
        view=visible_regions(boxes,confirmed,80,80)
        self.assertEqual(len(view),1)
        self.assertFalse(view[0]['held'])
        self.assertEqual(view[0]['xyxy'],[10.,10.,30.,30.])

    def test_hold_is_separate_from_real_model_output(self):
        fusion=RegionFusion()
        boxes=[{'xyxy':[10,10,30,30],'confidence':.8}]
        fusion.add(0.,boxes,80,80,False)
        supported=fusion.add(.1,boxes,80,80,True)
        self.assertTrue(visible_regions(boxes,supported,80,80)[0]['supported'])
        held=fusion.add(.2,[],80,80,True)
        regions=visible_regions([],held,80,80)
        self.assertEqual(len(regions),1)
        self.assertTrue(regions[0]['held'])
        self.assertEqual(visible_regions([],fusion.add(.3,[],80,80,False),80,80),[])

    def test_big_scene_change_cancels_recent_evidence(self):
        v=ViewStability()
        self.assertFalse(v.add(np.zeros((80,80,3),np.uint8),0)['stable'])
        self.assertTrue(v.add(np.zeros((80,80,3),np.uint8),.1)['stable'])
        self.assertFalse(v.add(np.full((80,80,3),255,np.uint8),.2)['stable'])

if __name__=='__main__':unittest.main()
