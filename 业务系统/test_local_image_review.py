import unittest,tempfile
from types import SimpleNamespace
from pathlib import Path
from patrol import App,detection_options
from local_image_review import append_candidates,crop_candidates
class LocalImageReviewTests(unittest.TestCase):
 def test_preserves_original_and_candidate_provenance(self):
  full=[dict(xyxy=[10,10,20,20],confidence=.8,label='droplet')]
  c=[dict(xyxy=[11,10,21,20],conf=.9,support_views=[0,1]),dict(xyxy=[40,40,50,50],conf=.6,support_views=[1,3])]
  result=append_candidates(full,c);self.assertEqual(result[0],full[0]);self.assertEqual(len(result),2);self.assertEqual(result[1]['source'],'local_zoom_candidate');self.assertEqual(result[1]['support_views'],[1,3])
 def test_local_mode_rejects_video_before_creating_job(self):
  with tempfile.TemporaryDirectory() as td:
   app=App(td);source=Path(td)/'demo.mp4';source.write_bytes(b'test')
   with self.assertRaisesRegex(ValueError,'仅支持单张图片'):app.start(dict(mode='file',point='test',model='improved',detail='local'),source)
   self.assertIsNone(app.job)
 def test_gpu_mismatch_rejected_for_local_inference(self):
  import numpy as np
  m=SimpleNamespace(predict=lambda *a,**k:[None]*4,predictor=SimpleNamespace(device='cpu'))
  with self.assertRaisesRegex(RuntimeError,'GPU'):crop_candidates(m,np.zeros((100,100,3),np.uint8),.25,'0')
 def test_nested_candidate_does_not_duplicate_a_full_box(self):
  full=[dict(xyxy=[0,0,10,100],confidence=.7,label='droplet')]
  c=[dict(xyxy=[2,60,8,90],conf=.9,support_views=[0,1])]
  self.assertEqual(append_candidates(full,c),full)
 def test_new_candidates_deduplicate_each_other(self):
  c=[dict(xyxy=[2,60,8,90],conf=.6,support_views=[0,1]),dict(xyxy=[0,0,10,100],conf=.8,support_views=[0,1])]
  out=append_candidates([],c);self.assertEqual(len(out),1);self.assertEqual(out[0]['confidence'],.8)
if __name__=='__main__':unittest.main()
