import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from patrol import App,detection_options,file_sha256,device_metadata

try:
    import region_overlay  # noqa: F401
    REGION_OK=True
except Exception:
    # 轻量包未包含 region_temporal.py / view_stability.py 时跳过区域专用用例。
    REGION_OK=False


class RegionModeTests(unittest.TestCase):
    def test_actual_cpu_backend_rejected_in_gpu_mode(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            app=App(td);app.job=dict(threshold=.25,**detection_options({}))
            app.model_meta=dict(device='0')
            app.detector=SimpleNamespace(predict=lambda *a,**k:[None],predictor=SimpleNamespace(device='cpu'))
            with self.assertRaisesRegex(RuntimeError,'实际推理'):app.detect(np.zeros((8,8,3),np.uint8))

    def test_gpu_required_no_silent_cpu_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'GPU'):device_metadata()
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='GPU fixture'):
            self.assertEqual(device_metadata()['device'],'0')
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(App(td).state()['compute_policy'],'cuda_required')

    def test_region_preset(self):
        cfg=detection_options({'detail':'region'})
        self.assertEqual((cfg['imgsz'],cfg['sample_interval'],cfg['frame_limit']),(640,0,6000))

    def test_frame_iteration_does_not_skip_short_candidates(self):
        class Capture:
            def __init__(self):self.index=0
            def get(self,_):return 25
            def read(self):
                self.index+=1
                return (True,np.zeros((8,8,3),np.uint8)) if self.index<=25 else (False,None)
        with tempfile.TemporaryDirectory() as td:
            app=App(td)
            app.job=detection_options({'detail':'region'})
            self.assertEqual(len(list(app.iter_frames(Capture(),25,False))),25)
            app.job=detection_options({})
            self.assertEqual(len(list(app.iter_frames(Capture(),25,False))),2)

    @unittest.skipUnless(REGION_OK,'轻量包未包含区域时序模块')
    def test_region_run_persists_current_boxes_and_does_not_upgrade_fault(self):
        with tempfile.TemporaryDirectory() as td:
            app=App(td);source=Path(td)/'fixture.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),25,(80,80))
            self.assertTrue(writer.isOpened())
            for i in range(40):writer.write(np.full((80,80,3),i*5,np.uint8))
            writer.release()
            app.job=dict(id='fixture',model='improved',event_id=None,point='test',condition='expected_dry',
                         mode='file',status='loading',processed=0,positive_frames=0,duplicate_frames=0,
                         threshold=.25,**detection_options({'detail':'region'}))
            app.model_meta=dict(label='test fixture',sha256='test-only',device='cpu')
            app.load_model=lambda _:app.model_meta
            calls=[]
            def detect(frame):
                index=len(calls);calls.append(index)
                boxes=[dict(xyxy=[20,20,30,35],confidence=.8,label='droplet')] if index in (1,15,30) else []
                return frame.copy(),boxes,1
            app.detect=detect
            app.run(source)
            self.assertEqual(app.job['status'],'completed')
            self.assertEqual(app.job['processed'],40)
            self.assertEqual(app.job['positive_frames'],3)
            event=app.store.events()[0]
            self.assertEqual(event['level'],'candidate')
            self.assertEqual(event['hits'],3)
            trace=app.store.root/app.job['region_trace']
            self.assertEqual(file_sha256(trace),app.job['region_trace_sha256'])
            records=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records),40)
            for record in records:
                self.assertEqual(sum(not r['held'] for r in record['regions']),len(record['boxes']))
            self.assertEqual(records[1]['boxes'][0]['xyxy'],[20,20,30,35])

if __name__=='__main__':unittest.main()
