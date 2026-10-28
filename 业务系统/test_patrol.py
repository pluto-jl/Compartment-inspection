import json
import tempfile
import unittest
import threading
from pathlib import Path
from patrol import Window, Store, App, csv_safe, report_html, detection_options


class RulesTest(unittest.TestCase):
    def test_inference_presets_and_reject_unknown(self):
        self.assertEqual(detection_options({})['imgsz'],640)
        self.assertEqual(detection_options({'detail':'detail'})['imgsz'],960)
        with self.assertRaises(ValueError):detection_options({'detail':'unbounded'})

    def test_selected_size_reaches_real_prediction_call(self):
        import numpy as np
        class ListValue:
            def cpu(self):return self
            def tolist(self):return []
        class Detector:
            def predict(self,image,**kwargs):
                self.arguments=kwargs
                from types import SimpleNamespace
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=ListValue(),conf=ListValue()))]
        with tempfile.TemporaryDirectory() as td:
            app=App(td);app.detector=Detector();app.model_meta={'device':'cpu'}
            app.job=dict(threshold=.25,**detection_options({'detail':'detail'}))
            app.detect(np.zeros((40,60,3),dtype=np.uint8))
            self.assertEqual(app.detector.arguments['imgsz'],960)
            self.assertEqual(app.detector.arguments['iou'],.5)

    def test_duplicate_not_counted_and_timed_positive_window(self):
        w=Window()
        self.assertEqual(w.add(0,True,'same'),(True,False))
        self.assertEqual(w.add(.5,True,'same'),(False,False))
        self.assertEqual(w.hits,1)
        self.assertEqual(w.add(.5,True,'next'),(True,False))
        self.assertEqual(w.add(1,True,'third'),(True,True))

    def test_interruptions_and_negative_frames_break_repeat_evidence(self):
        w=Window()
        for i in range(3):w.add(i*.5,True,str(i))
        self.assertEqual(w.add(10,True,'late'),(True,False))
        for i in range(5):w.add(10.5+i*.5,False,'negative'+str(i))
        self.assertEqual(w.add(13,True,'last'),(True,False))

    def test_csv_formula_escaping(self):
        for s in ('=1+1','+cmd','-1+2','@SUM(A1)','  =cmd'):
            self.assertTrue(csv_safe(s).startswith("'"))

    def test_offline_report_escapes_user_text(self):
        with tempfile.TemporaryDirectory() as td:
            data=dict(exported='2026-10-16',events=[],jobs=[dict(started='now',point='<script>alert(1)</script>',
                source_name='<img onerror=bad>',status='error',message='bad & text',model='baseline',threshold=.25,processed=0)])
            report=report_html(data,Store(td)).decode('utf-8')
            self.assertNotIn('<script>',report)
            self.assertIn('&lt;script&gt;',report)


class StoreTest(unittest.TestCase):
    def test_concurrent_job_reader_and_writer(self):
        with tempfile.TemporaryDirectory() as td:
            app=App(td);app.job=dict(id='concurrency',started='2026-10-16',status='running',processed=0)
            errors=[]
            def read():
                try:
                    for _ in range(150):app.jobs()
                except Exception as exc:errors.append(exc)
            reader=threading.Thread(target=read);reader.start()
            for i in range(100):app.update(processed=i)
            reader.join();self.assertEqual(errors,[])
            self.assertEqual(app.jobs()[0]['processed'],99)

    def test_detection_updates_preserve_review_and_evidence(self):
        # Exercise the real event writer without requiring a model or GPU.
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            app=App(td)
            app.job=dict(id='testjob',event_id=None,point='P01',condition='expected_dry',mode='file')
            app.model_meta=dict(label='test fixture',sha256='test-only')
            frame=np.zeros((64,64,3),dtype=np.uint8)
            boxes=[dict(xyxy=[1,2,10,20],confidence=.9,label='droplet')]
            w=Window();w.add(0,True,'a')
            app.record(frame,frame,boxes,0,False,w)
            key=app.job['event_id'];event=app.store.event(key)
            app.store.review(key,'dismissed','controlled normal flow',event['revision'])
            w.add(.5,True,'b');w.add(1,True,'c')
            lower=[dict(xyxy=[2,2,9,20],confidence=.6,label='droplet')]
            app.record(frame,frame,lower,1,True,w)
            after=app.store.event(key)
            self.assertEqual(after['review'],'dismissed')
            self.assertEqual(after['note'],'controlled normal flow')
            self.assertEqual(after['level'],'suspected')
            self.assertEqual(after['confidence'],.9)
            self.assertEqual(after['boxes'],boxes)
            self.assertEqual(len(after['observations']),2)
            self.assertEqual(len(after['reviews']),1)
            self.assertEqual(len(list((Path(td)/'evidence').glob('*.jpg'))),4)
            with self.assertRaises(ValueError):app.store.review(key,'confirmed','stale',0)
            reloaded=Store(td).event(key)
            self.assertEqual(reloaded['revision'],1)

    def test_restart_marks_incomplete_job(self):
        with tempfile.TemporaryDirectory() as td:
            app=App(td);path=Path(td)/'jobs'/'fixture.json'
            path.write_text(json.dumps(dict(id='fixture',status='running',started='2026-10-16')),encoding='utf-8')
            App(td)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'],'interrupted')


if __name__=='__main__':unittest.main()
