# -*- coding: utf-8 -*-
"""稳定版后端单测：不依赖 GPU，覆盖本次修复的能力探测/上传校验/路径校验/任务容错/停止语义。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import patrol
from patrol import App, DEFAULT_PORT, BACKEND_REVISION, evidence_path, sniff_media_kind, supported_modes


class StableBackendTests(unittest.TestCase):
    def test_single_source_port_and_revision(self):
        self.assertEqual(DEFAULT_PORT, 7862)
        self.assertEqual(BACKEND_REVISION, '20261019-stable-v1')

    def test_region_mode_follows_module_availability(self):
        """区域模式是否开放取决于时序模块是否存在：存在即开放，缺失则隐藏。"""
        modes = supported_modes()
        if patrol.region_available():
            self.assertIn('region', modes)
            self.assertEqual(modes, ['standard', 'detail', 'region', 'local'])
        else:
            self.assertNotIn('region', modes)
            self.assertEqual(modes, ['standard', 'detail', 'local'])

    def test_state_exposes_capabilities_for_ui(self):
        with tempfile.TemporaryDirectory() as td:
            state = App(td).state()
            self.assertEqual(state['service_port'], 7862)
            self.assertEqual(state['region_available'], patrol.region_available())
            self.assertEqual(state['backend_revision'], BACKEND_REVISION)
            self.assertIn('standard', state['supported_modes'])
            self.assertEqual('region' in state['supported_modes'], patrol.region_available())

    def test_start_rejects_region_when_unavailable(self):
        """仅在区域模块缺失时才应拒绝；模块齐全时不应误拒。"""
        if patrol.region_available():
            self.skipTest('区域模块已随包提供，本用例只覆盖缺失场景')
        with tempfile.TemporaryDirectory() as td:
            app = App(td)
            with self.assertRaisesRegex(ValueError, '不可用'):
                app.start(dict(mode='sample', point='P01', condition='unknown',
                               model='improved', detail='region', threshold=.25))

    def test_jobs_survives_corrupt_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as td:
            app = App(td)
            jobs = app.store.root / 'jobs'
            (jobs / 'good.json').write_text(json.dumps(dict(id='good', started='2026-10-19T00:00:00Z')), encoding='utf-8')
            (jobs / 'legacy.json').write_text(json.dumps(dict(id='legacy')), encoding='utf-8')
            (jobs / 'broken.json').write_text('{not json', encoding='utf-8')
            (jobs / 'good_timing.json').write_text('{}', encoding='utf-8')
            rows = app.jobs()
            self.assertEqual([r['id'] for r in rows], ['good', 'legacy'])

    def test_media_sniffing(self):
        self.assertEqual(sniff_media_kind(b'\xff\xd8\xff\xe0' + b'\x00' * 20), 'image')
        self.assertEqual(sniff_media_kind(b'\x89PNG\r\n\x1a\n' + b'\x00' * 20), 'image')
        self.assertEqual(sniff_media_kind(b'RIFF\x10\x00\x00\x00AVI ' + b'\x00' * 8), 'video')
        self.assertEqual(sniff_media_kind(b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 8), 'video')
        self.assertIsNone(sniff_media_kind(b'not an image at all'))

    def test_evidence_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'evidence').mkdir()
            ok = evidence_path(root, 'abc_detected.jpg')
            self.assertEqual(ok.parent, (root / 'evidence').resolve())
            for bad in ('../models.json', 'a/b.jpg', 'a\\b.jpg', '.hidden.jpg', 'x.png'):
                with self.assertRaises(ValueError):
                    evidence_path(root, bad)

    def test_stop_during_model_load_keeps_stopped_state(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            app = App(td)
            image = td / 'input.jpg'
            cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))[1].tofile(image)
            app.job = dict(id='stopcase', model='improved', event_id=None, point='p',
                           condition='unknown', mode='file', status='loading',
                           processed=0, positive_frames=0, duplicate_frames=0,
                           threshold=.25, detail='standard', imgsz=640, nms_iou=.5,
                           sample_interval=.5, frame_limit=1800)
            app.load_model = lambda key: dict(label='fixture', sha256='fixture', device='cpu',
                                              device_name='CPU fixture', precision='FP32')
            app.stop_event.set()
            app.run(image)
            self.assertEqual(app.job['status'], 'stopped')
            self.assertEqual(app.job['processed'], 0)
            self.assertIn('未开始观察', app.job['message'])


    def test_jobs_reuse_single_worker_thread(self):
        """连续任务复用同一个工作线程，避免 torch/OpenMP 线程池按任务累积。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            td = Path(td)
            image = td / 'input.jpg'
            cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))[1].tofile(image)
            app = App(td)

            def fake_load(key):
                meta = dict(label='fixture', sha256='fixture', device='cpu',
                            device_name='CPU fixture', precision='FP32')
                app.model_meta = meta
                return meta

            app.load_model = fake_load
            app.detect = lambda frame: (frame, [], 1.0)
            thread_ids = []
            for _ in range(3):
                app.start(dict(mode='file', point='p', condition='unknown', model='improved',
                               detail='standard', threshold=.25), source=image)
                for _ in range(300):
                    if app.job['status'] in ('completed', 'error', 'stopped'):
                        break
                    time.sleep(.02)
                self.assertEqual(app.job['status'], 'completed')
                time.sleep(.1)
                thread_ids.append(id(app._job_thread))
            self.assertEqual(len(set(thread_ids)), 1)
            self.assertEqual(app._job_thread.name, 'patrol-job')


if __name__ == '__main__':
    unittest.main()
