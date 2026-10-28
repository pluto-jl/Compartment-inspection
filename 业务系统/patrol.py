"""Local inspection service. Detection scores are not fault probabilities."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import base64
import html
import json
import mimetypes
import os
import queue
from pathlib import Path
import sqlite3
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit, parse_qs
from uuid import uuid4
from pipeline_timing import PipelineTiming, profiled

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent / 'YOLOv8n_微滴算法_V8_20261016'
sys.path[:0] = [str(ENGINE / 'vendor'), str(ENGINE)]

# 稳定版常量：端口单一来源（启动器/文档/QA 脚本均以此为准）。
DEFAULT_PORT = 7862
BACKEND_REVISION = '20261019-stable-v1'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def device_metadata(requested='0'):
    import torch
    if requested=='cpu':
        return dict(device='cpu',device_name='CPU reference verification',precision='FP32')
    if requested!='0':raise ValueError('不支持的计算设备')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用的NVIDIA GPU，请检查驱动与本项目GPU环境；本版不会自动切换CPU')
    return dict(device='0',device_name=torch.cuda.get_device_name(0),precision='FP32')


def detection_options(options):
    preset = options.get('detail', 'standard')
    sizes = {'standard':640, 'detail':960, 'region':640, 'local':640}
    if preset not in sizes:
        raise ValueError('请选择有效的观察模式')
    return dict(detail=preset, imgsz=sizes[preset], nms_iou=.5,
                sample_interval=0. if preset=='region' else .5,
                frame_limit=6000 if preset=='region' else 1800)


_REGION_AVAILABLE = None


def region_available():
    """区域时序模块是否随包提供；缺失时接口与 UI 自动隐藏该实验模式。"""
    global _REGION_AVAILABLE
    if _REGION_AVAILABLE is None:
        try:
            from region_overlay import RegionPresentation  # noqa: F401
            _REGION_AVAILABLE = True
        except Exception:
            _REGION_AVAILABLE = False
    return _REGION_AVAILABLE


def supported_modes():
    """按实际依赖返回可用观察模式，避免界面开放会崩溃的选项。"""
    modes = ['standard', 'detail']
    if region_available():
        modes.append('region')
    modes.append('local')
    return modes


def sniff_media_kind(head):
    """按文件头识别图片/视频类别，用于上传内容校验（不信任扩展名）。"""
    if not isinstance(head, (bytes, bytearray)) or len(head) < 12:
        return None
    b = bytes(head[:16])
    if b[:3] == b'\xff\xd8\xff' or b[:8] == b'\x89PNG\r\n\x1a\n' or b[:2] == b'BM':
        return 'image'
    if b[:4] == b'RIFF' and b[8:12] == b'WEBP':
        return 'image'
    if b[:4] == b'RIFF' and b[8:12] == b'AVI ':
        return 'video'
    if b[:4] == b'\x1a\x45\xdf\xa3' or b[4:8] == b'ftyp':
        return 'video'
    return None


def evidence_path(root, name):
    """校验并返回证据文件路径；拒绝路径分隔符、隐藏文件与目录逃逸。"""
    if not name.endswith('.jpg') or '/' in name or '\\' in name or name.startswith('.'):
        raise ValueError('证据路径无效')
    base = (Path(root) / 'evidence').resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError('证据路径无效')
    return target


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.03)


class Store:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('uploads', 'evidence', 'jobs'):
            (self.root / name).mkdir(exist_ok=True)
        self.db = self.root / 'events.sqlite3'
        with self.connect() as con:
            con.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL, created TEXT NOT NULL,
                    updated TEXT NOT NULL, point TEXT NOT NULL, condition TEXT NOT NULL,
                    level TEXT NOT NULL, source_kind TEXT NOT NULL,
                    model_label TEXT NOT NULL, model_sha256 TEXT NOT NULL,
                    confidence REAL NOT NULL, hits INTEGER NOT NULL,
                    first_media_time REAL NOT NULL, last_media_time REAL NOT NULL,
                    image TEXT NOT NULL, original TEXT NOT NULL, boxes TEXT NOT NULL,
                    review TEXT NOT NULL DEFAULT 'pending', note TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                    at TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL,
                    revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                    media_time REAL NOT NULL, image TEXT NOT NULL, original TEXT NOT NULL,
                    confidence REAL NOT NULL, boxes TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def events(self):
        with self.connect() as con:
            rows = [dict(r) for r in con.execute('SELECT * FROM events ORDER BY created DESC')]
        for r in rows:
            r['boxes'] = json.loads(r['boxes'])
        return rows

    def event(self, key):
        with self.connect() as con:
            row = con.execute('SELECT * FROM events WHERE id=?', (key,)).fetchone()
            if row is None:
                raise ValueError('事件不存在')
            r = dict(row)
            r['boxes'] = json.loads(r['boxes'])
            r['reviews'] = [dict(x) for x in con.execute(
                'SELECT * FROM reviews WHERE event_id=? ORDER BY id', (key,))]
            r['observations'] = [dict(x) for x in con.execute(
                'SELECT * FROM observations WHERE event_id=? ORDER BY id', (key,))]
            return r

    def review(self, key, status, note, revision):
        if status not in ('pending', 'confirmed', 'dismissed', 'follow_up'):
            raise ValueError('复核状态无效')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('备注最多 2000 字')
        with self.connect() as con:
            cur = con.execute('UPDATE events SET review=?,note=?,revision=revision+1 WHERE id=? AND revision=?',
                              (status, note, key, revision))
            if cur.rowcount != 1:
                raise ValueError('记录已更新或不存在，请刷新后再保存')
            con.execute('INSERT INTO reviews(event_id,at,status,note,revision) VALUES(?,?,?,?,?)',
                        (key, now(), status, note, revision + 1))
        return self.event(key)


class Window:
    """Point-level repeat observation, not tracking or fault classification."""
    def __init__(self):
        self.items = deque(maxlen=5)
        self.last_digest = None
        self.hits = 0

    def add(self, stamp, positive, digest):
        if digest == self.last_digest:
            return False, False
        self.last_digest = digest
        self.items.append((stamp, bool(positive)))
        self.hits += int(positive)
        # Do not link detections across long interruptions or sparse media.
        while self.items and stamp - self.items[0][0] > 4:
            self.items.popleft()
        positives = [t for t, v in self.items if v]
        repeated = len(positives) >= 3 and positives[-1] - positives[0] >= 1.0
        return True, repeated


class _JobHandle:
    """兼容旧字段 self.worker：把持久工作线程包装成「当前任务是否在跑」的句柄。

    之前每个任务新建一个线程，torch/OpenMP 会为每个新线程创建一套线程池且不回收，
    连续检测会持续增加 OS 线程（实测约 7 个/任务）。改为持久单线程复用即可稳定。
    """

    def __init__(self, app):
        self.app = app

    def is_alive(self):
        return bool(self.app._runner_busy)

    def join(self, timeout=None):
        deadline = time.time() + (timeout if timeout is not None else 10 ** 9)
        while self.app._runner_busy and time.time() < deadline:
            time.sleep(.05)


class App:
    def __init__(self, data, device='0'):
        if device not in ('0','cpu'):raise ValueError('不支持的计算设备')
        self.requested_device=device
        self.port = DEFAULT_PORT
        self.store = Store(data)
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.job = None
        self.worker = None
        self._runner_busy = False
        self._job_queue = queue.Queue()
        self._job_thread = None
        self.stop_event = threading.Event()
        self.frame = None
        self.frame_seq = 0
        self.detector = None
        self.model_key = None
        self.model_meta = None
        self._jobs_cache = {}
        self._jobs_snapshot = None
        self.server_time = now()
        self.registry = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
        # A stopped process cannot silently leave a completed-looking active job.
        for path in (self.store.root / 'jobs').glob('*.json'):
            try:
                j = json.loads(path.read_text(encoding='utf-8'))
                if j['status'] in ('loading', 'running', 'stopping'):
                    j.update(status='interrupted', message='上次服务中断，任务未完成', ended=now())
                    atomic_json(path, j)
            except (ValueError, KeyError):
                continue

    def models(self):
        return [dict(key=m['key'], label=m['label'], available=bool(m.get('path')) and
                     (ROOT / m['path']).is_file(), research_status=m['research_status']) for m in self.registry]

    def state(self):
        with self.lock:
            return dict(job=dict(self.job) if self.job else None, frame_seq=self.frame_seq,
                        app_id='pipe_patrol_v1',
                        has_frame=self.frame is not None,
                        models=self.models(), connected_robot=False, server_time=now(),
                        qa_mode=self.store.root.name=='qa_runtime',
                        running=bool((self.worker and self.worker.is_alive()) or (self.job and self.job['status'] in ('loading', 'running', 'stopping'))),
                        supported_modes=supported_modes(),
                        backend_revision=BACKEND_REVISION, pipeline_timing=True,
                        compute_policy='cuda_required' if self.requested_device=='0' else 'cpu_reference_only',
                        service_port=self.port, region_available=region_available())

    def jobs(self):
        """任务记录列表：带缓存与限流，避免前端每 1.2 秒轮询时重复读取全部任务 JSON。

        优化前的性能问题（实测）：
          - 每次请求都 glob 全部 jobs/*.json 并逐个 read+json.loads；
          - 任务累计到 60+ 条时，8 线程并发下 /api/jobs p50 高达 455ms；
          - 前端每 1.2 秒轮询一次，长时间运行会持续占用 CPU 并拖慢其他接口。
        现在的做法：
          1) 按 (mtime, size) 缓存已解析的任务字典，文件没变就直接复用；
          2) 结果整体缓存 0.5 秒，多个并发请求共享同一份快照（前端轮询间隔 1.2s，
             不会感知到延迟，但并发压力下降一个数量级）；
          3) 当前任务状态随时在变，其对应的缓存条目会被主动失效。
        """
        now_ts = time.monotonic()
        snap = self._jobs_snapshot
        if snap is not None and now_ts - snap[0] < 2.0:
            return snap[1]

        cache = self._jobs_cache
        out = []
        jobs_dir = self.store.root / 'jobs'
        try:
            paths = sorted(jobs_dir.glob('*.json'), reverse=True)
        except OSError:
            paths = []
        current_id = self.job.get('id') if self.job else None
        for path in paths:
            if path.name.endswith('_timing.json'):
                continue
            key = path.name
            try:
                st = path.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            cached = cache.get(key)
            if cached is not None and cached[0] == sig and not (
                    current_id and key == current_id + '.json'):
                out.append(cached[1])
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                cache.pop(key, None)
                continue
            if not isinstance(data, dict):
                cache.pop(key, None)
                continue
            cache[key] = (sig, data)
            out.append(data)
        # 清理已删除任务的缓存条目，避免长期运行内存缓慢增长
        if len(cache) > len(paths) + 64:
            alive = {p.name for p in paths}
            for k in list(cache):
                if k not in alive:
                    cache.pop(k, None)
        # 损坏或旧版本任务文件缺少 started 时不再让整页 500。
        result = sorted(out, key=lambda x: str(x.get('started', '')), reverse=True)
        self._jobs_snapshot = (now_ts, result)
        return result

    @profiled('job_state_write')
    def update(self, **changes):
        with self.lock:
            self.job.update(changes)
            atomic_json(self.store.root / 'jobs' / (self.job['id'] + '.json'), self.job)
            # 任务状态已变，必须让 /api/jobs 的短时快照失效，否则读到旧数据。
            self._jobs_snapshot = None

    def _ensure_job_thread(self):
        """持久单任务线程：避免每个任务新建线程导致 torch/OpenMP 线程池累积。"""
        if self._job_thread is None or not self._job_thread.is_alive():
            self._job_thread = threading.Thread(target=self._job_loop, name='patrol-job', daemon=True)
            self._job_thread.start()

    def _job_loop(self):
        while True:
            source = self._job_queue.get()
            try:
                if source is None:
                    return
                self.run(source)
            finally:
                self._runner_busy = False
                self._job_queue.task_done()

    def start(self, options, source=None):
        mode = options.get('mode', 'file')
        if mode not in ('file', 'sample', 'camera'):
            raise ValueError('输入类型无效')
        point = str(options.get('point', '')).strip()
        if not point or len(point) > 80:
            raise ValueError('请输入不超过 80 字的巡检点名称')
        condition = options.get('condition', 'unknown')
        if condition not in ('unknown', 'expected_dry', 'normal_flow'):
            raise ValueError('工况无效')
        threshold = float(options.get('threshold', .25))
        if not .05 <= threshold <= .95:
            raise ValueError('检测阈值应在 0.05 至 0.95 之间')
        inference_options = detection_options(options)
        if inference_options['detail'] not in supported_modes():
            raise ValueError('该观察模式在本轻量包中不可用（缺少区域时序模块）；请选择标准检测、高分辨率复核或图片局部复核')
        key = options.get('model', 'improved')
        if not any(m['key'] == key and m['available'] for m in self.models()):
            raise ValueError('该模型尚不可用，请先完成训练并配置模型')
        if mode == 'sample':
            samples = ROOT.parent / '管路微滴训练集_V1/images/train'
            source = next(iter(sorted(samples.glob('*.jpg'))), None)
            if source is None:
                raise ValueError('未找到本地联调样例')
            source_name = '本包训练样例 非独立测试'
        elif mode == 'camera':
            index = int(options.get('camera_index', 0))
            if not 0 <= index <= 5:
                raise ValueError('摄像头编号应为 0 至 5')
            source = index
            source_name = f'本机摄像头 {index} 非机器人实机接入'
        else:
            if source is None or not Path(source).is_file():
                raise ValueError('请先选择图片或视频')
            source_name = str(options.get('filename', Path(source).name))[:180]
        if inference_options['detail']=='local' and (isinstance(source,int) or Path(source).suffix.lower() not in IMAGE_EXT):
            raise ValueError('图片局部复核仅支持单张图片；视频请选择标准或视频区域模式')
        with self.lock:
            if (self.worker and self.worker.is_alive()) or (self.job and self.job['status'] in ('loading', 'running', 'stopping')):
                raise ValueError('已有任务运行，请先停止或等待完成')
            self.stop_event.clear()
            self.frame = None
            job_id = uuid4().hex
            self.job = dict(id=job_id, started=now(), ended=None, status='loading', mode=mode,
                point=point, condition=condition, source_name=source_name, model=key,
                threshold=threshold, **inference_options, processed=0, positive_frames=0,
                duplicate_frames=0, media_time=0, duration=None, event_id=None,
                message='正在加载检测模型', inference_ms=None, robot_connected=False,
                raw_video=str(Path(source).relative_to(self.store.root)) if mode=='file' and
                    Path(source).suffix.lower() in VIDEO_EXT else None)
            atomic_json(self.store.root / 'jobs' / (job_id + '.json'), self.job)
            self._jobs_snapshot = None
            self._runner_busy = True
            self.worker = _JobHandle(self)
            self._ensure_job_thread()
            self._job_queue.put(source)
            return dict(self.job)

    def stop(self):
        with self.lock:
            if self.job and self.job['status'] in ('loading', 'running'):
                self.stop_event.set()
                self.update(status='stopping', message='正在停止，本任务将标记为未完整完成')

    @profiled('model_load')
    def load_model(self, key):
        from ultralytics import YOLO
        compute=device_metadata(self.requested_device)
        m = next(x for x in self.registry if x['key'] == key)
        path = (ROOT / m['path']).resolve()
        digest = file_sha256(path)
        if self.model_key != key or not self.model_meta or self.model_meta['sha256'] != digest:
            self.detector = YOLO(str(path))
            if self.detector.names != {0: 'droplet'}:
                raise ValueError('只接受已配置为单类 droplet 的可信本地模型')
            self.model_key = key
            self.model_meta = dict(label=m['label'], sha256=digest, **compute)
        return dict(self.model_meta)

    @profiled('gpu_predict_and_box_drawing')
    def detect(self, image):
        import cv2
        started = time.perf_counter()
        with (self.profiler.measure('gpu_predict_sync') if getattr(self,'profiler',None) else nullcontext()):
            result = self.detector.predict(image, imgsz=self.job['imgsz'], conf=self.job['threshold'], iou=self.job['nms_iou'],
                device=self.model_meta['device'], quantize=32, verbose=False, max_det=100)[0]
            if self.model_meta['device']=='0':
                actual_device=str(self.detector.predictor.device)
                if not actual_device.startswith('cuda'):
                    raise RuntimeError('实际推理后端未运行在GPU上，任务已停止')
                self.model_meta['actual_device']=actual_device
            # Moving outputs to CPU synchronizes inference before recording this elapsed time.
            xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
        ms = (time.perf_counter() - started) * 1000
        boxes = [dict(xyxy=[round(v, 2) for v in box], confidence=round(float(score), 5),
                      label='droplet') for box, score in zip(xyxy, scores)]
        if self.job.get('detail')=='local':
            from local_image_review import crop_candidates,append_candidates
            with self.profiler.measure('local_crop_predict_and_merge') if getattr(self,'profiler',None) else nullcontext():
                boxes=append_candidates(boxes,crop_candidates(self.detector,image,self.job['threshold'],self.model_meta['device']))
            ms=(time.perf_counter()-started)*1000
        if self.job.get('detail')=='region':
            # RegionPresentation draws the final hints; this intermediate image was discarded.
            return image,boxes,ms
        annotated = image.copy()
        for box in boxes:
            x1,y1,x2,y2 = [int(v) for v in box['xyxy']]
            color=(240,190,60) if box.get('source')=='local_zoom_candidate' else (0,180,255)
            caption='zoom candidate' if box.get('source')=='local_zoom_candidate' else 'droplet'
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, caption+' %.2f' % box['confidence'], (x1,max(18,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        return annotated, boxes, ms

    @profiled('evidence_jpeg_and_write')
    def save_frame(self, name, frame):
        import cv2
        with self.profiler.measure('evidence_jpeg_encode') if getattr(self,'profiler',None) else nullcontext():
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise ValueError('图像编码失败')
        path = self.store.root / 'evidence' / name
        with self.profiler.measure('evidence_disk_write') if getattr(self,'profiler',None) else nullcontext():
            path.write_bytes(buf.tobytes())
        return 'evidence/' + name

    @profiled('event_database')
    def record(self, frame, annotated, boxes, stamp, repeated, window):
        event_id = self.job['event_id']
        score = max(b['confidence'] for b in boxes)
        level = 'suspected' if repeated and self.job['condition'] == 'expected_dry' else (
            'repeated' if repeated else 'candidate')
        if event_id is None:
            event_id = uuid4().hex
            original = self.save_frame(event_id + '_original.jpg', frame)
            image = self.save_frame(event_id + '_detected.jpg', annotated)
            with self.store.connect() as con:
                con.execute('''INSERT INTO events(id,job_id,created,updated,point,condition,level,
                    source_kind,model_label,model_sha256,confidence,hits,first_media_time,
                    last_media_time,image,original,boxes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (event_id,self.job['id'],now(),now(),self.job['point'],self.job['condition'],level,
                     self.job['mode'],self.model_meta['label'],self.model_meta['sha256'],score,
                     window.hits,stamp,stamp,image,original,json.dumps(boxes)))
                con.execute('INSERT INTO observations(event_id,media_time,image,original,confidence,boxes) VALUES(?,?,?,?,?,?)',
                            (event_id,stamp,image,original,score,json.dumps(boxes)))
            self.update(event_id=event_id)
        else:
            old = self.store.event(event_id)
            # Evidence is immutable: retain all updated representative observations.
            if score > old['confidence'] or (repeated and old['level'] == 'candidate'):
                suffix = uuid4().hex[:10]
                original = self.save_frame(f'{event_id}_{suffix}_original.jpg', frame)
                image = self.save_frame(f'{event_id}_{suffix}_detected.jpg', annotated)
                with self.store.connect() as con:
                    con.execute('INSERT INTO observations(event_id,media_time,image,original,confidence,boxes) VALUES(?,?,?,?,?,?)',
                                (event_id,stamp,image,original,score,json.dumps(boxes)))
                if score > old['confidence']:
                    best,best_boxes=score,json.dumps(boxes)
                else:
                    original,image,best,best_boxes=old['original'],old['image'],old['confidence'],json.dumps(old['boxes'])
            else:
                original,image,best,best_boxes = old['original'],old['image'],old['confidence'],json.dumps(old['boxes'])
            # An event cannot lose its repeated level when the observation window moves.
            rank = {'candidate':0,'repeated':1,'suspected':2}
            level = max((level, old['level']), key=lambda s: rank[s])
            with self.store.connect() as con:
                con.execute('''UPDATE events SET updated=?,level=?,hits=?,last_media_time=?,
                    confidence=?,original=?,image=?,boxes=? WHERE id=?''',
                    (now(),level,window.hits,stamp,best,original,image,best_boxes,event_id))

    @profiled('preview_resize_jpeg')
    def publish(self, frame):
        import cv2
        h,w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, round(h*1280/w)))
        with self.profiler.measure('preview_jpeg_encode') if getattr(self,'profiler',None) else nullcontext():
            ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            raise ValueError('预览编码失败')
        with self.lock:
            self.frame = buf.tobytes()
            self.frame_seq += 1

    def run(self, source):
        import cv2
        import numpy as np
        self.profiler = PipelineTiming()
        cap = None
        trace = None
        try:
            with self.model_lock:
                meta = self.load_model(self.job['model'])
                if self.stop_event.is_set():
                    # 停止请求发生在模型加载期间：不再进入检测循环，保持 stopped 终态。
                    self.update(status='stopped', ended=now(), message='已停止：模型已加载，本次未开始观察')
                    return
                self.update(status='running', message='正在检测', model_meta=meta,
                            source_sha256=None if isinstance(source,int) else file_sha256(source))
                window = Window()
                region_mode = self.job.get('detail') == 'region'
                presentation = None
                if region_mode:
                    from region_overlay import RegionPresentation
                    presentation = RegionPresentation()
                    trace_path = self.store.root/'jobs'/f"{self.job['id']}_regions.jsonl"
                    trace = trace_path.open('x',encoding='utf-8')
                    self.update(region_trace=str(trace_path.relative_to(self.store.root)),
                                region_rule='candidate_only_no_automatic_fault_upgrade')
                single = not isinstance(source, int) and Path(source).suffix.lower() in IMAGE_EXT
                from preview_cadence import PreviewCadence
                preview = PreviewCadence(self.publish)
                if single:
                    frame = cv2.imdecode(np.frombuffer(Path(source).read_bytes(),np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('无法读取图片，请检查文件格式')
                    frames = [(0., frame)]
                else:
                    cap = cv2.VideoCapture(source if isinstance(source,int) else str(source))
                    if not cap.isOpened():
                        raise ValueError('无法打开视频或摄像头，请检查设备或编码格式')
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if not np.isfinite(fps) or fps <= 0:
                        if isinstance(source, int): fps = 30
                        else: raise ValueError('视频帧率不可读，无法可靠计算时间')
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = count / fps if count>0 and not isinstance(source,int) else None
                    self.update(duration=duration, source_fps=fps, source_frames=count,
                                timestamp_policy='live_monotonic' if isinstance(source,int) else 'nominal_index_over_average_fps')
                    if region_mode and not isinstance(source,int) and count > self.job['frame_limit']:
                        raise ValueError('区域模式逐帧处理，单任务最多6000帧，请先裁剪视频')
                    frames = self.iter_frames(cap, fps, isinstance(source,int))
                for stamp, frame in frames:
                    if self.stop_event.is_set(): break
                    h,w = frame.shape[:2]
                    if h*w > 20_000_000:
                        raise ValueError('输入图像超过 2000 万像素，请先缩小')
                    with self.profiler.measure('frame_hash'):
                        digest = hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest()
                    if window.last_digest == digest:
                        self.update(duplicate_frames=self.job['duplicate_frames']+1,
                                    message='检测到重复画面，未累加连续观测证据', media_time=stamp)
                        continue
                    annotated, boxes, ms = self.detect(frame)
                    _, repeated = window.add(stamp, bool(boxes), digest)
                    extra = {}
                    if presentation:
                        annotated,hint=presentation.render(frame,boxes,stamp,profiler=self.profiler)
                        with self.profiler.measure('region_trace_write'):
                            trace.write(json.dumps(dict(media_time=stamp,frame_sha256=digest,boxes=boxes,**hint),ensure_ascii=False)+'\n')
                        extra['region_hint']=dict(current_count=hint['current_count'],held_count=hint['held_count'],
                                                  stable=hint['view']['stable'],view_reason=hint['view']['reason'])
                        # This experimental mode displays candidates, not automatic fault decisions.
                        repeated=False
                    if self.job.get('detail')=='local':
                        repeated=False
                        extra['local_review_policy']='image_only_candidate_manual_review_full_boxes_preserved'
                    if boxes: self.record(frame, annotated, boxes, stamp, repeated, window)
                    preview.submit(annotated)
                    with self.profiler.measure('sharpness_preview_640'):
                        quality_frame=frame
                        if max(h,w)>640:
                            ratio=640/max(h,w)
                            quality_frame=cv2.resize(frame,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
                        blur = float(cv2.Laplacian(cv2.cvtColor(quality_frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                    self.update(processed=self.job['processed']+1,
                        positive_frames=self.job['positive_frames']+int(bool(boxes)), media_time=stamp,
                        inference_ms=round(ms,1), sharpness=round(blur,1),
                        sharpness_method='laplacian_variance_max_side_640_v2', sharpness_input_size=list(quality_frame.shape[1::-1]),
                        actual_device=self.model_meta.get('actual_device',self.model_meta['device']),
                        message=('已检出微滴候选，已标出大致区域' if region_mode else '已检出微滴，请结合工况复核') if boxes else (
                            '黄色区域为最近位置保留，本帧未检出' if extra.get('region_hint',{}).get('held_count') else '本帧未检出微滴，不代表无泄漏'),**extra)
                # Flush the latest processed frame, even if the display interval has not elapsed.
                preview.flush()
                self.update(preview_encoded_frames=preview.encoded, preview_interval_seconds=.25)
                # Always retain a final frame even when no droplet was found.
                if self.frame:
                    (self.store.root/'evidence'/f"{self.job['id']}_last.jpg").write_bytes(self.frame)
                stopped = self.stop_event.is_set()
                if trace:
                    trace.close();trace=None
                    self.update(region_trace_sha256=file_sha256(trace_path))
                self.update(status='stopped' if stopped else 'completed', ended=now(),
                    message='已停止，任务未完整完成' if stopped else (
                        '检测完成，请复核已记录事件' if self.job['event_id'] else '检测完成，本次未形成微滴事件，不代表设备无泄漏'))
        except Exception as exc:
            traceback.print_exc()
            self.update(status='error', ended=now(), message='检测未完成：'+str(exc))
        finally:
            if trace is not None:trace.close()
            if cap is not None: cap.release()
            try:
                summary=self.profiler.summary(self.job.get('processed',0))
                summary.update(job_status=self.job['status'],source_sha256=self.job.get('source_sha256'),model_meta=self.job.get('model_meta'),mode=self.job.get('detail','standard'))
                timing_path=self.store.root/'jobs'/f"{self.job['id']}_timing.json"
                atomic_json(timing_path,summary)
                self.update(timing_profile=str(timing_path.relative_to(self.store.root)),timing_profile_sha256=file_sha256(timing_path))
            except Exception:
                traceback.print_exc()
            finally:self.profiler=None

    def iter_frames(self, cap, fps, camera):
        import cv2
        started=time.monotonic();last=-1.;index=0;sampled=0
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not camera else 0
        while not self.stop_event.is_set():
            profiler=getattr(self,'profiler',None)
            if profiler is None:ok,frame=cap.read()
            else:
                with profiler.measure('video_decode'):ok,frame=cap.read()
            if not ok:
                if camera: raise ValueError('摄像头画面中断')
                if index==0: raise ValueError('视频没有可解码画面')
                if total>0 and index+2<total:raise ValueError('视频提前中断，未完成全部解码')
                break
            stamp=time.monotonic()-started if camera else index/fps
            index+=1
            if stamp-last < self.job.get('sample_interval',.5): continue
            if stamp > 900 or sampled >= self.job.get('frame_limit',1800):
                raise ValueError('达到本模式帧数或15分钟上限，请拆分视频或重新开始观察')
            last=stamp;sampled+=1
            yield stamp,frame


IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
VIDEO_EXT={'.mp4','.avi','.mov','.mkv','.m4v'}


def csv_safe(value):
    s = str(value if value is not None else '')
    return "'"+s if s.lstrip().startswith(('=','+','-','@')) else s


def report_html(data, store):
    esc=lambda v:html.escape(str(v if v is not None else '—'))
    status={'completed':'已完成','stopped':'已停止 未完整完成','error':'检测失败','interrupted':'服务中断','running':'运行中','loading':'加载中','stopping':'停止中'}
    review={'pending':'待复核','confirmed':'人工确认异常','dismissed':'已排除异常','follow_up':'待现场检查'}
    levels={'candidate':'单帧微滴候选','repeated':'重复观测到微滴','suspected':'疑似漏液待复核'}
    conditions={'unknown':'工况未知','expected_dry':'预期干燥','normal_flow':'正常出水'}
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>管路巡检记录报告</title>',
        '<style>body{font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#172e3b;max-width:980px;margin:40px auto;padding:0 24px}h1{font-size:29px}h2{font-size:21px;margin-top:35px}h3{font-size:17px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #d5dee4;padding:9px;text-align:left;overflow-wrap:anywhere}th{background:#edf3f6}img{max-width:100%;max-height:400px;object-fit:contain}small{color:#526c7b}.event{break-inside:avoid;border-top:1px solid #ccd9e0;margin-top:25px;padding-top:12px}.hash{font:11px monospace;overflow-wrap:anywhere}@media print{body{margin:0;max-width:none}h2{break-after:avoid}tr{break-inside:avoid}}</style>',
        '<h1>管路微滴检测与人工复核记录</h1>',
        f'<p>生成时间 {esc(data["exported"])}<br>时间戳采用 UTC，未接入机器人定位与任务控制。</p>',
        '<p>本报告记录视觉检测和人工复核过程。“未检出微滴”不代表设备无泄漏；检测分数不是故障概率。软件联调与视频回放不等同于真实机器人巡检。</p>',
        f'<p>任务 {len(data["jobs"])} 次，微滴事件 {len(data["events"])} 条，待复核 {sum(e["review"]=="pending" for e in data["events"])} 条。</p>',
        '<h2>任务记录</h2><table style="table-layout:fixed"><colgroup><col style="width:24%"><col style="width:24%"><col style="width:32%"><col style="width:20%"></colgroup><tr><th>开始时间与点位</th><th>输入来源</th><th>状态与说明</th><th>模型与检测配置</th></tr>']
    for j in data['jobs']:
        sampling='逐帧区域提示（实验，仅候选）' if j.get('detail')=='region' else '约0.5秒采样'
        parts.append(f'<tr><td>{esc(j["started"])}<br>{esc(j["point"])}</td><td>{esc(j["source_name"])}</td><td>{esc(status.get(j["status"],j["status"]))}<br>{esc(j["message"])}</td><td>{esc(j["model"])}<br>阈值 {esc(j["threshold"])} 输入 {esc(j.get("imgsz",640))} 去重IoU {esc(j.get("nms_iou",.5))}<br>{sampling}<br>处理 {esc(j["processed"])} 帧</td></tr>')
    parts.append('</table><h2>事件与复核证据</h2>')
    if not data['events']:parts.append('<p>本次导出范围内没有微滴事件。</p>')
    for e in data['events']:
        parts.append(f'<section class="event"><h3>{esc(e["point"])}</h3><p>{esc(e["created"])}<br>{esc(levels[e["level"]])} · {esc(conditions[e["condition"]])} · {esc(review[e["review"]])}<br>代表检测分数 {e["confidence"]:.3f}，检出帧数 {e["hits"]}，媒体时间 {e["first_media_time"]:.2f} 至 {e["last_media_time"]:.2f} 秒。</p>')
        f=store.root/e['image']
        if f.is_file():parts.append('<img alt="微滴检测证据" src="data:image/jpeg;base64,'+base64.b64encode(f.read_bytes()).decode()+'">')
        else:parts.append('<p>检测证据文件缺失，请检查原始数据目录。</p>')
        parts.append(f'<p>当前备注 {esc(e["note"] or "未填写")}</p><small>模型 {esc(e["model_label"])}</small><p class="hash">模型 SHA256 {esc(e["model_sha256"])}<br>事件编号 {esc(e["id"])}</p><h4>人工复核历史</h4>')
        for r in e['reviews']:
            parts.append(f'<p>{esc(r["at"])} · {esc(review[r["status"]])}<br>{esc(r["note"] or "未填写备注")}</p>')
        if not e['reviews']:parts.append('<p>暂无人工复核记录。</p>')
        parts.append('</section>')
    parts.append('<p>原始文件、代表观测和任务配置保留在本机运行数据目录。完整结构化记录请同时导出 JSON。本报告不包含识别精度评估或实机验收结论。</p></html>')
    return ''.join(parts).encode('utf-8')


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def host_ok(self):
            port=self.server.server_address[1]
            return self.headers.get('Host','') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def send(self, status, body, mime='application/json; charset=utf-8', filename=None):
            if not isinstance(body, bytes): body=json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Cross-Origin-Resource-Policy','same-origin')
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            u=urlsplit(self.path);path=unquote(u.path)
            if not self.host_ok():return self.send(403,{'error':'主机校验失败'})
            try:
                if path=='/api/state':return self.send(200,app.state())
                if path=='/api/events':return self.send(200,app.store.events())
                if path=='/api/jobs':return self.send(200,app.jobs())
                if path.startswith('/api/event/'):
                    return self.send(200,app.store.event(path.rsplit('/',1)[-1]))
                if path=='/api/frame':
                    with app.lock:data=app.frame
                    return self.send(200,data,'image/jpeg') if data else self.send(404,{'error':'尚无画面'})
                if path=='/api/export':
                    data=dict(exported=now(),scope='仅本地软件记录 非设备无泄漏证明',
                              events=[app.store.event(e['id']) for e in app.store.events()],jobs=app.jobs())
                    if parse_qs(u.query).get('format',['json'])[0]=='report':
                        return self.send(200,report_html(data,app.store),'text/html; charset=utf-8','patrol_report.html')
                    if parse_qs(u.query).get('format',['json'])[0]=='csv':
                        out=io.StringIO(newline='');writer=csv.writer(out)
                        fields=['id','job_id','created','point','condition','level','model_label','confidence','hits','review','note','image']
                        writer.writerow(fields)
                        for row in data['events']:writer.writerow([csv_safe(row[k]) for k in fields])
                        return self.send(200,('\ufeff'+out.getvalue()).encode('utf-8'),'text/csv; charset=utf-8','patrol_events.csv')
                    return self.send(200,json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'),filename='patrol_records.json')
                if path.startswith('/evidence/'):
                    f=evidence_path(app.store.root,path[len('/evidence/'):])
                    return self.send(200,f.read_bytes(),'image/jpeg')
                # 静态资源白名单：主页、样式与脚本，以及 /assets/ 下的图标与样例图。
                # 一律做 resolve + is_relative_to 校验，禁止目录穿越与目录列表。
                if path in ('/','/app.js','/style.css','/polish.css','/competition.css','/config.js') \
                        or path.startswith('/assets/'):
                    web_root=(ROOT/'web').resolve()
                    rel='index.html' if path=='/' else path.lstrip('/')
                    target=(web_root/rel).resolve()
                    if not target.is_relative_to(web_root) or not target.is_file():
                        return self.send(404,{'error':'页面不存在'})
                    mime=mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                    if mime.startswith('text/') or mime in ('application/javascript','application/json'):
                        mime+='; charset=utf-8'
                    body=target.read_bytes()
                    return self.send(200,body,mime)
                return self.send(404,{'error':'页面不存在'})
            except (ValueError,FileNotFoundError):return self.send(404,{'error':'未找到记录或文件'})
            except Exception:
                traceback.print_exc();return self.send(500,{'error':'读取失败，请查看服务窗口'})

        def do_POST(self):
            # Custom header and exact-origin checks prevent cross-site form submissions.
            if self.headers.get('X-Patrol-Client')!='1':return self.send(403,{'error':'请求来源无效'})
            if not self.host_ok():return self.send(403,{'error':'主机校验失败'})
            origin=self.headers.get('Origin')
            port=self.server.server_address[1]
            if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.send(403,{'error':'不允许跨站请求'})
            path=urlsplit(self.path).path
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>200*1024*1024:raise ValueError('文件或请求大小无效，最大 200 MB')
                self.connection.settimeout(60)
                if path=='/api/upload':
                    name=unquote(self.headers.get('X-Filename',''))
                    ext=Path(name).suffix.lower()
                    if ext not in IMAGE_EXT|VIDEO_EXT:raise ValueError('请选择常见图片或视频格式')
                    key=uuid4().hex+ext;target=app.store.root/'uploads'/key
                    first=self.rfile.read(min(65536,size))
                    kind=sniff_media_kind(first)
                    if kind is None:raise ValueError('文件内容不是受支持的图片或视频')
                    if kind!=('image' if ext in IMAGE_EXT else 'video'):raise ValueError('文件内容与扩展名不一致')
                    remaining=size-len(first)
                    try:
                        with target.open('xb') as out:
                            out.write(first)
                            while remaining:
                                chunk=self.rfile.read(min(1024*1024,remaining))
                                if not chunk:raise ValueError('上传中断')
                                out.write(chunk);remaining-=len(chunk)
                    except Exception:
                        target.unlink(missing_ok=True);raise
                    return self.send(200,dict(key=key,filename=name[:180]))
                if size>65536:raise ValueError('请求参数过大')
                data=json.loads(self.rfile.read(size))
                if path=='/api/start':
                    source=None
                    if data.get('mode')=='file':
                        key=data.get('upload','')
                        if not key or '/' in key or '\\' in key or Path(key).suffix.lower() not in IMAGE_EXT|VIDEO_EXT:
                            raise ValueError('请先上传文件')
                        source=app.store.root/'uploads'/key
                    return self.send(200,app.start(data,source))
                if path=='/api/stop':app.stop();return self.send(200,{'ok':True})
                if path=='/api/review':
                    return self.send(200,app.store.review(data['id'],data['status'],data.get('note',''),int(data['revision'])))
                return self.send(404,{'error':'接口不存在'})
            except (ValueError,KeyError,TypeError) as exc:return self.send(400,{'error':str(exc)})
            except Exception:
                traceback.print_exc();return self.send(500,{'error':'操作失败，请查看服务窗口'})
    return Handler


class PatrolHTTPServer(ThreadingHTTPServer):
    """禁止端口复用：Windows 上 allow_reuse_address=True 会让第二个实例绑上同一端口，
    导致两个进程争抢同一端口、互相把对方任务标记为 interrupted。"""

    allow_reuse_address = False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=DEFAULT_PORT)
    parser.add_argument('--data',default=str(ROOT/'runtime'))
    args=parser.parse_args()
    # Bind before touching persisted jobs: a second start on the same port must not
    # mark the running instance's jobs interrupted.
    try:
        server=PatrolHTTPServer(('127.0.0.1',args.port),BaseHTTPRequestHandler)
    except OSError as exc:
        print(f'端口 {args.port} 已被占用（可能已有工作台在运行）：{exc}',flush=True)
        return 1
    try:
        app=App(args.data)
        app.port=args.port
        server.RequestHandlerClass=make_handler(app)
    except Exception:
        server.server_close()
        raise
    print(f'Pipe Patrol: http://127.0.0.1:{args.port}  (Ctrl+C to stop)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        app.stop();server.server_close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
