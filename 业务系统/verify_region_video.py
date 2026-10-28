"""Real A3 video integration check, isolated from user history."""
import json
import tempfile
import time
import argparse
from pathlib import Path
from patrol import App,ROOT,ENGINE,detection_options,file_sha256


def main(device='0'):
    import torch
    torch.set_num_threads(2)
    source=ROOT.parent/'管路微滴训练集_V1/视频素材/Pexels_5698926_演示保留.mp4'
    cache=ENGINE/'runs/当前统一数据试训/视频对比结果/门控修正/逐帧结果.json'
    expected=json.loads(cache.read_text(encoding='utf-8'))
    original_sha=file_sha256(source)
    assert original_sha==expected['summary']['source_sha256']
    out=ROOT/'evidence/区域模式验证'
    if device=='0':out=out/'GPU'
    out.mkdir(exist_ok=True,parents=True)
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='pipe_region_') as td:
        app=App(td,device=device)
        app.job=dict(id='real_region_verification',model='improved',event_id=None,point='public_video_demo',
            condition='normal_flow',mode='file',status='loading',processed=0,positive_frames=0,duplicate_frames=0,
            threshold=.25,**detection_options({'detail':'region'}))
        inference_times=[]
        real_detect=app.detect
        def timed_detect(frame):
            result=real_detect(frame);inference_times.append(result[2]);return result
        app.detect=timed_detect
        app.run(source)
        assert app.job['status']=='completed',app.job
        assert app.job['processed']==462 and app.job['duplicate_frames']==0
        trace=app.store.root/app.job['region_trace']
        records=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        assert len(records)==462
        assert app.model_meta['sha256']==expected['summary']['weight_sha256']
        count_different_frames=0;rounded_difference_frames=0
        for i,row in enumerate(records):
            assert abs(row['media_time']-i/25)<1e-8
            olds=expected['frames'][i]['A3']
            count_different_frames+=int(len(row['boxes'])!=len(olds))
            equal=len(row['boxes'])==len(olds) and all(actual['xyxy']==[round(v,2) for v in old['xyxy']] and
                actual['confidence']==round(old['confidence'],5) for actual,old in zip(row['boxes'],olds))
            rounded_difference_frames+=int(not equal)
            if device=='cpu':assert equal
            assert sum(not r['held'] for r in row['regions'])==len(row['boxes'])
        event=app.store.events()[0]
        assert event['level']=='candidate'
        (out/'代表检测画面.jpg').write_bytes((app.store.root/event['image']).read_bytes())
        (out/'原始代表画面.jpg').write_bytes((app.store.root/event['original']).read_bytes())
        (out/'逐帧记录.jsonl').write_bytes(trace.read_bytes())
        elapsed=round(time.perf_counter()-started,2)
        # The framework deep-copies the source model into its inference backend.
        backend=app.detector.predictor.model
        actual_device=str(backend.device)
        parameter_device=str(next(backend.model.parameters()).device)
        if device=='0':assert actual_device.startswith('cuda:0') and parameter_device.startswith('cuda:0')
        import cv2
        import numpy as np
        image=cv2.imdecode(np.frombuffer((out/'原始代表画面.jpg').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        for _ in range(3):real_detect(image)
        warmed=[real_detect(image)[2] for _ in range(20)]
        report=dict(status='passed',source=source.name,source_sha256=original_sha,
            weight_sha256=app.model_meta['sha256'],device=actual_device,device_name=app.model_meta['device_name'],
            backend_parameter_device=parameter_device,
            precision='FP32',processed=app.job['processed'],
            positive_frames=app.job['positive_frames'],duplicate_frames=app.job['duplicate_frames'],
            nominal_fps=25,all_predictions_equal_frozen_cpu_cache_after_app_rounding=rounded_difference_frames==0,
            cpu_reference_count_difference_frames=count_different_frames,
            cpu_reference_rounded_difference_frames=rounded_difference_frames,
            all_current_candidates_visible=True,
            region_visible_frames=sum(bool(r['regions']) for r in records),
            held_only_frames=sum(bool(r['regions']) and not r['boxes'] for r in records),
            event_level=event['level'],event_hits=event['hits'],
            trace_sha256=file_sha256(out/'逐帧记录.jsonl'),elapsed_seconds=elapsed,
            whole_pipeline_frames_per_second=round(len(records)/elapsed,2),
            warmed_image_predict_ms_median=round(float(np.median(warmed)),2),
            warmed_image_predict_ms_p95=round(float(np.quantile(warmed,.95)),2),
            video_predict_ms_median=round(float(np.median(inference_times)),2),
            user_runtime_untouched=True,independent_test=False,complete_human_ground_truth=False,
            live_camera_tested=False,robot_connected=False,real_time_claim=False)
    assert file_sha256(source)==original_sha
    (out/'验证结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['0','cpu'],default='0')
    main(parser.parse_args().device)
