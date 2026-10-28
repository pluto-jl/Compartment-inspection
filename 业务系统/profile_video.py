"""Profile the real GPU path in a temporary store; never alter user jobs."""
import json,tempfile,argparse
from pathlib import Path
from patrol import App,ROOT,detection_options,file_sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path,default=ROOT.parent/'管路微滴训练集_V1/视频素材/Pexels_5698926_演示保留.mp4');parser.add_argument('--out',type=Path,default=ROOT/'evidence/分段计时_20261017');parser.add_argument('--full-preview-reference',action='store_true');args=parser.parse_args()
    if args.full_preview_reference:
        import preview_cadence
        original=preview_cadence.PreviewCadence
        preview_cadence.PreviewCadence=lambda publish: original(publish,interval=0)
    import torch
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    source_sha=file_sha256(args.video)
    with tempfile.TemporaryDirectory(prefix='pipe_profile_') as td:
        app=App(td)
        app.job=dict(id='gpu_profile',model='improved',event_id=None,point='public_demo_profile',condition='normal_flow',mode='file',status='loading',processed=0,positive_frames=0,duplicate_frames=0,threshold=.25,**detection_options({'detail':'region'}))
        app.run(args.video)
        if app.job['status']!='completed':raise RuntimeError(app.job['message'])
        report=json.loads((app.store.root/app.job['timing_profile']).read_text(encoding='utf-8'))
        report.update(source=args.video.name,source_sha256=source_sha,positive_frames=app.job['positive_frames'],user_runtime_untouched=True,independent_test=False,robot_connected=False)
        report.update(preview_policy='every_processed_frame_reference' if args.full_preview_reference else 'wall_clock_250ms_and_final',preview_encoded_frames=app.job['preview_encoded_frames'])
        last_frame=app.store.root/'evidence/gpu_profile_last.jpg'
        report['final_preview_sha256']=file_sha256(last_frame)
        (args.out/'最终画面.jpg').write_bytes(last_frame.read_bytes())
        events=app.store.events()
        report['event_summary']=[{k:e[k] for k in ('hits','confidence','level','review')} for e in events]
        report['source_code_sha256']=file_sha256(ROOT/'patrol.py')
        (args.out/'分段计时.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (args.out/'逐帧检测.jsonl').write_bytes((app.store.root/app.job['region_trace']).read_bytes())
        report['trace_sha256']=file_sha256(args.out/'逐帧检测.jsonl')
        (args.out/'分段计时.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert file_sha256(args.video)==source_sha
    print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
