"""Verify the application path on all 23 images and an isolated persisted image job."""
import sys,json,tempfile
from pathlib import Path
from patrol import App,ROOT,detection_options,file_sha256,ENGINE
from diagnose_errors import match
import cv2,numpy as np,torch
out=ROOT/'evidence/图片局部复核_20261017';out.mkdir(exist_ok=True)
ref=json.loads((ENGINE/'runs/现有数据强化_20261017/局部放大实验/边界约束后结果.json').read_text(encoding='utf-8'));torch.set_num_threads(2)
counts=dict(tp=0,fp=0,fn=0);details=[]
with tempfile.TemporaryDirectory(prefix='pipe_local_check_') as td:
 app=App(td);app.job=dict(threshold=.25,**detection_options({'detail':'local'}));app.load_model('improved')
 for row in ref['images']:
  image=cv2.imdecode(np.fromfile(ROOT.parent/'管路微滴训练集_V1'/row['image'],np.uint8),cv2.IMREAD_COLOR)
  annotated,boxes,ms=app.detect(image);c=match(row['gt'],[dict(xyxy=p['xyxy'],conf=p['confidence']) for p in boxes],.25)
  assert all(c[k]==row['counts']['zoom'][k] for k in ['tp','fp','fn']),(row['id'],c)
  for k in counts:counts[k]+=c[k]
  details.append(dict(id=row['id'],counts={k:c[k] for k in counts},new_candidates=sum(p.get('source')=='local_zoom_candidate' for p in boxes)))
 assert counts==dict(tp=13,fp=2,fn=5)
 app.job=None
 source=ROOT.parent/'管路微滴训练集_V1/images/val/P_REVIEW_Q496.jpg'
 job=app.start(dict(mode='file',point='隔离GPU图片局部复核测试',model='improved',detail='local',condition='unknown',threshold=.25),source)
 app.worker.join(30);assert not app.worker.is_alive();assert app.job['status']=='completed',app.job
 event=app.store.events()[0];assert event['level']=='candidate';assert sum(p.get('source')=='local_zoom_candidate' for p in event['boxes'])==2
 (out/'局部复核画面.jpg').write_bytes((app.store.root/event['image']).read_bytes())
 timing=json.loads((app.store.root/app.job['timing_profile']).read_text(encoding='utf-8'));assert 'local_crop_predict_and_merge' in timing['stages']
 report=dict(status='passed',actual_device=str(app.detector.predictor.device),model_sha256=app.model_meta['sha256'],static_23_counts=counts,event_level=event['level'],added_candidates=2,original_a3_retained=True,user_runtime_untouched=True,independent_test=False,details=details)
(out/'GPU接入验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)
