"""Fresh GPU integration on all development images plus six added-FP images."""
from pathlib import Path
import json,tempfile
from patrol import App,ROOT,ENGINE,detection_options
from diagnose_errors import match
import cv2,numpy as np,torch
out=ROOT/'evidence/局部中心去重_20261017';out.mkdir(exist_ok=True)
report=json.loads((ENGINE/'runs/现有数据强化_20261017/局部候选去重优化/中心包含约束核验.json').read_text(encoding='utf-8'))
ids={'P_Q50','P_Q474','P_DROP_0034','P_DROP_0481','P_DROP_0831','P_REVIEW_Q254'}
items=[x for x in report['images'] if x['split']=='val' or x['id'] in ids];assert len(items)==29
torch.set_num_threads(2);details=[]
with tempfile.TemporaryDirectory(prefix='pipe_dedup_') as td:
 app=App(td);app.job=dict(threshold=.25,**detection_options({'detail':'local'}));app.load_model('improved')
 for x in items:
  image=cv2.imdecode(np.fromfile(ROOT.parent/'管路微滴训练集_V1'/x['image'],np.uint8),cv2.IMREAD_COLOR)
  annotated,boxes,ms=app.detect(image);c=match(x['gt'],[dict(xyxy=b['xyxy'],conf=b['confidence']) for b in boxes],.25)
  assert all(c[k]==x['counts']['dedup_zoom'][k] for k in ['tp','fp','fn']),(x['id'],c)
  assert [b['xyxy'] for b in boxes[:len(x['full_predictions'])]]==[[round(v,2) for v in p['xyxy']] for p in x['full_predictions']],x['id']
  if x['id'] in {'P_Q50','P_REVIEW_Q496'}:
   assert cv2.imwrite(str(out/(x['id']+'_去重后.jpg')),annotated)
  details.append(dict(id=x['id'],counts={k:c[k] for k in ['tp','fp','fn']},original_full_boxes_preserved=True,added_candidates=sum(b.get('source')=='local_zoom_candidate' for b in boxes)))
 result=dict(status='passed',fresh_gpu_images=29,all_development_images=23,additional_fp_images=6,actual_device=str(app.detector.predictor.device),original_weight_sha256=app.model_meta['sha256'],matches_frozen_replay=True,user_runtime_untouched=True,independent_test=False,details=details)
(out/'GPU新推理核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
