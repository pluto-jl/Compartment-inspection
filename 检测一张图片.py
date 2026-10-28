from pathlib import Path
import argparse,json,sys,os
root=Path(__file__).resolve().parent;engine=root/'YOLOv8n_微滴算法_V8_20261016'
sys.path[:0]=[str(engine/'vendor'),str(engine)]
import torch,cv2,numpy as np
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--device',choices=['0','cpu'],default='0');a=p.parse_args()
if a.device=='0' and not torch.cuda.is_available():raise SystemExit('未检测到NVIDIA GPU。可先打开离线查看页面；CPU参考检测请明确使用--device cpu。')
path=a.image or root/'管路微滴训练集_V1/images/train/P_Q16.jpg';im=cv2.imdecode(np.fromfile(path,np.uint8),1)
if im is None:raise SystemExit('无法读取图片')
model=YOLO(str(root/'模型/best.pt'));r=model.predict(im,imgsz=640,conf=.25,iou=.5,device=a.device,quantize=32,verbose=False,max_det=100)[0]
out=root/'检测输出';out.mkdir(exist_ok=True);boxes=[]
for b,c in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist()):
 x,y,z,t=map(round,b);cv2.rectangle(im,(x,y),(z,t),(0,170,255),2);boxes.append(dict(xyxy=b,score=c))
cv2.imencode('.jpg',im)[1].tofile(out/(path.stem+'_检测.jpg'))
(out/(path.stem+'_检测.json')).write_text(json.dumps(dict(image=str(path),actual_device=str(model.predictor.device),boxes=boxes),ensure_ascii=False,indent=2),encoding='utf-8')
print('结果已保存到',out)
