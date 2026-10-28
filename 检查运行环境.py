from pathlib import Path
import sys,importlib,hashlib
r=Path(__file__).resolve().parent;engine=r/'YOLOv8n_微滴算法_V8_20261016';sys.path[:0]=[str(engine/'vendor'),str(engine)]
for name in ['torch','torchvision','cv2','numpy','PIL','scipy','matplotlib','yaml','tqdm','requests','psutil','polars']:
 try:importlib.import_module(name);print(name,'OK')
 except ImportError as e:raise SystemExit(str(e)+'；请按环境安装说明安装依赖')
import torch
from ultralytics import YOLO
p=r/'模型/best.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='f6c4f73e9bf20ffe2b1e29a3651d948074242839429ef6df2a14ee6b0eefc4d3'
model=YOLO(str(p));assert model.names=={0:'droplet'}
print('定稿模型加载成功；GPU可用:',torch.cuda.is_available())
if not torch.cuda.is_available():print('无可用GPU时可看离线页面或明确运行CPU参考示例；GPU工作台不会自动回退CPU。')
