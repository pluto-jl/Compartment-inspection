# 巡检者 · 管路微滴检测与复核系统

> 基于 YOLOv8n 的船舶机舱管路微滴(液体泄漏)视觉检测项目
> 参赛:2026 第八届全球校园 AI 算法精英大赛 · 算法创新赛(AI+学科交叉)
> 编号:AIC-2026-89565014 | 版本:2026-10-20 省赛定稿

## 项目简介

船舶机舱内管路密集、空间狭小,微小液体泄漏难以及时发现。本项目训练了一个面向管路微滴检测的 YOLOv8n 模型,并针对误检引入**自定义 CSEdgeGate 门控**与**局部放大复核**机制,配套本地 HTTP 检测工作台,实现"检测 → 复核 → 记录"闭环。

## 主要特性

- YOLOv8n 微滴检测:3,022,725 参数 / 24 层,第 22 层为自定义 CSEdgeGate 门控
- 局部放大复核 `local_zoom_review.py`、分块检测实验 `tiled_detection.py`
- 本地 HTTP 工作台(固定端口 7862),前后端分离,含证据留存
- 技术报告 19 页 / 13 图 / 14 表;验收 37/37、稳定性 16/16、单测 42/42 通过

## 目录结构

| 目录/文件 | 说明 |
|---|---|
| `算法包/` | 检测算法源码(YOLOv8n + CSEdgeGate + vendor/ultralytics 定制版) |
| `业务系统/` | 工作台后端 `patrol.py` + web 前端 + 测试脚本 |
| `模型/` | `best.pt` 定稿权重(3,022,725 参数) |
| `管路微滴训练集_V1/` | 数据集:1031 张训练图 + 23 张验证图 + YOLO 标签 + `data.yaml` |
| `docs/` | 技术报告(pdf/docx)、说明文档、环境安装说明、模型卡片、SHA-256 文件清单 |
| `*.bat / *.py`(根目录) | 启动工作台、单图检测、环境自检、完整性校验脚本 |

## 快速开始

```powershell
# 1. 准备环境(Python 3.12)
py -3.12 -m venv .venv
# PyTorch:按 https://pytorch.org/get-started/locally/ 选择本机 CUDA 版本安装
.venv\Scripts\python.exe -m pip install -r docs\requirements-inference.txt

# 2. 自检环境
.venv\Scripts\python.exe 检查运行环境.py

# 3. 启动检测工作台 → http://127.0.0.1:7862
启动GPU工作台.bat
```

> ⚠️ **必须整包使用 `算法包/vendor/ultralytics` 定制版**,该版本在 `nn/tasks.py` 顶层 import `csedgegate`,不能被 pip 安装的 ultralytics 替换。
> 无 GPU 时使用 `CPU参考检测示例.bat` 做单图推理;更详细的从零安装步骤见 `docs/环境安装说明.txt`。

## 数据集说明

- 单类别 `droplet`(微滴),YOLO 格式标注,`data.yaml` 使用相对路径
- 训练集 1031 张、开发验证集 23 张;样本与来源审核记录见 `管路微滴训练集_V1/审核记录/`

## 模型卡片

- 权重:`模型/best.pt`(SHA-256:`f6c4f73e9bf20ffe2b1e29a3651d948074242839429ef6df2a14ee6b0eefc4d3`)
- 推理:`imgsz=640`、`conf=0.25`、`nms_iou=0.5`
- 详见 `docs/模型说明.json`

## 许可与声明

- `算法包/vendor/ultralytics` 为定制版 Ultralytics,遵循 **GNU AGPL-3.0** 许可(见 `算法包/ULTRALYTICS_LICENSE`)
- 本项目为竞赛开发原型,模型与数据集用于学术/竞赛展示,非工业级产品,不构成安全承诺
- 原始完整交付包(含打包产物、运行数据与过程材料)未包含在本仓库,见项目根目录 `4_交付包_完整版20261017/`