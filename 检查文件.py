from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent;m=json.loads((r/'文件清单.json').read_text(encoding='utf-8'))
for name,expected in m['files'].items():
 p=r/name
 if not p.is_file():raise SystemExit('缺少文件: '+name)
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=expected:raise SystemExit('内容改变: '+name)
print('全部',len(m['files']),'个文件完整。此检查不代表检测准确率或跨电脑驱动兼容性。')
