@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "检测一张图片.py" --device 0
) else (
  python "检测一张图片.py" --device 0
)
if errorlevel 1 echo 未能运行，请查看“环境安装说明.txt”。
pause
