from pathlib import Path
import sys,runpy,webbrowser,threading,os
root=Path(__file__).resolve().parent
os.environ['PYTHONUTF8']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
app=root/'管路巡检系统_20261016';sys.path.insert(0,str(app));sys.argv=[str(app/'patrol.py'),'--port','7862']
threading.Timer(2,lambda:webbrowser.open('http://127.0.0.1:7862')).start()
runpy.run_path(str(app/'patrol.py'),run_name='__main__')
