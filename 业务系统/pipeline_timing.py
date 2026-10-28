"""Wall-clock profiling with exclusive stages; nested work is counted only once."""
from contextlib import contextmanager
from functools import wraps
import time

class PipelineTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock=clock;self.started=clock();self.stack=[];self.stages={}
    @contextmanager
    def measure(self,name):
        node=[self.clock(),0.0];self.stack.append(node)
        try:yield
        finally:
            elapsed=self.clock()-node[0];self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            exclusive=max(0.,elapsed-node[1]);r=self.stages.setdefault(name,dict(calls=0,total_seconds=0.0,max_seconds=0.0))
            r['calls']+=1;r['total_seconds']+=exclusive;r['max_seconds']=max(r['max_seconds'],exclusive)
    def summary(self,frames):
        wall=self.clock()-self.started;accounted=sum(s['total_seconds'] for s in self.stages.values())
        stages={k:dict(calls=v['calls'],exclusive_ms=round(v['total_seconds']*1000,3),mean_ms=round(v['total_seconds']*1000/v['calls'],3),max_ms=round(v['max_seconds']*1000,3),wall_percent=round(v['total_seconds']/wall*100,2)) for k,v in self.stages.items()}
        return dict(schema_version=1,wall_seconds=round(wall,4),processed_frames=frames,whole_pipeline_fps=round(frames/wall,3),stages=stages,unattributed_ms=round(max(0.,wall-accounted)*1000,3),timing_kind='exclusive_wall_clock',video_reencoding=False,notes='No MP4 writer in this backend. JPEG evidence/preview encoding is measured. GPU prediction includes output CPU synchronization. First prediction includes framework startup; not a steady-state benchmark.')

def profiled(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self,*args,**kwargs):
            profiler=getattr(self,'profiler',None)
            if profiler is None:return fn(self,*args,**kwargs)
            with profiler.measure(name):return fn(self,*args,**kwargs)
        return wrapped
    return decorate
