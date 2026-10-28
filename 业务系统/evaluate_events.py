"""Offline event coverage; requires exhaustive human reviewed windows.
Raw boxes only: held presentation regions never count as fresh detections.
False episodes use a predeclared 0.5 s gap rule, not application job totals.
"""
import argparse,hashlib,json,math
from pathlib import Path

def evaluate(rows,gt,gap=.5):
    if gt.get('independent_human_review') is not True or not gt.get('reviewer'):
        raise ValueError('需要独立人工验收与实际验收人；不能用空模板计算正式指标')
    times=[r['media_time'] for r in rows]
    if any(not math.isfinite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('检测时间必须有限、非负且严格递增')
    windows=gt.get('windows',[])
    if not windows:raise ValueError('需要完整复核的时间区间')
    prior=-1;out=[]
    for w in sorted(windows,key=lambda w:w['start']):
        start,end=w['start'],w['end']
        if not (math.isfinite(start) and math.isfinite(end) and 0<=start<end) or start<prior:raise ValueError('时间区间重叠或无效')
        prior=end
        if w.get('exhaustive_review') is not True:raise ValueError('每个区间需完整复核，不能仅标成功帧')
        if w.get('condition') not in ('normal_flow','expected_dry','unknown'):raise ValueError('工况无效')
        events=w.get('events',[])
        for e in events:
            if not start<=e['start']<e['end']<=end:raise ValueError('事件必须位于复核区间内')
            r=e['roi_xyxy']
            if len(r)!=4 or not all(math.isfinite(v) for v in r) or r[0]>=r[2] or r[1]>=r[3]:raise ValueError('空间区域无效')
        wr=[r for r in rows if start<=r['media_time']<end]
        result=[];false_times=[]
        for e in events:
            active=[r for r in wr if e['start']<=r['media_time']<e['end']]
            hit=[r['media_time'] for r in active if any(inside(b,e['roi_xyxy']) for b in r.get('boxes',[]))]
            result.append(dict(id=e['id'],detected=bool(hit),first_discovery_delay_media_seconds=round(min(hit)-e['start'],6) if hit else None,position_coverage_of_sampled_event_frames=len(hit)/len(active) if active else None,sampled_event_frames=len(active)))
        for r in wr:
            active=[e for e in events if e['start']<=r['media_time']<e['end']]
            # False candidates alongside true candidates still count; held-only hints do not.
            if any(not any(inside(b,e['roi_xyxy']) for e in active) for b in r.get('boxes',[])):
                false_times.append(r['media_time'])
        times=sorted(set(false_times));episodes=sum(i==0 or t-times[i-1]>gap for i,t in enumerate(times))
        missed=sum(not e['detected'] for e in result)
        out.append(dict(id=w['id'],condition=w['condition'],reviewed_seconds=end-start,gt_events=len(events),missed_events=missed,event_miss_rate=missed/len(events) if events else None,false_candidate_episodes=episodes,false_candidate_episodes_per_minute=episodes/((end-start)/60),events=result))
    return dict(schema_version=1,status='evaluated_human_reviewed_intervals',independent_blind_test=bool(gt.get('independent_blind_test',False)),gap_seconds=gap,windows=out,notes='Half-open intervals; source-pixel boxes, fixed human ROI per event; moving views must be split into fixed-ROI events. Location coverage uses sampled frames only. First-discovery delay is media time, not live end-to-end latency. False episodes aggregate temporally per reviewed window; not physical defect count or automatic alarms. Normal-flow is reported separately; do not combine it with abnormal-leak recall.')

def inside(box,roi):
    x1,y1,x2,y2=box['xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
    return roi[0]<=cx<=roi[2] and roi[1]<=cy<=roi[3]

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True,type=Path);p.add_argument('--truth',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--trace-meta',required=True,type=Path);a=p.parse_args()
    rows=[json.loads(line) for line in a.trace.read_text(encoding='utf-8-sig').splitlines() if line.strip()];gt=json.loads(a.truth.read_text(encoding='utf-8-sig'))
    meta=json.loads(a.trace_meta.read_text(encoding='utf-8-sig'))
    if meta.get('trace_sha256')!=hashlib.sha256(a.trace.read_bytes()).hexdigest() or meta.get('source_sha256')!=gt.get('source_sha256') or not gt.get('source_sha256'):
        raise ValueError('原视频与逐帧检测摘要未匹配，不能进行评分')
    result=evaluate(rows,gt);result.update(trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),truth_sha256=hashlib.sha256(a.truth.read_bytes()).hexdigest(),source_sha256=gt.get('source_sha256'))
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
