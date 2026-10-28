"""Engineering integration checks on a separate QA database, not accuracy tests."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=os.environ.get('PATROL_BASE','http://127.0.0.1:7862')


def request(path, data=None, raw=None, headers=None):
    payload=raw if raw is not None else None if data is None else json.dumps(data).encode()
    h={'X-Patrol-Client':'1','Content-Type':'application/json'}
    h.update(headers or {})
    req=urllib.request.Request(BASE+path,payload,headers=h)
    with urllib.request.urlopen(req,timeout=30) as r:
        b=r.read()
        return json.loads(b) if r.headers['Content-Type'].startswith('application/json') else b


def wait():
    for _ in range(100):
        s=request('/api/state')
        if not s['running']:return s['job']
        time.sleep(.15)
    raise AssertionError('Task timed out')


def upload(path):
    return request('/api/upload',raw=path.read_bytes(),headers={'X-Filename':path.name})['key']


def start(uploaded, **changes):
    opts=dict(mode='file',upload=uploaded,filename='engineering fixture',point='QA fixture not real patrol',
              model='improved',condition='expected_dry',threshold=.25)
    opts.update(changes);request('/api/start',opts);return wait()


def main():
    # Never run this destructive-to-test-history script against the user data directory.
    assert request('/api/state').get('qa_mode'), 'Run the server with --data qa_runtime first; never use your production records'
    fixtures=ROOT/'qa_fixtures';fixtures.mkdir(exist_ok=True)
    images=sorted((ROOT.parent/'管路微滴训练集_V1/images/train').glob('*.jpg'))
    image=cv2.imdecode(np.frombuffer(images[0].read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    video=fixtures/'engineering_sequence.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,640))
    assert writer.isOpened()
    for i in range(40):
        f=cv2.resize(image,(640,640));cv2.putText(f,f'ENGINEERING FIXTURE {i}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        writer.write(f)
    writer.release()
    # Real image upload and real GPU detector, not a mocked detector.
    imgjob=start(upload(images[0]))
    assert imgjob['status']=='completed',imgjob
    assert imgjob['processed']==1
    assert len(imgjob['source_sha256'])==64
    assert imgjob['event_id'] is not None
    event=request('/api/event/'+imgjob['event_id'])
    assert event['level']=='candidate'
    assert event['model_sha256']==hashlib.sha256((ROOT.parent/'模型/best.pt').read_bytes()).hexdigest()
    assert len(request('/'+event['original']))>1000
    assert len(request('/'+event['image']))>1000
    reviewed=request('/api/review',dict(id=event['id'],status='follow_up',note='integration test only',revision=event['revision']))
    assert reviewed['revision']==1 and len(reviewed['reviews'])==1
    # Actual encoded video exercises sampling and temporal state end to end.
    videojob=start(upload(video))
    assert videojob['status']=='completed',videojob
    assert videojob['processed']>=5,videojob
    assert videojob['raw_video']
    ve=request('/api/event/'+videojob['event_id'])
    assert ve['level']=='suspected',ve
    # Broken image must be visibly failed, never completed or normal.
    broken=fixtures/'broken.jpg';broken.write_bytes(b'not an image')
    badjob=start(upload(broken))
    assert badjob['status']=='error' and badjob['event_id'] is None
    # Cross-site writes and path traversal are rejected.
    for req in [urllib.request.Request(BASE+'/api/stop',b'{}'),
        urllib.request.Request(BASE+'/api/stop',b'{}',headers={'X-Patrol-Client':'1','Origin':'http://other.example'})]:
        try:urllib.request.urlopen(req);raise AssertionError('Cross-site write accepted')
        except urllib.error.HTTPError as e:assert e.code==403
    try:request('/evidence/../models.json');raise AssertionError('Traversal accepted')
    except urllib.error.HTTPError as e:assert e.code==404
    export=request('/api/export')
    assert any(e['id']==event['id'] and e['reviews'] for e in export['events'])
    csv=request('/api/export?format=csv')
    assert csv.startswith(b'\xef\xbb\xbf')
    report=request('/api/export?format=report').decode('utf-8')
    assert 'data:image/jpeg;base64,' in report
    assert 'integration test only' in report
    (ROOT/'qa_report.html').write_text(report,encoding='utf-8')
    result=dict(scope='software engineering checks only, not accuracy or robot test',
        gpu_device=imgjob['model_meta']['device'],image_upload_and_detection=True,
        image_event_evidence_and_model_hash=True,review_history=True,video_sampling=True,
        video_processed_frames=videojob['processed'],video_repeat_rule=True,
        invalid_image_fails_visibly=True,cross_site_mutation_rejected=True,path_traversal_rejected=True,
        json_csv_export=True,offline_report_embedded_images=True,source_hash_recorded=True,
        robot_tested=False,camera_hardware_tested=False,
        improved_model_accuracy_tested=False)
    out=ROOT/'evidence';out.mkdir(exist_ok=True)
    (out/'integration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
