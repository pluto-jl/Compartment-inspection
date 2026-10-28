import unittest
from pipeline_timing import PipelineTiming
from evaluate_events import evaluate
class EvaluationTests(unittest.TestCase):
    def test_timing_files_not_listed_as_jobs(self):
        import tempfile,json
        from pathlib import Path
        from patrol import App
        with tempfile.TemporaryDirectory() as td:
            app=App(td)
            (Path(td)/'jobs/one_timing.json').write_text(json.dumps({'wall_seconds':1}),encoding='utf-8')
            self.assertEqual(app.jobs(),[])
    def test_worker_finalizing_blocks_new_start_and_reports_running(self):
        import tempfile
        from types import SimpleNamespace
        from patrol import App
        with tempfile.TemporaryDirectory() as td:
            app=App(td);app.worker=SimpleNamespace(is_alive=lambda:True)
            app.job={'status':'completed'}
            self.assertTrue(app.state()['running'])
            with self.assertRaisesRegex(ValueError,'已有任务'):
                app.start(dict(mode='sample',point='test',model='improved'))

    def test_exclusive_clock_no_double_count(self):
        stamp=[0.];p=PipelineTiming(clock=lambda:stamp[0])
        with p.measure('outer'):
            stamp[0]=1
            with p.measure('inner'):stamp[0]=3
            stamp[0]=4
        s=p.summary(1);self.assertEqual(s['stages']['outer']['exclusive_ms'],2000);self.assertEqual(s['stages']['inner']['exclusive_ms'],2000);self.assertEqual(s['unattributed_ms'],0)
    def test_truth_required(self):
        with self.assertRaises(ValueError):evaluate([],{'independent_human_review':False})
    def test_held_excluded_false_episode_and_normal_separate(self):
        box=lambda x:dict(xyxy=[x,0,x+2,2])
        rows=[dict(media_time=t,boxes=b,regions=[dict(held=True,xyxy=[0,0,2,2])]) for t,b in [(0,[]),(.1,[box(0)]),(.2,[box(20)]),(.3,[box(20)]),(.9,[box(20)]),(1.1,[])]]
        gt=dict(independent_human_review=True,reviewer='synthetic test fixture',windows=[dict(id='dry',condition='expected_dry',start=0,end=1,exhaustive_review=True,events=[dict(id='e1',start=0,end=.5,roi_xyxy=[0,0,5,5])]),dict(id='flow',condition='normal_flow',start=1,end=2,exhaustive_review=True,events=[dict(id='e2',start=1,end=1.5,roi_xyxy=[0,0,5,5])])])
        r=evaluate(rows,gt)['windows'];self.assertEqual(r[0]['false_candidate_episodes'],2);self.assertEqual(r[0]['events'][0]['first_discovery_delay_media_seconds'],.1);self.assertEqual(r[1]['missed_events'],1);self.assertEqual(r[1]['false_candidate_episodes'],0)
if __name__=='__main__':unittest.main()
