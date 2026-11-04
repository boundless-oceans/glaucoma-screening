"""Offline ledger regression checks using synthetic aggregate reports only."""
import copy,hashlib,importlib.util,json,sys,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summarizer',ROOT/'scripts/summarize_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'outputs'/('summary_test_'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.report=dict(status='completed',metrics=dict(auroc=.9,average_precision=.6,tp=100,fn=91,fp=58,tn=5600,threshold=.5))
        (self.root/'report.json').write_text(json.dumps(self.report))
        self.item=dict(id='synthetic',title='Synthetic fixture',report='report.json',metrics='metrics',guards=[['status','completed']],decision='test',reason='synthetic only',platform=None,submission=None)
        self.catalog=dict(schema='experiment-catalog-v1',entries=[self.item])

    def test_known_metrics_and_unknown_platform_are_separate(self):
        rows=m.collect(self.catalog,self.root)
        self.assertEqual(rows[0]['metrics']['auroc'],.9)
        self.assertIsNone(rows[0]['platform']);self.assertIn('待反馈/未提交',m.markdown(rows))
        self.assertFalse(rows[0]['t4_accepted']);self.assertNotIn('torch',sys.modules)

    def test_missing_or_partial_report_is_not_best(self):
        self.item['report']='missing.json'
        self.assertEqual(m.collect(self.catalog,self.root)[0]['evidence'],'missing_report')
        self.item['report']='report.json';self.report['status']='paused'
        (self.root/'report.json').write_text(json.dumps(self.report))
        with self.assertRaises(ValueError):m.collect(self.catalog,self.root)

    def test_hash_mismatch_and_escape_rejected(self):
        (self.root/'complete.json').write_text(json.dumps({'report.json':'0'*64}))
        with self.assertRaises(ValueError):m.collect(self.catalog,self.root)
        self.item['report']='../outside.json'
        with self.assertRaises(ValueError):m.collect(self.catalog,self.root)

    def test_submission_mismatch_rejected(self):
        (self.root/'candidate.csv').write_text('Synthetic fixture only')
        self.item['submission']=dict(path='candidate.csv',sha256='0'*64)
        with self.assertRaises(ValueError):m.collect(self.catalog,self.root)

if __name__=='__main__':unittest.main()
