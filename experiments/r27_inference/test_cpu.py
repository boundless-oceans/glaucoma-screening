"""Stdlib-only portable contracts; never imports torch."""
import csv,json,subprocess,sys,unittest,uuid
from pathlib import Path
from contracts import verify,MODES
from io_contracts import inputs,summary,timed_call

class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[2]/'outputs'/('r27_portable_cpu_'+uuid.uuid4().hex)
        cls.root.mkdir(parents=True)

    def test_input_order_case_and_rejections(self):
        root=self.root/'input';root.mkdir();(root/'b.JPG').write_bytes(b'x');(root/'a.JPG').write_bytes(b'y')
        meta=self.root/'input.csv'
        with meta.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','True_Label']);w.writerows([['a.jpg','1'],['b.jpg','0']])
        self.assertEqual([n for n,_ in inputs(root,meta)],['a.jpg','b.jpg'])
        meta.write_text('Image_name\n../a.JPG\n')
        with self.assertRaises(ValueError):inputs(root,meta)
        meta.write_text('Image_name\na.JPG\na.jpg\n')
        with self.assertRaises(ValueError):inputs(root,meta)

    def test_timing_scope_and_statistics(self):
        events=[];ticks=iter([2.,2.05])
        details,elapsed=timed_call(lambda p:events.append('predict') or {'probability':.2},None,lambda:events.append('sync'),lambda:next(ticks))
        self.assertEqual(events,['sync','predict','sync']);self.assertAlmostEqual(elapsed,50.)
        r=summary([80,90,150],'NVIDIA T4',True)
        self.assertTrue(r['hardware_is_t4']);self.assertFalse(r['official_acceptance'])
        self.assertFalse(r['all_measured_under_or_equal_100ms']);self.assertEqual(r['over_100ms'],1)

    def test_incomplete_bundle_rejected(self):
        root=self.root/'bad';root.mkdir();(root/'bundle.json').write_text(json.dumps(dict(schema='r27-portable-v1',detail=False,files={})))
        with self.assertRaisesRegex(ValueError,'Incomplete'):verify(root)

    def test_cli_help_never_loads_models(self):
        here=Path(__file__).resolve().parent
        for script in ('prepare.py','run.py'):
            result=subprocess.run([sys.executable,str(here/script),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('torch',sys.modules)

    @unittest.skipIf(sys.platform=='linux','Windows isolation guard')
    def test_windows_benchmark_rejected_before_file_access(self):
        here=Path(__file__).resolve().parent
        result=subprocess.run([sys.executable,str(here/'run.py'),'benchmark','--mode',MODES[1],'--images','missing','--metadata','missing','--output',str(self.root/'must_not_create')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('local GPU use disabled',result.stderr)
        self.assertFalse((self.root/'must_not_create').exists())

if __name__=='__main__':unittest.main()
