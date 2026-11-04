"""Unlabeled-test admission and local-execution guards; no torch."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from export_detail import validate_tests

class ExportTests(unittest.TestCase):
    def test_test_identity_admission(self):
        rows=[dict(image_id=f'test_{i:05}.jpg',relative_path=f'test/test_{i:05}.JPG',role='test',official_split='test',label='') for i in range(5855)]
        validate_tests(rows)
        for key,value in [('label','1'),('role','valid'),('relative_path','test/../test_00000.jpg'),('image_id','test_00001.jpg')]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):validate_tests(bad)
        with self.assertRaises(ValueError):validate_tests(rows[:-1])

    @unittest.skipIf(sys.platform=='linux','Windows guard only')
    def test_windows_export_rejected_before_creating_project(self):
        script=Path(__file__).with_name('export_detail.py')
        # Missing project deliberately: guard must fire before any filesystem access.
        result=subprocess.run([sys.executable,str(script),'export','--project','__must_not_create_export__'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('no local model execution',result.stderr)
        self.assertNotIn('Traceback',result.stderr)

if __name__=='__main__':unittest.main()
