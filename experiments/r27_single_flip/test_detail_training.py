"""CPU checks of file stages and fusion/ordering; real GPU smoke remains required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import detail_training as training


class TrainingTests(unittest.TestCase):
    def test_completion_detects_artifact_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'assets';out.mkdir()
            (out/'a.json').write_text('{}')
            training.complete(out);training.require(root,'assets')
            (out/'a.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):training.require(root,'assets')

    def test_prepare_does_not_import_torch_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root=Path(temp);project=root/'project'
            stack.enter_context(patch.object(training,'verify_materials'))
            stack.enter_context(patch.object(training,'digest',return_value=training.TRAIN_SHA))
            stack.enter_context(patch.object(sys,'argv',['pilot','prepare','--project',str(project),'--materials-root',temp,'--data-root',temp]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(training.main(),0)
            config=json.loads((project/'project.json').read_text())
            self.assertEqual(config['epochs'],3);self.assertEqual(config['budget_hours'],3)
            with self.assertRaises(FileExistsError):training.main()
        self.assertNotIn('torch',sys.modules)

    def test_windows_rejects_model_execution(self):
        with patch.object(sys,'platform','win32'),patch.object(sys,'argv',['pilot','train','--project','.']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                training.main()
        self.assertEqual(error.exception.code,2)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main()
