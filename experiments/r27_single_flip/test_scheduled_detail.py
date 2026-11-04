"""CPU lifecycle and numerical-expression checks; no torch imports."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from PIL import Image
from parallel_decode import background_decode, crop_decoded
with patch.dict(sys.modules,{'predict':types.SimpleNamespace(fuse_probability=None)}):
    from scheduled_detail_adapter import background_crop

class ScheduleTests(unittest.TestCase):
    def test_crop_pixel_equality_and_cleanup_on_failure(self):
        image=Image.new('RGB',(40,20),'white')
        roi=dict(decoded_size=[20,10],original_size=[40,20],roi_box=[-3,2,7,12])
        expected,box=crop_decoded(image,roi,mirrored=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch('parallel_decode.decode_original',return_value=image):
                with self.assertRaisesRegex(RuntimeError,'global failed'):
                    with background_decode(pool,'unused') as original:
                        with background_crop(pool,original,roi,True) as future:
                            crop,mapped=future.result()
                            self.assertEqual(mapped,box)
                            self.assertEqual(crop.tobytes(),expected.tobytes())
                            raise RuntimeError('global failed')
        for obj in (image,crop):
            with self.assertRaises(ValueError):obj.getpixel((0,0))

    def test_numerical_expressions_and_precision_contexts_preserved(self):
        root=Path(__file__).resolve().parent
        def expressions(file):
            tree=ast.parse((root/file).read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('predict_'))
            keys={'gl','heat','point','center','roi','ll','p','mapped'}
            values={n.targets[0].id:ast.dump(n.value) for n in ast.walk(fn) if isinstance(n,ast.Assign)
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys}
            contexts=sorted(ast.dump(n.context_expr) for n in ast.walk(fn) if isinstance(n,ast.withitem)
                            and isinstance(n.context_expr,ast.Call) and 'torch.' in ast.unparse(n.context_expr))
            return values,contexts
        self.assertEqual(expressions('parallel_detail_adapter.py'),expressions('scheduled_detail_adapter.py'))

if __name__=='__main__':unittest.main()
