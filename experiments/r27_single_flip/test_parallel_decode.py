"""Pixel equality and cleanup checks; no torch/GPU imports."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import patch
import ast
from pathlib import Path
import unittest
from PIL import Image
from parallel_decode import decode_original, crop_decoded, background_decode
from detail_crop import prepare_detail

class DecodeTests(unittest.TestCase):
    def test_jpeg_orientations_views_and_outside_padding(self):
        image=Image.new('RGB',(83,57))
        image.putdata([((x*7)%256,(y*11)%256,(x*y)%256) for y in range(57) for x in range(83)])
        for orientation in (1,2,3,4,5,6,7,8):
            stream=BytesIO();exif=Image.Exif();exif[274]=orientation
            image.save(stream,format='JPEG',exif=exif);data=stream.getvalue()
            real_open=Image.open
            def opened(_):return real_open(BytesIO(data))
            with patch('parallel_decode.Image.open',side_effect=opened):
                decoded=decode_original('synthetic.jpg')
            for mirrored in (False,True):
                for box in ([1,2,20,25],[-6,-4,24,30]):
                    roi=dict(decoded_size=[32,32],original_size=list(decoded.size),roi_box=box)
                    with patch('detail_crop.Image.open',side_effect=opened):
                        expected,ebox=prepare_detail('synthetic.jpg',roi,32,mirrored)
                    actual,abox=crop_decoded(decoded,roi,32,mirrored)
                    self.assertEqual(ebox,abox)
                    self.assertEqual(expected.tobytes(),actual.tobytes())
            decoded.close()

    def test_background_cleanup_on_exception(self):
        image=Image.new('RGB',(8,8))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch('parallel_decode.decode_original',return_value=image):
                with self.assertRaisesRegex(RuntimeError,'prediction failed'):
                    with background_decode(pool,'unused') as future:
                        future.result()
                        raise RuntimeError('prediction failed')
        with self.assertRaises(ValueError):image.getpixel((0,0))

    def test_model_expressions_unchanged(self):
        root=Path(__file__).resolve().parent
        def predictive(name):
            tree=ast.parse((root/name).read_text())
            # Assignment RHS keeps classifier/locator precision and numerical expressions.
            keys={'gl','heat','point','center','roi','ll','p'}
            return [ast.dump(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys]
        self.assertEqual(predictive('detail_adapter.py'),predictive('parallel_detail_adapter.py'))

if __name__=='__main__':unittest.main()
