"""Synthetic CPU geometry checks, no model imports."""
import unittest
from io import BytesIO
from unittest.mock import patch
from PIL import Image, ImageOps
from detail_crop import prepare_detail, source_box


class DetailTests(unittest.TestCase):
    def test_mirrored_roi_matches_full_frame_mirror_with_padding(self):
        pixels = Image.new('RGB', (40, 20))
        pixels.putdata([(x*5, y*10, 0) for y in range(20) for x in range(40)])
        roi = dict(decoded_size=[20, 10], original_size=[40, 20], roi_box=[-3, 2, 7, 12])
        box = source_box(roi, pixels.size)
        expected = ImageOps.mirror(pixels).crop(box)
        with patch('detail_crop.Image.open', return_value=pixels):
            actual, mapped = prepare_detail('synthetic.jpg', roi, 20, mirrored=True)
        self.assertEqual(mapped, box)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_bad_dimensions_and_output_size(self):
        roi = dict(decoded_size=[float('nan'), 10], original_size=[40, 20], roi_box=[0, 0, 1, 1])
        with self.assertRaises(ValueError):
            source_box(roi, (40, 20))
        for size in (True, 2.5, 0, -1):
            with self.assertRaises(ValueError):
                prepare_detail('synthetic.jpg', roi, size)

    def test_physical_roi_and_padding(self):
        roi = dict(decoded_size=[100, 50], original_size=[400, 200], roi_box=[-10, 10, 30, 50])
        self.assertEqual(source_box(roi, (400, 200)), (-40, 40, 120, 200))
        data = BytesIO()
        Image.new('RGB', (400, 200), 'white').save(data, format='JPEG')
        data.seek(0)
        opened = Image.open(data)
        with patch('detail_crop.Image.open', return_value=opened):
            output, box = prepare_detail('synthetic.jpg', roi, 160)
        self.assertEqual(output.size, (160, 160))
        self.assertEqual(output.getpixel((0, 80)), (0, 0, 0))
        self.assertEqual(output.getpixel((100, 80)), (255, 255, 255))

    def test_exif_orientation_and_size_guard(self):
        data = BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (400, 200), 'white').save(data, format='JPEG', exif=exif)
        data.seek(0)
        opened = Image.open(data)
        roi = dict(decoded_size=[50, 100], original_size=[200, 400], roi_box=[10, 20, 30, 40])
        with patch('detail_crop.Image.open', return_value=opened):
            output, box = prepare_detail('synthetic.jpg', roi)
        self.assertEqual(box, (40, 80, 120, 160))
        with self.assertRaises(ValueError):
            source_box(roi, (400, 200))


if __name__ == '__main__':
    unittest.main()
