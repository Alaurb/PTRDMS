import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.prepare_four_stage_classifier import CLASSES, REQUIRED, prepare
from scripts.train_four_stage_classifier import _check_classifier_layout


class DatasetPreflightTests(unittest.TestCase):
    def fixture(self, root):
        images = root / 'images'
        images.mkdir()
        rows = []
        for si, split in enumerate(('train', 'val', 'test')):
            for ci, label in enumerate(CLASSES):
                name = f'{split}_{label}.png'
                Image.new('RGB',(32,32),(si*70,ci*40,100)).save(images/name)
                rows.append(dict(image=name, x1='1.2',y1='1.2',x2='20.7',y2='20.7',
                                 label=label,split=split,panorama_id=name,group_id=name,
                                 capture_date=f'2026-09-{si+1:02d}',route=f'row{si}'))
        return images, rows

    def write(self, root, rows):
        annotations = root / 'labels.csv'
        with annotations.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=sorted(REQUIRED))
            writer.writeheader();writer.writerows(rows)
        return annotations

    def test_last_invalid_box_leaves_no_partial_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images,rows=self.fixture(root)
            rows[-1]['x2']='nan'
            with self.assertRaisesRegex(ValueError,'box'):
                prepare(self.write(root,rows),images,root/'out')
            self.assertFalse((root/'out').exists())

    def test_crop_rounding_and_training_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images,rows=self.fixture(root)
            prepare(self.write(root,rows),images,root/'out')
            _check_classifier_layout(root/'out')
            crop=next((root/'out/train/immature-period').glob('*.jpg'))
            with Image.open(crop) as image:
                self.assertEqual(image.size,(20,20))
            crop.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                _check_classifier_layout(root/'out')

    def test_unregistered_crop_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images,rows=self.fixture(root)
            prepare(self.write(root,rows),images,root/'out')
            Image.new('RGB',(20,20)).save(root/'out/test/immature-period/unregistered.jpg')
            with self.assertRaisesRegex(ValueError,'differ'):
                _check_classifier_layout(root/'out')
