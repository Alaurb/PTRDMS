import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from ripeness_demo.detectors import TwoStageYoloDetector
from ripeness_demo.detectors import CLASS_ALIASES, model_taxonomy, class_order_for
from ripeness_demo.splits import validate_splits
from scripts.evaluate_pipeline import evaluate


class ReviewFixTests(unittest.TestCase):
    def test_two_stage_keeps_maturity_and_audits_other(self):
        detector = TwoStageYoloDetector.__new__(TwoStageYoloDetector)
        detector.confidence = .25
        detector.brightness_gain = 1
        detector.detector_imgsz = 1280
        detector.classifier_imgsz = 96
        box = SimpleNamespace(xyxy=[np.array([0,0,10,10])], conf=[.81])
        detector.detector = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[box])])
        detector.classifier = SimpleNamespace(names={0:'maturity'}, predict=lambda *a, **kw: [SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=.64))])
        predictions = detector.detect(Image.new('RGB',(20,20)))
        self.assertEqual(predictions[0].class_name, 'mature')
        self.assertEqual(predictions[0].raw_class, 'maturity')
        self.assertEqual(predictions[0].classifier_confidence, .64)
        detector.classifier.names = {0:'other'}
        self.assertEqual(detector.detect(Image.new('RGB',(20,20))), [])
        self.assertEqual(detector.audit[0]['decision'], 'rejected_other')

    def test_paper_maturity_vocabulary_is_preserved_and_unknown_rejected(self):
        self.assertEqual(CLASS_ALIASES['maturity'], 'mature')
        self.assertEqual(CLASS_ALIASES['maturity-period'], 'mature')
        self.assertEqual(CLASS_ALIASES['green-maturity-period'], 'green_mature')
        self.assertEqual(
            model_taxonomy({0:'discoloration-period',1:'green-maturity-period',2:'immature-period',3:'maturity-period'}),
            'paper_four_stage',
        )
        with self.assertRaises(ValueError):
            model_taxonomy({0:'unexpected'})
        with self.assertRaises(ValueError):
            model_taxonomy({0:'harvest_ready',1:'maturity'})

    def test_empty_green_gem_and_immature_only_keep_schema(self):
        for labels in ([], ['immature']):
            self.assertIn('harvest_ready', class_order_for(labels, 'green_gem'))
        with self.assertRaises(ValueError):
            class_order_for(['mature'], 'green_gem')

    def test_end_to_end_counts_wrong_class_miss_and_duplicate(self):
        def pred(name, box):
            return dict(frame='a', side='left', class_name=name,
                        **dict(zip(('bbox_x1','bbox_y1','bbox_x2','bbox_y2'),box)))
        truth={'views':[dict(frame='a',side='left',objects=[
            dict(class_name='harvest_ready',bbox=[0,0,10,10]),
            dict(class_name='immature',bbox=[20,20,30,30])])]}
        result=evaluate([pred('mature_green',[0,0,10,10]),pred('mature_green',[0,0,10,10])],truth,'green_gem')
        self.assertEqual((result['wrong_class'],result['missed'],result['extra']),(1,1,1))
        self.assertEqual(result['per_class']['harvest_ready']['fn'],1)

    def test_duplicate_images_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'a').write_bytes(b'image'); (root/'b').write_bytes(b'image')
            rows=[dict(image=n,split=s,panorama_id=n,group_id=n,capture_date=n,route=n)
                  for n,s in [('a','train'),('b','test')]]
            with self.assertRaisesRegex(ValueError,'leakage'):
                validate_splits(rows,root)
