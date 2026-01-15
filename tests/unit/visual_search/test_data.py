import unittest
from pathlib import Path
from visual_search import data
import tempfile
import shutil
import os
import json
import pandas as pd
from PIL import Image

class TestXModalExample(unittest.TestCase):
    def test_serialization(self):
        ex = data.XModalExample(id="1", image_path=Path("/tmp/img.jpg"), descriptions=["a cat", "a pet"])
        d = ex.model_dump()
        self.assertEqual(d["id"], "1")
        self.assertEqual(d["image_path"], "/tmp/img.jpg")
        self.assertEqual(d["descriptions"], ["a cat", "a pet"])

class TestFlickr8kDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_dir = Path(self.tmpdir) / "Images"
        self.img_dir.mkdir()
        # Create dummy images
        for i in range(2):
            img = Image.new("RGB", (10, 10), color=(i*50, i*50, i*50))
            img.save(self.img_dir / f"img{i}.jpg")
        # Create captions file
        self.captions_file = Path(self.tmpdir) / "captions.txt"
        df = pd.DataFrame({"image": ["img0.jpg", "img1.jpg"], "caption": ["cat", "dog"]})
        df.to_csv(self.captions_file, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_read_data(self):
        ds = data.Flickr8kDataset(image_folder=self.img_dir, captions_file=self.captions_file)
        ds.read_data()
        self.assertEqual(len(ds.examples), 2)
        self.assertEqual(ds.examples[0].descriptions, ["cat"])
        self.assertEqual(ds.examples[1].descriptions, ["dog"])

    def test_iter_images(self):
        ds = data.Flickr8kDataset(image_folder=self.img_dir, captions_file=self.captions_file)
        ds.read_data()
        batches = list(ds.iter_images(batch_size=1))
        self.assertEqual(len(batches), 2)
        for imgs, ids in batches:
            self.assertEqual(len(imgs), 1)
            self.assertEqual(len(ids), 1)

    def test_iter_texts(self):
        ds = data.Flickr8kDataset(image_folder=self.img_dir, captions_file=self.captions_file)
        ds.read_data()
        batches = list(ds.iter_texts(batch_size=1))
        self.assertEqual(len(batches), 2)
        texts, ids = batches[0]
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(ids), 1)

class TestCocoDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_dir = Path(self.tmpdir) / "val2017"
        self.img_dir.mkdir()
        # Create dummy image
        img = Image.new("RGB", (10, 10), color=(123, 123, 123))
        img.save(self.img_dir / "000000000001.jpg")
        # Create dummy annotation file
        self.ann_file = Path(self.tmpdir) / "captions_val2017.json"
        coco = {
            "images": [{"id": 1, "file_name": "000000000001.jpg"}],
            "annotations": [{"image_id": 1, "caption": "a test caption"}]
        }
        with open(self.ann_file, "w") as f:
            json.dump(coco, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_read_data(self):
        ds = data.CocoDataset(image_folder=self.img_dir, annotations_file=self.ann_file)
        ds.read_data()
        self.assertEqual(len(ds.examples), 1)
        self.assertEqual(ds.examples[0].descriptions, ["a test caption"])

    def test_iter_images(self):
        ds = data.CocoDataset(image_folder=self.img_dir, annotations_file=self.ann_file)
        ds.read_data()
        batches = list(ds.iter_images(batch_size=1))
        self.assertEqual(len(batches), 1)
        imgs, ids = batches[0]
        self.assertEqual(len(imgs), 1)
        self.assertEqual(len(ids), 1)

    def test_iter_texts(self):
        ds = data.CocoDataset(image_folder=self.img_dir, annotations_file=self.ann_file)
        ds.read_data()
        batches = list(ds.iter_texts(batch_size=1))
        self.assertEqual(len(batches), 1)
        texts, ids = batches[0]
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(ids), 1)

class TestGetDataset(unittest.TestCase):
    def test_get_dataset(self):
        self.assertIs(data.get_dataset("flickr8k"), data.Flickr8kDataset)
        self.assertIs(data.get_dataset("coco"), data.CocoDataset)

if __name__ == "__main__":
    unittest.main()
