import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from visual_search.search_index import SearchIndex


class FakeIndex:
    def __init__(self, size=0, keys=None, distances=None):
        self.size = size
        self._keys = keys if keys is not None else []
        self._distances = distances if distances is not None else []

    def search(self, query, top_k, exact=True):
        class R:
            pass
        r = R()
        r.keys = self._keys
        r.distances = self._distances
        return r


class TestSearchIndex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch('usearch.index.Index.restore')
    def test_load_and_search_image(self, mock_restore):
        # Prepare metadata
        meta = {
            "modality": "xmodal",
            "index": {
                "image": {"count": 2, "ids": ["img1", "img2"]},
                "text": {"count": 2, "ids": ["img1", "img2"]}
            },
            "id2data": {
                "img1": {"image_path": "datasets/flickr8k/Images/img1.jpg", "descriptions": ["a cat"]},
                "img2": {"image_path": "datasets/flickr8k/Images/img2.jpg", "descriptions": ["a dog"]}
            }
        }
        meta_file = self.tmpdir / 'index_meta.json'
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

        # Create dummy index files so SearchIndex attempts to restore
        (self.tmpdir / 'image_usearch.index').write_text('')
        (self.tmpdir / 'text_usearch.index').write_text('')

        # Configure fake indices to return predictable search results
        fake_image_index = FakeIndex(size=2, keys=[0, 1], distances=[0.95, 0.85])
        fake_text_index = FakeIndex(size=2, keys=[1, 0], distances=[0.9, 0.8])
        # The restore mock should return different objects depending on file path
        def restore_side_effect(path):
            if 'image_usearch.index' in path:
                return fake_image_index
            else:
                return fake_text_index
        mock_restore.side_effect = restore_side_effect

        si = SearchIndex(self.tmpdir)
        # Search images
        q = np.random.rand(512).astype(np.float32)
        results = si.search(q, target_modality='image', top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], 'img1')
        self.assertAlmostEqual(results[0]['similarity'], 0.95)
        self.assertEqual(results[1]['id'], 'img2')

        # Search texts
        t_results = si.search(q, target_modality='text', top_k=2)
        self.assertEqual(t_results[0]['id'], 'img2')
        self.assertAlmostEqual(t_results[0]['similarity'], 0.9)

    @patch('usearch.index.Index.restore')
    def test_search_text_without_text_index_raises(self, mock_restore):
        # Metadata indicates image-only modality
        meta = {
            "modality": "image",
            "index": {
                "image": {"count": 1, "ids": ["img1"]}
            },
            "id2data": {
                "img1": {"image_path": "datasets/flickr8k/Images/img1.jpg", "descriptions": ["a cat"]}
            }
        }
        meta_file = self.tmpdir / 'index_meta.json'
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

        # Only create image index file
        (self.tmpdir / 'image_usearch.index').write_text('')
        mock_restore.return_value = FakeIndex(size=1, keys=[0], distances=[0.99])

        si = SearchIndex(self.tmpdir)
        q = np.random.rand(512).astype(np.float32)
        with self.assertRaises(RuntimeError):
            si.search(q, target_modality='text', top_k=1)

    @patch('usearch.index.Index.restore')
    def test_search_with_unsupported_target_modality_raises(self, mock_restore):
        meta = {
            "modality": "xmodal",
            "index": {
                "image": {"count": 1, "ids": ["img1"]}
            },
            "id2data": {
                "img1": {"image_path": "datasets/flickr8k/Images/img1.jpg", "descriptions": ["a cat"]}
            }
        }
        meta_file = self.tmpdir / 'index_meta.json'
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        (self.tmpdir / 'image_usearch.index').write_text('')
        mock_restore.return_value = FakeIndex(size=1, keys=[0], distances=[0.5])

        si = SearchIndex(self.tmpdir)
        q = np.random.rand(512).astype(np.float32)
        with self.assertRaises(ValueError):
            si.search(q, target_modality='audio', top_k=1)


if __name__ == '__main__':
    unittest.main()
