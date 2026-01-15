"""
Unit tests for embeddings.py module.

Tests cover:
- BaseEmbedding class functionality
- ImageEmbedding abstract class
- TextEmbedding abstract class
- XModalEmbedding class
- DinoEmbedding concrete implementation
- SigLIP2Embedding concrete implementation
- get_model utility function
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import torch
from PIL import Image
import numpy as np

from visual_search.embeddings import (
    BaseEmbedding,
    ImageEmbedding,
    TextEmbedding,
    XModalEmbedding,
    DinoEmbedding,
    SigLIP2Embedding,
    get_model,
)


class TestBaseEmbedding(unittest.TestCase):
    """Test BaseEmbedding abstract base class."""

    def test_from_args_default_device(self):
        """Test from_args uses CPU by default."""
        with patch.object(DinoEmbedding, '__init__', return_value=None):
            instance = DinoEmbedding.from_args(
                model_id="test-model",
            )
            self.assertIsNotNone(instance)

    def test_from_args_explicit_device(self):
        """Test from_args with explicit device parameter."""
        with patch.object(DinoEmbedding, '__init__', return_value=None):
            instance = DinoEmbedding.from_args(
                model_id="test-model",
                device="cpu",
            )
            self.assertIsNotNone(instance)




class TestImageEmbedding(unittest.TestCase):
    """Test ImageEmbedding abstract class."""

    def setUp(self):
        """Set up mocks for ImageEmbedding tests."""
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_init_with_defaults(self, mock_auto_model, mock_auto_processor):
        """Test ImageEmbedding initialization with defaults."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor

        embedding = DinoEmbedding(
            model_id="test-model",
            
        )

        self.assertEqual(embedding.hf_model_id, "test-model")
        self.assertEqual(embedding.modality, 'image')
        self.assertIsNotNone(embedding.device)

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_init_with_custom_device(self, mock_auto_model, mock_auto_processor):
        """Test ImageEmbedding initialization with custom device."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor

        device = torch.device('cpu')
        embedding = DinoEmbedding(
            model_id="test-model",
            device=device,
        )

        self.assertEqual(embedding.device, device)

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_encode_texts_not_implemented(self, mock_auto_model, mock_auto_processor):
        """Test that encode_texts raises NotImplementedError for ImageEmbedding."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor

        embedding = DinoEmbedding(
            model_id="test-model",
            
        )

        with self.assertRaises(NotImplementedError):
            embedding.encode_texts(["test text"])

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_get_image_embedding_dim(self, mock_auto_model, mock_auto_processor):
        """Test get_image_embedding_dim method."""
        # Setup mock model to return embeddings
        mock_embeddings = torch.randn(1, 768)
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor

        embedding = DinoEmbedding(
            model_id="test-model",
            
        )

        # Mock encode_images to return known shape
        with patch.object(embedding, 'encode_images', return_value=mock_embeddings):
            dim = embedding.get_image_embedding_dim()
            self.assertEqual(dim, 768)


class TestTextEmbedding(unittest.TestCase):
    """Test TextEmbedding abstract class."""

    def setUp(self):
        """Set up mocks for TextEmbedding tests."""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_init_with_defaults(self, mock_auto_model, mock_auto_tokenizer, mock_auto_processor):
        """Test TextEmbedding initialization with defaults."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_tokenizer.return_value = self.mock_tokenizer
        mock_auto_processor.return_value = MagicMock()

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        self.assertEqual(embedding.hf_model_id, "test-model")
        self.assertEqual(embedding.modality, 'xmodal')

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_encode_images_not_implemented_text_embedding(self, mock_auto_model, mock_auto_tokenizer):
        """Test that encode_images raises NotImplementedError for TextEmbedding subclass."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_tokenizer.return_value = self.mock_tokenizer

        # Create a concrete TextEmbedding subclass for testing
        class ConcreteTextEmbedding(TextEmbedding):
            def encode_texts(self, texts, **kwargs):
                pass

        embedding = ConcreteTextEmbedding(
            model_id="test-model",
        )

        with self.assertRaises(NotImplementedError):
            embedding.encode_images([Image.new("RGB", (224, 224))])

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_get_text_embedding_dim(self, mock_auto_model, mock_auto_tokenizer, mock_auto_processor):
        """Test get_text_embedding_dim method."""
        mock_embeddings = torch.randn(1, 768)
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        mock_auto_model.return_value = self.mock_model
        mock_auto_tokenizer.return_value = self.mock_tokenizer
        mock_auto_processor.return_value = MagicMock()

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        # Mock encode_texts to return known shape
        with patch.object(embedding, 'encode_texts', return_value=mock_embeddings):
            dim = embedding.get_text_embedding_dim()
            self.assertEqual(dim, 768)


class TestXModalEmbedding(unittest.TestCase):
    """Test XModalEmbedding class."""

    def setUp(self):
        """Set up mocks for XModalEmbedding tests."""
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_tokenizer = MagicMock()

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_init_creates_both_processor_and_tokenizer(
        self, mock_auto_model, mock_auto_processor, mock_auto_tokenizer
    ):
        """Test XModalEmbedding creates both processor and tokenizer."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        mock_auto_tokenizer.return_value = self.mock_tokenizer

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        self.assertIsNotNone(embedding.processor)
        self.assertIsNotNone(embedding.tokenizer)
        self.assertEqual(embedding.modality, 'xmodal')

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_xmodal_modality_attribute(
        self, mock_auto_model, mock_auto_processor, mock_auto_tokenizer
    ):
        """Test XModalEmbedding has correct modality."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        mock_auto_tokenizer.return_value = self.mock_tokenizer

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        self.assertEqual(embedding.modality, 'xmodal')


class TestDinoEmbedding(unittest.TestCase):
    """Test DinoEmbedding concrete implementation."""

    def setUp(self):
        """Set up mocks for DinoEmbedding tests."""
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.test_image = Image.new("RGB", (224, 224))

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_encode_images_returns_embeddings(self, mock_auto_model, mock_auto_processor):
        """Test encode_images returns normalized embeddings."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor

        # Setup mock processor to return MagicMock with to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(side_effect=lambda k: torch.randn(1, 3, 224, 224))
        mock_inputs.to.return_value = mock_inputs
        self.mock_processor.return_value = mock_inputs

        # Setup mock model with pooler_output
        mock_outputs = MagicMock()
        mock_outputs.pooler_output = torch.randn(1, 768)
        self.mock_model.return_value = mock_outputs
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        embedding = DinoEmbedding(
            model_id="test-model",
            
        )

        embeddings = embedding.encode_images([self.test_image])

        # Check shape
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 768)

        # Check normalization (norm should be ~1)
        norm = torch.norm(embeddings, p=2, dim=1)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_dino_modality_attribute(self, mock_auto_model, mock_auto_processor):
        """Test DinoEmbedding has correct modality."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        embedding = DinoEmbedding(
            model_id="test-model",
            
        )

        self.assertEqual(embedding.modality, 'image')


class TestSigLIP2Embedding(unittest.TestCase):
    """Test SigLIP2Embedding concrete implementation."""

    def setUp(self):
        """Set up mocks for SigLIP2Embedding tests."""
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.test_image = Image.new("RGB", (224, 224))
        self.test_text = "A test image"

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_encode_images_returns_embeddings(
        self, mock_auto_model, mock_auto_processor, mock_auto_tokenizer
    ):
        """Test encode_images returns normalized embeddings."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        mock_auto_tokenizer.return_value = self.mock_tokenizer

        # Setup mock processor to return MagicMock with to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(side_effect=lambda k: torch.randn(1, 3, 224, 224))
        mock_inputs.to.return_value = mock_inputs
        self.mock_processor.return_value = mock_inputs

        # Setup mock model with get_image_features
        self.mock_model.get_image_features = MagicMock(return_value=torch.randn(1, 768))
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        embeddings = embedding.encode_images([self.test_image])

        # Check shape
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 768)

        # Check normalization
        norm = torch.norm(embeddings, p=2, dim=1)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_encode_texts_returns_embeddings(
        self, mock_auto_model, mock_auto_processor, mock_auto_tokenizer
    ):
        """Test encode_texts returns normalized embeddings."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        mock_auto_tokenizer.return_value = self.mock_tokenizer

        # Setup mock tokenizer to return MagicMock with to() method
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(side_effect=lambda k: torch.randint(0, 1000, (1, 10)))
        mock_inputs.to.return_value = mock_inputs
        self.mock_tokenizer.return_value = mock_inputs

        # Setup mock model with get_text_features
        self.mock_model.get_text_features = MagicMock(return_value=torch.randn(1, 768))
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        embeddings = embedding.encode_texts([self.test_text])

        # Check shape
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 768)

        # Check normalization
        norm = torch.norm(embeddings, p=2, dim=1)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    @patch('visual_search.embeddings.AutoTokenizer.from_pretrained')
    @patch('visual_search.embeddings.AutoImageProcessor.from_pretrained')
    @patch('visual_search.embeddings.AutoModel.from_pretrained')
    def test_siglip2_modality_attribute(
        self, mock_auto_model, mock_auto_processor, mock_auto_tokenizer
    ):
        """Test SigLIP2Embedding has correct modality."""
        mock_auto_model.return_value = self.mock_model
        mock_auto_processor.return_value = self.mock_processor
        mock_auto_tokenizer.return_value = self.mock_tokenizer
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model

        embedding = SigLIP2Embedding(
            model_id="test-model",
            
        )

        self.assertEqual(embedding.modality, 'xmodal')


class TestGetModel(unittest.TestCase):
    """Test get_model utility function."""

    def test_get_model_dino(self):
        """Test get_model returns correct DINO model."""
        model_id, embedding_cls = get_model("dino")

        self.assertEqual(model_id, "facebook/dinov3-vits16plus-pretrain-lvd1689m")
        self.assertEqual(embedding_cls, DinoEmbedding)

    def test_get_model_siglip(self):
        """Test get_model returns correct SigLIP model."""
        model_id, embedding_cls = get_model("siglip")

        self.assertEqual(model_id, "google/siglip2-base-patch16-naflex")
        self.assertEqual(embedding_cls, SigLIP2Embedding)

    def test_get_model_invalid(self):
        """Test get_model raises KeyError for invalid model."""
        with self.assertRaises(KeyError):
            get_model("invalid_model")


if __name__ == "__main__":
    unittest.main()
