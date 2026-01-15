from abc import ABC, abstractmethod
from typing import Literal, Any
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
import torch
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

class BaseEmbedding(ABC):
    modality: Literal['image', 'text', 'xmodal']
    model: Any
    hf_model_id: str
    processor: Any
    device: Any

    @classmethod
    def from_args(cls, model_id: str, device: str = None):
        device = torch.device(device or ("cpu"))
        return cls(model_id, device=device)

    @abstractmethod
    def __init__(self, model_id: str, device=None):
        pass

class ImageEmbedding(BaseEmbedding):
    modality: Literal['image'] = 'image'

    def __init__(self, model_id: str, device=None):
        self.device = device or torch.device('cpu')
        self.hf_model_id = model_id
        # Load model in float32, then convert to bf16 for efficiency
        self.model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id, token=HF_TOKEN)

    @abstractmethod
    def encode_images(self, images, **kwargs):
        pass

    def get_image_embedding_dim(self):
        dummy_image = Image.new("RGB", (224, 224))
        emb = self.encode_images([dummy_image]).cpu()
        return emb.shape[1]

    def encode_texts(self, texts, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not support text encoding.")

class TextEmbedding(BaseEmbedding):
    modality: Literal['text'] = 'text'

    def __init__(self, model_id: str, device=None):
        self.device = device or torch.device('cpu')
        self.hf_model_id = model_id
        # Load model in float32, then convert to bf16 for efficiency
        self.model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    @abstractmethod
    def encode_texts(self, texts, **kwargs):
        pass

    def get_text_embedding_dim(self):
        dummy_text = "This is a dummy text."
        emb = self.encode_texts([dummy_text]).cpu()
        return emb.shape[1]

    def encode_images(self, images, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not support image encoding.")

class XModalEmbedding(ImageEmbedding, TextEmbedding):
    modality: Literal['xmodal'] = 'xmodal'

    def __init__(self, model_id: str, device=None):
        # Call both parent initializers for processor/tokenizer
        self.device = device or torch.device('cpu')
        self.hf_model_id = model_id
        # Load model in float32, then convert to bf16 for efficiency
        self.model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id, token=HF_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)


class DinoEmbedding(ImageEmbedding):
    modality: Literal['image'] = 'image'

    def encode_images(self, images, **kwargs):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

class SigLIP2Embedding(XModalEmbedding):
    modality: Literal['xmodal'] = 'xmodal'

    def encode_images(self, images, **kwargs):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            embeddings = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode_texts(self, texts, **kwargs):
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(self.device)
        with torch.inference_mode():
            embeddings = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def get_model(model_name: Literal['dino', 'siglip']):
    return {
        "dino": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", DinoEmbedding),
        "siglip": ("google/siglip2-base-patch16-naflex", SigLIP2Embedding),
    }[model_name]
