"""Visual Search Module

A comprehensive visual search system combining DINO and SigLIP embeddings with FAISS indices
for efficient image and text-based search.

Modules:
    - embeddings: Embedding models (DINO, SigLIP)
    - data: Dataset classes (Flickr8k, COCO)
    - build_index: Index building and management
    - search_index: Search functionality over built indices
    - app: Gradio-based search interface
"""

from .embeddings import (
    BaseEmbedding,
    ImageEmbedding,
    TextEmbedding,
    XModalEmbedding,
    DinoEmbedding,
    SigLIP2Embedding,
    get_model,
)
from .data import (
    XModalExample,
    XModalDataset,
    Flickr8kDataset,
    CocoDataset,
    get_dataset,
)
from .search_index import (
    SearchIndex
)
from .build_index import (
    process_and_save_embeddings,
)
from .app import (
    run_ui,
)

__all__ = [
    # Embeddings
    "BaseEmbedding",
    "ImageEmbedding",
    "TextEmbedding",
    "XModalEmbedding",
    "DinoEmbedding",
    "SigLIP2Embedding",
    "get_model",
    # Data
    "XModalExample",
    "XModalDataset",
    "Flickr8kDataset",
    "CocoDataset",
    "get_dataset",
    # Indexing
    "process_and_save_embeddings",
    # Search UI
    "SearchIndex",
    "run_ui",
]

__version__ = "0.1.0"
