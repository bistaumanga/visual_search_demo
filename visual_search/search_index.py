"""usearch search index wrapper with metadata support."""

import json
from typing import Literal
import logging
from pathlib import Path
import numpy as np
import usearch.index

logger = logging.getLogger(__name__)

# Configuration
TOP_K = 20


class SearchIndex:
    """Wrapper for usearch search with metadata."""
    
    def __init__(self, index_dir):
        """Initialize search index from directory.
        
        Args:
            index_dir: Path to directory containing usearch indices and metadata
        """
        logger.info(f"Initializing SearchIndex: index_dir={index_dir}")
        self.index_dir = Path(index_dir)
        self.modality: Literal["image", "text", "xmodal"] = None
        self.metadata = None
        self.image_index = None
        self.text_index = None
        self._load()
        logger.info(f"SearchIndex initialized successfully")
    
    def _load(self):
        """Load usearch index and metadata."""
        logger.info(f"Loading indices from {self.index_dir}")
        meta_file = self.index_dir / "index_meta.json"
        
        if not meta_file.exists():
            logger.error(f"Metadata file not found: {meta_file}")
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        
        logger.debug(f"Loading metadata from {meta_file}")
        with open(meta_file, "r") as f:
            self.metadata = json.load(f)
            self.modality = self.metadata.get("modality", None)
            nimages = self.metadata.get("index", {}).get("image", {}).get("count", 0)
            ntexts = self.metadata.get("index", {}).get("text", {}).get("count", 0)
        logger.info(f"Metadata loaded from {meta_file}, modality={self.modality}, #images={nimages}, #texts={ntexts}")
        
        # Load image index
        image_usearch_file = self.index_dir / "image_usearch.index"
        
        if image_usearch_file.exists():
            logger.debug(f"Loading image usearch index from {image_usearch_file}")
            self.image_index = usearch.index.Index.restore(str(image_usearch_file))
            logger.info(f"Image index loaded from {image_usearch_file}, size={self.image_index.size}")
        else:
            logger.error(f"Image usearch file not found: {image_usearch_file}")
        
        # Load text index if xmodal
        text_usearch_file = self.index_dir / "text_usearch.index"
        
        if text_usearch_file.exists():
            logger.debug(f"Loading text usearch index from {text_usearch_file}")
            self.text_index = usearch.index.Index.restore(str(text_usearch_file))
            logger.info(f"Text index loaded from {text_usearch_file}, size={self.text_index.size}")
        else:
            logger.warning(f"Text usearch file not found: {text_usearch_file}")
    
    def search(self, query_embedding: np.ndarray, target_modality: Literal["image", "text"] = "image", top_k: int = TOP_K):
        """Search /text index by embedding."""

        logger.debug(f"Searching for {target_modality}, top_k={top_k}")
        if target_modality == "image":
            if self.image_index is None:
                logger.error("Image index not loaded")
                raise RuntimeError("Image index not loaded")
            index = self.image_index
        elif target_modality == "text":
            if self.text_index is None:
                logger.error("Text index not loaded or not xmodal")
                raise RuntimeError("Text index not loaded or not xmodal")
            index = self.text_index
        else:
            logger.error(f"Unsupported target_modality: {target_modality}")
            raise ValueError(f"Unsupported target_modality: {target_modality}")
        
        # faiss.normalize_L2(query_embedding)

        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        
        logger.debug("Performing usearch search...")
        results_dict = index.search(query_embedding.reshape(1, -1), top_k, exact=True)
        indices = results_dict.keys if hasattr(results_dict, 'keys') else results_dict.get('keys', [])
        distances = results_dict.distances if hasattr(results_dict, 'distances') else results_dict.get('distances', [])
        logger.debug(f"Search completed, found {len(indices)} results")
        
        results = []
        ids_list = self.metadata["index"][target_modality]["ids"]
        for i, (idx, distance) in enumerate(zip(indices, distances)):
            example_id = ids_list[int(idx)]
            results.append({
                "rank": i + 1,
                "id": example_id,
                "image_path": self.metadata["id2data"][example_id]["image_path"],
                "captions": self.metadata["id2data"][example_id].get("descriptions", []),
                "similarity": float(distance),
            })
        
        logger.info(f"Returning {len(results)} {target_modality} after faiss search")
        return results