import faulthandler
import os
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import logging

from typing import Literal, Union

from visual_search.embeddings import get_model
from visual_search.search_index import SearchIndex

# Configure logging
faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATASETS = ["flickr8k", "coco"]
TOP_K = 20

# Global cache for models to prevent re-loading and memory leaks
_MODEL_CACHE = {}

# Global cache for search indices
_INDEX_CACHE = {}

def load_model(model_name):
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading {model_name} into cache...")
        try:
            logger.info(f"Getting model info for {model_name}...")
            MODEL_ID, embedding_cls = get_model(model_name)
            logger.info(f"Model ID: {MODEL_ID}, Class: {embedding_cls.__name__}")
            
            logger.info("Initializing model with from_args()...")
            logger.info(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
            
            _MODEL_CACHE[model_name] = embedding_cls.from_args(
                model_id=MODEL_ID,
                device="mps" if torch.mps.is_available() else "cpu",
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            raise
    return _MODEL_CACHE[model_name]

def load_search_index(dataset_name, model_name):
    """Load search index with caching."""
    cache_key = f"{dataset_name}_{model_name}"
    
    if cache_key not in _INDEX_CACHE:
        logger.info(f"Loading search index into cache: {cache_key}...")
        try:
            index_dir = Path(f".indices/{dataset_name}/{model_name}")
            _INDEX_CACHE[cache_key] = SearchIndex(index_dir)
            logger.info(f"Search index {cache_key} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load search index {cache_key}: {e}", exc_info=True)
            raise
    return _INDEX_CACHE[cache_key]


def search_images(
    query_input: Union[Image.Image, str],
    source_modality: Literal["image", "text"],
    dataset_choice: str,
    top_k: int,
    model_choice: Literal["dino", "siglip"] = "siglip",
):
    """Perform multimodal image search.
    
    Args:
        query_input: Query image (PIL Image) or text string
        source_modality: "image" for image-to-image search, "text" for text-to-image search
        dataset_choice: Dataset name (e.g., "flickr8k", "coco")
        top_k: Number of results to return
        model_choice: Model name (only used for image search)
    
    Returns:
        List of (image, caption) tuples for Gradio Gallery
    """
    logger.info(f"search_images called: source_modality={source_modality}, dataset={dataset_choice}, top_k={top_k}, embedder={model_choice}")
    
    try:
        # Validate input
        if source_modality == "image":
            if query_input is None:
                logger.warning("No image provided")
                return None
            # Load model for image search
            embedding_model = load_model(model_choice)
            logger.debug(f"Processing image query with model={model_choice}")
            
            # Convert PIL image
            if isinstance(query_input, str):
                image = Image.open(query_input).convert("RGB")
            else:
                image = query_input.convert("RGB")
            logger.debug(f"Image loaded: size={image.size}")
            
            # Encode image
            logger.debug("Encoding image...")
            with torch.inference_mode():
                query_embedding = embedding_model.encode_images([image])
                query_embedding = query_embedding.cpu().float().numpy()
            logger.debug(f"Image encoded: shape={query_embedding.shape}")
            
        elif source_modality == "text":
            if not query_input or query_input.strip() == "":
                logger.warning("Empty text query")
                return None
            # Load SigLIP model for text search
            embedding_model = load_model("siglip")
            logger.debug(f"Processing text query: '{query_input}'")
            
            # Encode text
            logger.debug("Encoding text...")
            with torch.inference_mode():
                query_embedding = embedding_model.encode_texts([query_input])
                query_embedding = query_embedding.cpu().float().numpy()
            logger.debug(f"Text encoded: shape={query_embedding.shape}")
            
            # Text search always uses siglip model
            model_choice = "siglip"
        else:
            raise ValueError(f"Invalid source_modality: {source_modality}")
        
        # Load search index (cached)
        search_index = load_search_index(dataset_choice, model_choice)
        
        # Perform search
        results = search_index.search(query_embedding, target_modality="image", top_k=top_k)
        logger.debug(f"Search returned {len(results)} images")
        
        # Format output - list of (image, caption) tuples for gallery
        image_gallery = []
        
        logger.debug("Loading result images...")
        for result in results:
            img_path = result["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                # Create caption with ID and score
                caption = f"ID: {result['id']} | Score: {result['similarity']:.4f}"
                image_gallery.append((img, caption))
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
        
        logger.info(f"Returning {len(image_gallery)} images")
        return image_gallery
    
    except Exception as e:
        logger.error(f"Error in search_images ({source_modality}): {e}", exc_info=True)
        return None


def create_ui():
    """Create Gradio interface."""
    logger.info("Creating Gradio UI...")
    
    with gr.Blocks(title="Multimodal Image Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Multimodal Image Search")
        gr.Markdown("Search images using DINO/SigLIP embeddings and FAISS indices")
        
        with gr.Tabs():
            # Tab 1: Image-to-Image Search
            with gr.TabItem("🖼️ Image-to-Image Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Configuration")
                        model_choice = gr.Radio(
                            choices=["dino", "siglip"],
                            value="dino",
                            label="Model",
                            info="Choose embedding model"
                        )
                        dataset_choice_i2i = gr.Radio(
                            choices=DATASETS,
                            value="flickr8k",
                            label="Dataset",
                            info="Choose dataset"
                        )
                        top_k_i2i = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=20,
                            step=1,
                            label="Top K Results",
                            info="Number of similar images to return"
                        )
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            show_label=True
                        )
                        search_button_i2i = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        image_gallery = gr.Gallery(
                            label="Similar Images",
                            show_label=True,
                            columns=4,
                            rows=5,
                            object_fit="scale-down",
                            height="auto",
                        )
                
                # Hidden component for source modality
                source_modality_i2i = gr.Textbox(value="image", visible=False)
                
                search_button_i2i.click(
                    search_images,
                    inputs=[image_input, source_modality_i2i, dataset_choice_i2i, top_k_i2i, model_choice],
                    outputs=[image_gallery]
                )
            
            # Tab 2: Text-to-Image Search
            with gr.TabItem("📝 Text-to-Image Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Configuration")
                        dataset_choice_t2i = gr.Radio(
                            choices=DATASETS,
                            value="flickr8k",
                            label="Dataset",
                            info="Choose dataset"
                        )
                        top_k_t2i = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=20,
                            step=1,
                            label="Top K Results",
                            info="Number of similar images to return"
                        )
                        text_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter text description...",
                            lines=3
                        )
                        search_button_t2i = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        image_gallery_t2i = gr.Gallery(
                            label="Similar Images",
                            show_label=True,
                            columns=4,
                            rows=5,
                            object_fit="scale-down",
                            height="auto",
                        )
                
                # Hidden components for source modality and model choice (text search always uses siglip)
                source_modality_t2i = gr.Textbox(value="text", visible=False)
                model_choice_t2i = gr.Textbox(value="siglip", visible=False)
                
                search_button_t2i.click(
                    search_images,
                    inputs=[text_input, source_modality_t2i, dataset_choice_t2i, top_k_t2i, model_choice_t2i],
                    outputs=[image_gallery_t2i]
                )
        
        gr.Markdown("---")
        gr.Markdown(
            "**Note:** Make sure FAISS indices are built using `index_xmodal.py` before running this UI.\n\n"
            "Example: `python -m visual_search.index_xmodal --model dino --dataset flickr8k`"
        )
    
    logger.info("Gradio UI created successfully")
    return demo


def run_ui():
    """Run the UI server."""
    logger.info("Starting search_ui...")
    try:
        logger.info("Creating UI...")
        demo = create_ui()
        logger.info("Launching Gradio app...")
        demo.launch(share=False)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_ui()
