import argparse
import json
import logging
from pathlib import Path
import torch
from usearch.index import Index, search
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file as safetensors_save_torch

from visual_search.data import get_dataset
from visual_search.embeddings import get_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _process_modality(embedding_model, dataset, args, modality_type):
    """Helper to process either images or texts."""
    if modality_type == "image":
        get_dim = embedding_model.get_image_embedding_dim
        iterator = dataset.iter_images
        encoder = embedding_model.encode_images
        desc = "Processing images"
    else:
        get_dim = embedding_model.get_text_embedding_dim
        iterator = dataset.iter_texts
        encoder = embedding_model.encode_texts
        desc = "Processing texts"
    
    dimension = get_dim()
    index = Index(ndim=dimension, metric='cos', dtype=np.float16)
    all_embeddings, ids_list = [], []
    
    total = sum(len(ex.descriptions) for ex in dataset.examples) if modality_type == "text" else len(dataset.examples)
    total_batches = (total + args.batch_size - 1) // args.batch_size
    
    for batch, ids in tqdm(iterator(args.batch_size), total=total_batches, desc=desc):
        embeddings = encoder(batch).cpu()    
        all_embeddings.append(embeddings)
        ids_list.extend(ids)
        index.add(np.arange(len(ids_list) - len(ids), len(ids_list)), embeddings.float().numpy())
    
    return index, ids_list, dimension, torch.cat(all_embeddings)


def process_and_save_embeddings(embedding_model, dataset, args, output_dir, is_xmodal):
    """Process images and texts (if xmodal), save indices and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    image_index, image_ids, image_dim, image_embeddings = _process_modality(embedding_model, dataset, args, "image")
    image_index.save(str(output_dir / "image_usearch.index"))
    # Convert to bf16 for storage
    image_embeddings_bf16 = image_embeddings.to(torch.bfloat16)
    safetensors_save_torch({"embeddings": image_embeddings_bf16}, str(output_dir / "image_vectors.safetensors"))
    logger.info(f"✅ Indexed {image_index.size} images")
    
    # Build index metadata
    index_data = {
        "model": {"name": args.model, "hf_model_id": embedding_model.hf_model_id, "precision": "fp16"},
        "dataset": {"name": args.dataset, "num_examples": len(dataset.examples)},
        "modality": embedding_model.modality,
        "index": {"image": {"embedding_dim": image_dim, "count": image_index.size, "ids": image_ids}},
        "id2data": {ex.id: ex.model_dump() for ex in dataset.examples},
    }
    
    # Process texts if xmodal
    if is_xmodal:
        text_index, text_ids, text_dim, text_embeddings = _process_modality(embedding_model, dataset, args, "text")
        text_index.save(str(output_dir / "text_usearch.index"))
        # Convert to bf16 for storage
        text_embeddings_bf16 = text_embeddings.to(torch.bfloat16)
        safetensors_save_torch({"embeddings": text_embeddings_bf16}, str(output_dir / "text_vectors.safetensors"))
        index_data["index"]["text"] = {"embedding_dim": text_dim, "count": text_index.size, "ids": text_ids}
        logger.info(f"✅ Indexed {text_index.size} texts")
    
    # Save metadata
    meta_file = output_dir / "index_meta.json"
    json.dump(index_data, open(meta_file, "w"), indent=3)
    logger.info(f"✅ Saved metadata to {meta_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dino", "siglip"], default="dino", help="Which model to index (dino or siglip)")
    parser.add_argument("--dataset", choices=["flickr8k", "coco"], default="flickr8k", help="Which dataset to index (flickr8k or coco)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--check", action="store_true", help="Run on first 250 examples only for quick validation")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(args.log_level)

    # Load model and dataset
    MODEL_ID, embedding_cls = get_model(args.model)
    embedding_model = embedding_cls.from_args(
        model_id=MODEL_ID,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    logger.info("Loading dataset...")
    dataset = get_dataset(args.dataset)()
    dataset.read_data()
    
    # Limit to first 250 examples if check mode
    if args.check:
        dataset.examples = dataset.examples[:250]
        logger.info(f"✓ Check mode: limiting to first 250 examples")
    
    logger.info(f"Dataset: {len(dataset.examples)} examples")
    logger.info(f"Model: {args.model} ({embedding_model.modality})")

    # Get setup
    output_dir = Path(f".indices/{args.dataset}/{args.model}")
    is_xmodal = embedding_model.modality == 'xmodal'

    # Process and save
    process_and_save_embeddings(embedding_model, dataset, args, output_dir, is_xmodal)

if __name__ == "__main__":
    main()
