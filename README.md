# Visual Search

A multimodal visual search system supporting image and text retrieval using DINO and SigLIP models, with efficient vector indexing via usearch. Datasets supported: COCO and Flickr8k.

This project has been assisted by `Github Copilot` and Google's `Gemini Code Assist` (VSCode extensions).

## Features
- Image and text embedding with DINO and SigLIP2 (HuggingFace Transformers) on flickr8k and coco dataset
- Fast vector search using usearch (cosine metric)
- Gradio web UI for search and gallery display

## Installation

```bash
# Clone the repository
$ git clone https://github.com/bistaumanga/visual_search_demo
$ cd visual_search_demo

# Create and activate virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate

# Install dependencies
$ uv sync
```

## Usage

### Build Index
Run the following script to build indices for all models and datasets:

```bash
bash scripts/index.sh
```
Or manually:
```bash
python -m visual_search.build_index --model dino --dataset coco
python -m visual_search.build_index --model siglip --dataset coco
python -m visual_search.build_index --model dino --dataset flickr8k
python -m visual_search.build_index --model siglip --dataset flickr8k
```

### Start Web UI
```bash
python visual_search/app.py
```

## Project Structure
```
visual_search/
  app.py            # Gradio UI
  build_index.py    # Index building logic
  data.py           # Dataset classes
  embeddings.py     # Embedding models
  search_index.py   # Search API
  __init__.py       # Module exports
datasets/ # these files are necessary at minimum
  coco/
    annotations/
      captions_val2017.json
    val2017/ # contains images
  flickr8k/
    Images/ # contains images
    captions.txt
scripts/index.sh    # Index build script
```

## Testing
Run all unit tests:
```bash
pytest -v
```

## Environment Variables
- `HF_TOKEN`: HuggingFace token for model downloads (set in .env)

## Additional notes
- tested on apple M3 processor with mps backend
  - on M3 (possible apple M series),  Dino-v3 models results NaN with `float16`, use `bfloat16` or `float32`
  - The code probably runs fine with `faiss` search (on non non apple M chips) with minimal changes. It results segmentation fault when using `SQfp16` for half precision on M3 processor.

## License
MITgit 
