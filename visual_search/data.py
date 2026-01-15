from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterator, Tuple, Literal
from collections import defaultdict
import json
import pandas as pd
from pydantic import BaseModel, Field, field_serializer
from PIL import Image

class XModalExample(BaseModel):
    id: str = Field(..., description="Unique identifier for the example")
    image_path: Path = Field(..., description="Path to the image file")
    descriptions: list[str] = Field(..., description="List of captions for the image.")

    @field_serializer("image_path")
    def _serialize_image_path(self, v):
        return str(v)

class XModalDataset(ABC):
    def __init__(self):
        self.examples: List[XModalExample] = []

    @abstractmethod
    def read_data(self):
        pass

    def iter_images(self, batch_size: int) -> Iterator[Tuple[List[Image.Image], List[str]]]:
        for i in range(0, len(self.examples), batch_size):
            batch_examples = self.examples[i : i + batch_size]
            batch_images = []
            try:
                for ex in batch_examples:
                    img = Image.open(ex.image_path).convert("RGB")
                    batch_images.append(img)
                yield batch_images, [ex.id for ex in batch_examples]
            finally:
                # Close all images to release resources
                for img in batch_images:
                    if hasattr(img, 'close'):
                        img.close()

    def iter_texts(self, batch_size: int) -> Iterator[Tuple[list[str], list[str]]]:
        """
        Yields batches of (texts, example_ids), where:
        - texts: flat list of all descriptions in the batch (one per description)
        - example_ids: list of ids, repeated for each description
        Batching is done after flattening all descriptions.
        """
        all_texts = []
        all_ids = []
        for ex in self.examples:
            for desc in ex.descriptions:
                all_texts.append(desc)
                all_ids.append(ex.id)
        for i in range(0, len(all_texts), batch_size):
            yield all_texts[i:i+batch_size], all_ids[i:i+batch_size]


class Flickr8kDataset(XModalDataset):
    def __init__(self, image_folder: Path = Path("./datasets/flickr8k/Images"), captions_file: Path = Path("./datasets/flickr8k/captions.txt")):
        super().__init__()
        self.image_folder = image_folder
        self.captions_file = captions_file

    def read_data(self):
        fname2captions = defaultdict(list)
        df = pd.read_csv(self.captions_file)
        for row in df.to_dict(orient='records'):
            filename = row['image']
            caption = row['caption']
            fname2captions[filename].append(caption)

        # List images in folder
        image_files = sorted([f for f in self.image_folder.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        
        for img_path in image_files:
            # Check if we have captions for this image
            if img_path.name in fname2captions:
                example = XModalExample(
                    id=img_path.stem,
                    image_path=img_path,
                    descriptions=fname2captions[img_path.name]
                )
                self.examples.append(example)


class CocoDataset(XModalDataset):
    def __init__(self, image_folder: Path = Path("./datasets/coco/val2017"), annotations_file: Path = Path("./datasets/coco/annotations/captions_val2017.json")):
        super().__init__()
        self.image_folder = image_folder
        self.annotations_file = annotations_file

    def read_data(self):
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)

        id2fname = {img['id']: img['file_name'] for img in coco_data['images']}
        id2captions = defaultdict(list)
        for ann in coco_data['annotations']:
            id2captions[ann['image_id']].append(ann['caption'])

        for img_id, fname in id2fname.items():
            img_path = self.image_folder / fname
            if img_path.exists() and img_id in id2captions:
                example = XModalExample(
                    id=img_path.stem,
                    image_path=img_path,
                    descriptions=id2captions[img_id]
                )
                self.examples.append(example)

_DATASET_CLASS_MAP = {
    "flickr8k": Flickr8kDataset,
    'coco': CocoDataset
}

def get_dataset(name: Literal['flickr8k', 'coco']):
    return _DATASET_CLASS_MAP[name]
