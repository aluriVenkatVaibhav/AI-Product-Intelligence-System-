from typing import List

import torch
from transformers import AutoProcessor, CLIPModel

from src.config import CLIP_MODEL_NAME, DEVICE


class CLIPEngine:
    """
    CLIP model wrapper for image and text embeddings.
    """

    def __init__(self):

        print("=" * 60)
        print("Loading CLIP...")
        print("=" * 60)

        self.processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

        self.model.to(DEVICE)

        self.model.eval()

        print(f"Device : {DEVICE}")
        print("CLIP Loaded Successfully.\n")

    @torch.no_grad()
    def encode_images(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = self.model.get_image_features(**inputs)

        # Some transformers versions return an object instead of a tensor
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state[:, 0]
        else:
            features = outputs

        features = torch.nn.functional.normalize(
            features,
            dim=-1,
        )

        return features.cpu()

    @torch.no_grad()
    def encode_text(self, texts):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = self.model.get_text_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state[:, 0]
        else:
            features = outputs

        features = torch.nn.functional.normalize(
            features,
            dim=-1,
        )

        return features.cpu()