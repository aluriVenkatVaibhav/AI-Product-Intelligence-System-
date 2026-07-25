from pathlib import Path
from typing import Union

from PIL import Image


class ImagePreprocessor:
    """
    Handles image loading and preprocessing.
    """

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image and convert it to RGB.

        Args:
            image_path: Path to the image.

        Returns:
            PIL.Image.Image
        """

        image = Image.open(image_path).convert("RGB")

        return image