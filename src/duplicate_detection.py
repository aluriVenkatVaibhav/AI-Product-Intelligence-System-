from typing import List, Tuple

import numpy as np

from src.config import (
    FAISS_INDEX_FILE,
    IMAGE_DIR,
    IMAGE_EMBEDDINGS_FILE,
    IMAGE_IDS_FILE,
    METADATA_FILE,
)
from src.models.search_result import SearchResult
from src.vector_db import VectorDatabase


class DuplicateDetector:
    """
    Detect visually similar products using CLIP embeddings and FAISS.
    """

    def __init__(self) -> None:
        """
        Initialize the duplicate detector by loading the FAISS index,
        metadata, image IDs, and embeddings.
        """

        self.db = VectorDatabase()

        self.db.load(FAISS_INDEX_FILE)

        self.db.load_metadata(
            IMAGE_IDS_FILE,
            METADATA_FILE,
        )

        self.embeddings = np.load(IMAGE_EMBEDDINGS_FILE)

    def _build_result(
        self,
        product_id: int,
        similarity: float,
    ) -> SearchResult:
        """
        Build a SearchResult object from product metadata.

        Args:
            product_id: Product ID.
            similarity: Similarity score.

        Returns:
            SearchResult object.
        """

        row = self.db.metadata.loc[product_id]

        return SearchResult(
            product_id=product_id,
            product_name=row["productDisplayName"],
            master_category=row["masterCategory"],
            sub_category=row["subCategory"],
            article_type=row["articleType"],
            gender=row["gender"],
            colour=row["baseColour"],
            usage=row["usage"],
            image_path=IMAGE_DIR / f"{product_id}.jpg",
            similarity=float(similarity),
        )

    def find_duplicates(
        self,
        product_id: int,
        top_k: int = 5,
        similarity_threshold: float = 0.90,
    ) -> Tuple[SearchResult, List[SearchResult]]:
        """
        Find visually similar products.

        Args:
            product_id: Product ID to search.
            top_k: Number of similar products to return.
            similarity_threshold: Minimum cosine similarity.

        Returns:
            Tuple containing:
                - Original product
                - List of visually similar products
        """

        position = np.where(self.db.ids == product_id)[0]

        if len(position) == 0:
            raise ValueError(
                f"Product ID {product_id} not found."
            )

        position = int(position[0])

        query_embedding = self.embeddings[
            position
        ].reshape(1, -1)

        original = self._build_result(
            product_id=product_id,
            similarity=1.0,
        )

        scores, indices = self.db.search(
            query_embedding,
            top_k=top_k + 1,
        )

        duplicates: List[SearchResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            candidate_id = int(self.db.ids[index])

            # Ignore the original image
            if candidate_id == product_id:
                continue

            # Ignore weak matches
            if score < similarity_threshold:
                continue

            duplicates.append(
                self._build_result(
                    product_id=candidate_id,
                    similarity=score,
                )
            )

        return original, duplicates