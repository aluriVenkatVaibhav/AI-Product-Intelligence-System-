from typing import List

from src.config import (
    FAISS_INDEX_FILE,
    IMAGE_DIR,
    IMAGE_IDS_FILE,
    METADATA_FILE,
)
from src.embedding import CLIPEngine
from src.models.search_result import SearchResult
from src.vector_db import VectorDatabase


class ReverseSearchEngine:
    """
    Performs reverse product search by converting a text query into a CLIP
    embedding and retrieving the most similar products from the FAISS index.
    """

    def __init__(self) -> None:
        """
        Initialize the CLIP model and load the FAISS index and product metadata.
        """

        self.clip = CLIPEngine()

        self.db = VectorDatabase()

        self.db.load(FAISS_INDEX_FILE)

        self.db.load_metadata(
            IMAGE_IDS_FILE,
            METADATA_FILE,
        )

    def _build_result(
        self,
        product_id: int,
        similarity: float,
    ) -> SearchResult:
        """
        Build a SearchResult object from product metadata.

        Args:
            product_id: Product ID.
            similarity: Cosine similarity score.

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

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search for products matching a natural language query.

        Args:
            query: User's search query.
            top_k: Number of products to retrieve.

        Returns:
            List of SearchResult objects sorted by similarity.
        """

        # Generate CLIP text embedding
        query_embedding = self.clip.encode_text([query]).numpy()

        # Search FAISS index
        scores, indices = self.db.search(
            query_embedding,
            top_k,
        )

        results: List[SearchResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            product_id = int(self.db.ids[index])

            results.append(
                self._build_result(
                    product_id=product_id,
                    similarity=score,
                )
            )

        return results