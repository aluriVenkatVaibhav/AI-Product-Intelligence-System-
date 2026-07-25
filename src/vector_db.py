from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import pandas as pd


class VectorDatabase:
    """
    Wrapper around a FAISS index for storing and searching product embeddings.
    """

    def __init__(self, embedding_dimension: int = 512) -> None:
        """
        Initialize an empty FAISS index.

        Args:
            embedding_dimension: Dimension of the embedding vectors.
        """
        self.dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(self.dimension)

        self.ids: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the FAISS index from image embeddings.

        Args:
            embeddings: NumPy array of shape (N, embedding_dimension).
        """
        self.index.add(embeddings.astype(np.float32))

    def save(self, path: Path) -> None:
        """
        Save the FAISS index to disk.

        Args:
            path: Output index file path.
        """
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        """
        Load a FAISS index from disk.

        Args:
            path: Path to the saved FAISS index.
        """
        self.index = faiss.read_index(str(path))

    def load_metadata(
        self,
        ids_path: Path,
        metadata_path: Path,
    ) -> None:
        """
        Load image IDs and product metadata.

        Args:
            ids_path: Path to image_ids.npy.
            metadata_path: Path to metadata.parquet.
        """
        self.ids = np.load(ids_path)

        self.metadata = (
            pd.read_parquet(metadata_path)
            .set_index("id")
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the most similar embeddings.

        Args:
            query_embedding: Query embedding of shape (1, embedding_dimension).
            top_k: Number of nearest neighbours to return.

        Returns:
            Tuple of (similarity_scores, indices).
        """

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must have shape (1, embedding_dimension)."
            )

        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k,
        )

        return scores, indices

    def get_products(
        self,
        indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Retrieve product metadata for FAISS search results.

        Args:
            indices: Array of FAISS indices.

        Returns:
            DataFrame containing product metadata.
        """

        if self.ids is None or self.metadata is None:
            raise RuntimeError(
                "Metadata has not been loaded. Call load_metadata() first."
            )

        product_ids = self.ids[indices]

        return self.metadata.loc[product_ids]