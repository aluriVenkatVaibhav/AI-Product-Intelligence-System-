from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SearchResult:
    """
    Represents one search result returned by the retrieval engine.
    """

    product_id: int
    product_name: str
    master_category: str
    sub_category: str
    article_type: str
    gender: str
    colour: str
    usage: str
    image_path: Path
    similarity: float