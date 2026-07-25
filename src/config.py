from pathlib import Path
import torch

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
STYLE_CSV = DATASET_DIR / "styles.csv"

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# ==========================
# Embedding Files
# ==========================

IMAGE_EMBEDDINGS_FILE = EMBEDDING_DIR / "image_embeddings.npy"
IMAGE_IDS_FILE = EMBEDDING_DIR / "image_ids.npy"
METADATA_FILE = EMBEDDING_DIR / "metadata.parquet"

# ==========================
# Generation
# ==========================



NUM_WORKERS = 4

for folder in [
    EMBEDDING_DIR,
    VECTOR_DB_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================
# Model Configuration
# ==========================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ==========================
# Embedding Configuration
# ==========================

IMAGE_SIZE = 224
BATCH_SIZE = 128

# ==========================
# Vector Database
# ==========================

FAISS_INDEX_FILE = VECTOR_DB_DIR / "fashion_products.index"


RECOMMENDATION_RULES_FILE = (
    PROJECT_ROOT / "recommendations" / "recommendation_rules.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"