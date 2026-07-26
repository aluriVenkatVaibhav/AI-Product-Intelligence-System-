from src.duplicate_detection import DuplicateDetector
from src.utils import visualize_duplicates


def main():

    detector = DuplicateDetector()

    product_id = 58808

    original, duplicates = detector.find_duplicates(
        product_id=product_id,
        top_k=5,
        similarity_threshold=0.85,
    )

    print("\n" + "=" * 70)
    print("ORIGINAL PRODUCT")
    print("=" * 70)

    print(f"Product ID : {original.product_id}")
    print(f"Name       : {original.product_name}")
    print(f"Category   : {original.master_category}")
    print(f"Article    : {original.article_type}")

    if not duplicates:

        print("\nNo visually similar products found.")

        return

    print("\n" + "=" * 70)
    print("POTENTIAL DUPLICATES")
    print("=" * 70)

    for rank, product in enumerate(duplicates, start=1):

        print(f"\nDuplicate #{rank}")

        print(f"Product ID : {product.product_id}")
        print(f"Name       : {product.product_name}")
        print(f"Category   : {product.master_category}")
        print(f"Article    : {product.article_type}")
        print(f"Similarity : {product.similarity:.4f}")

    visualize_duplicates(
        original=original,
        duplicates=duplicates,
    )


if __name__ == "__main__":
    main()