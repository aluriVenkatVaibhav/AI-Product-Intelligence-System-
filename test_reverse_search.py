from src.reverse_search import ReverseSearchEngine
from src.utils import visualize_search_results


def main():
    """
    Test Reverse Product Search.
    """

    engine = ReverseSearchEngine()

    query = "blue casual shirt"

    results = engine.search(
        query=query,
        top_k=6,
    )

    print("\n" + "=" * 70)
    print("REVERSE PRODUCT SEARCH")
    print("=" * 70)

    print(f"\nQuery: {query}")

    if not results:
        print("\nNo matching products found.")
        return

    for rank, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Result #{rank}")
        print("-" * 70)

        print(f"Product ID : {result.product_id}")
        print(f"Name       : {result.product_name}")
        print(f"Category   : {result.master_category}")
        print(f"Article    : {result.article_type}")
        print(f"Gender     : {result.gender}")
        print(f"Colour     : {result.colour}")
        print(f"Usage      : {result.usage}")
        print(f"Similarity : {result.similarity:.4f}")

    visualize_search_results(
        results=results,
        query=query,
    )


if __name__ == "__main__":
    main()