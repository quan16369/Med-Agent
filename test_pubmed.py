"""
Test PubMed integration
"""

from medassist.tools.pubmed import PubMedSearcher
import os


def test_pubmed_search():
    """Test basic PubMed search"""
    
    # Initialize (with API key if available)
    api_key = os.getenv("PUBMED_API_KEY")
    pubmed = PubMedSearcher(api_key=api_key)
    
    # Test search
    query = "type 2 diabetes metformin treatment"
    print(f"Searching PubMed for: '{query}'")
    print()
    
    results = pubmed.search(query, max_results=3)
    
    if results:
        print(f"✓ Found {len(results)} articles")
        print()
        for i, abstract in enumerate(results, 1):
            print(f"Article {i}:")
            print(abstract[:200] + "...")
            print()
    else:
        print("✗ No results found (check internet connection or API limits)")


if __name__ == "__main__":
    print("="*60)
    print("PubMed Integration Test")
    print("="*60)
    print()
    
    test_pubmed_search()
    
    print()
    print("="*60)
    print("Test completed!")
    print("="*60)
