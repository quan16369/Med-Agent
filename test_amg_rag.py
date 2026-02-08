"""
Test AMG-RAG System with Sample MEDQA Question

Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
"""

import os
from medassist import AMG_RAG_System


def test_medqa_sample():
    """
    Test with a sample MEDQA question.
    """
    # Sample MEDQA question
    query = """
    A 55-year-old man with type 2 diabetes mellitus comes to the physician 
    because of a 2-month history of progressive numbness and tingling in his 
    feet and lower legs. He has had diabetes for 10 years and has been 
    non-compliant with his medications. His HbA1c is 10.2%. 
    
    What is the most likely underlying mechanism of his current symptoms?
    
    A) Schwann cell demyelination
    B) Axonal degeneration
    C) Autoimmune destruction of peripheral nerves
    D) Compression of nerve roots
    """
    
    print("="*80)
    print("AMG-RAG SYSTEM TEST - MEDQA Sample Question")
    print("="*80)
    print(f"\nQuery:\n{query}")
    print("\n" + "="*80)
    
    # Initialize AMG-RAG system (will auto-detect provider)
    try:
        system = AMG_RAG_System(
            model_name="medgemma-3-8b",
            temperature=0.0,
            pubmed_max_results=3,  # Limit for faster testing
            min_entity_relevance=5,
            enable_pubmed=True     # Set to False for quick test without PubMed
        )
        
        # Answer question
        result = system.answer_question(query, verbose=True)
        
        # Display results
        print("\n" + "="*80)
        print("EXTRACTED ENTITIES:")
        print("="*80)
        for i, entity in enumerate(result["entities"], 1):
            print(f"{i}. {entity}")
        
        print("\n" + "="*80)
        print("REASONING PATHS (sample):")
        print("="*80)
        for i, path in enumerate(result["reasoning_paths"][:5], 1):
            print(f"\nPath {i}:")
            print(path)
        
        print("\n" + "="*80)
        print("METADATA:")
        print("="*80)
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except RuntimeError as e:
        print(f"\n⚠️  ERROR: {e}")
        print("\nTo run this test, configure one of the following:")
        print("\n1. Google GenAI API:")
        print("   export GOOGLE_API_KEY='your-api-key'")
        print("\n2. Local Ollama with MedGemma:")
        print("   ollama pull medgemma")
        print("   ollama serve")
        print("\n3. Disable PubMed for quick test:")
        print("   Set enable_pubmed=False in AMG_RAG_System()")


def test_simple_query():
    """
    Test with a simple medical query (no multiple choice).
    """
    query = "What is the standard treatment for type 2 diabetes?"
    
    print("="*80)
    print("AMG-RAG SYSTEM TEST - Simple Query")
    print("="*80)
    print(f"\nQuery: {query}")
    print("\n" + "="*80)
    
    try:
        system = AMG_RAG_System(
            model_name="medgemma-3-8b",
            temperature=0.0,
            pubmed_max_results=5,
            enable_pubmed=True
        )
        
        result = system.answer_question(query, verbose=True)
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except RuntimeError as e:
        print(f"\n⚠️  ERROR: {e}")
        print("\nPlease configure an LLM provider (see test_medqa_sample for instructions)")


if __name__ == "__main__":
    import sys
    
    # Check for test type argument
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        test_simple_query()
    else:
        test_medqa_sample()
