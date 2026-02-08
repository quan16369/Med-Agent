"""
Example: Quick Start with AMG-RAG
Shows basic usage of the system with MedGemma.
"""

from medassist import AMG_RAG_System

# Example 1: Simple medical question
def example_basic():
    """Basic usage with automatic provider detection."""
    
    system = AMG_RAG_System(
        model_name="medgemma-3-8b",
        temperature=0.0,
        enable_pubmed=False  # Disable for quick test
    )
    
    result = system.answer_question(
        "What is the mechanism of action of metformin?"
    )
    
    print(result["answer"])


# Example 2: MEDQA-style question with PubMed
def example_medqa():
    """Full pipeline with evidence retrieval."""
    
    system = AMG_RAG_System(
        model_name="medgemma-3-8b",
        temperature=0.0,
        pubmed_max_results=5,
        enable_pubmed=True
    )
    
    query = """
    A 65-year-old woman with rheumatoid arthritis is started on methotrexate.
    What supplement should be prescribed with this medication?
    
    A) Calcium
    B) Folic acid
    C) Vitamin D
    D) Iron
    """
    
    result = system.answer_question(query)
    
    print(f"Entities: {result['entities']}")
    print(f"Answer: {result['answer']}")


# Example 3: Custom configuration
def example_custom():
    """Custom configuration for different use cases."""
    
    system = AMG_RAG_System(
        model_name="medgemma-3-27b",      # Larger model
        temperature=0.2,                   # Slight creativity
        pubmed_max_results=10,             # More evidence
        min_entity_relevance=7,            # Stricter filtering
        enable_pubmed=True
    )
    
    result = system.answer_question(
        "Explain the pathophysiology of diabetic neuropathy."
    )
    
    # Access knowledge graph
    kg = result["metadata"]["kg_stats"]
    print(f"Knowledge Graph: {kg['num_entities']} entities, {kg['num_relations']} relations")


if __name__ == "__main__":
    print("Example 1: Basic usage")
    print("="*70)
    example_basic()
    
    print("\n\nExample 2: MEDQA question with PubMed")
    print("="*70)
    example_medqa()
    
    print("\n\nExample 3: Custom configuration")
    print("="*70)
    example_custom()
