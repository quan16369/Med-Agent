"""
AMG-RAG Demo - Simple standalone script
Medical Knowledge Graph QA System based on EMNLP 2025 paper
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.models.knowledge_graph import MedicalKnowledgeGraph, MedicalEntity, MedicalRelationship
from medassist.tools.graph_retrieval import GraphConditionalRetrieval
from medassist.tools.medical_ner import BioBERTNER


def create_sample_medical_graph():
    """Create sample medical knowledge graph"""
    
    kg = MedicalKnowledgeGraph(use_memory=True)
    
    # Diseases
    kg.add_entity(MedicalEntity("disease_diabetes", "Diabetes Mellitus", "disease", 
                                ["diabetes", "DM", "type 2 diabetes"]))
    kg.add_entity(MedicalEntity("disease_hypertension", "Hypertension", "disease",
                                ["high blood pressure", "HTN"]))
    kg.add_entity(MedicalEntity("disease_neuropathy", "Diabetic Neuropathy", "disease",
                                ["nerve damage", "peripheral neuropathy"]))
    kg.add_entity(MedicalEntity("disease_retinopathy", "Diabetic Retinopathy", "disease",
                                ["eye damage", "diabetic eye disease"]))
    kg.add_entity(MedicalEntity("disease_chd", "Coronary Heart Disease", "disease",
                                ["CHD", "heart disease", "coronary artery disease"]))
    kg.add_entity(MedicalEntity("disease_kidney", "Diabetic Nephropathy", "disease",
                                ["kidney disease", "diabetic kidney disease"]))
    
    # Symptoms
    kg.add_entity(MedicalEntity("symptom_hyperglycemia", "Hyperglycemia", "symptom",
                                ["high blood sugar", "elevated glucose"]))
    kg.add_entity(MedicalEntity("symptom_numbness", "Numbness", "symptom",
                                ["tingling", "loss of sensation"]))
    kg.add_entity(MedicalEntity("symptom_blurred_vision", "Blurred Vision", "symptom",
                                ["vision problems", "vision loss"]))
    kg.add_entity(MedicalEntity("symptom_chest_pain", "Chest Pain", "symptom",
                                ["angina", "chest discomfort"]))
    kg.add_entity(MedicalEntity("symptom_fatigue", "Fatigue", "symptom",
                                ["tiredness", "weakness"]))
    kg.add_entity(MedicalEntity("symptom_thirst", "Excessive Thirst", "symptom",
                                ["polydipsia", "increased thirst"]))
    
    # Treatments
    kg.add_entity(MedicalEntity("treatment_metformin", "Metformin", "treatment",
                                ["glucophage"]))
    kg.add_entity(MedicalEntity("treatment_insulin", "Insulin", "treatment",
                                ["insulin therapy"]))
    kg.add_entity(MedicalEntity("treatment_statin", "Statin", "treatment",
                                ["atorvastatin", "cholesterol medication"]))
    kg.add_entity(MedicalEntity("treatment_ace_inhibitor", "ACE Inhibitor", "treatment",
                                ["lisinopril", "blood pressure medication"]))
    
    # Biomarkers
    kg.add_entity(MedicalEntity("biomarker_hba1c", "HbA1c", "biomarker",
                                ["hemoglobin a1c", "glycated hemoglobin"]))
    kg.add_entity(MedicalEntity("biomarker_glucose", "Blood Glucose", "biomarker",
                                ["blood sugar", "glucose level"]))
    
    # Relationships - Diabetes causes complications
    kg.add_relationship(MedicalRelationship("disease_diabetes", "symptom_hyperglycemia", 
                                           "causes", 0.95, 150, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_diabetes", "symptom_thirst",
                                           "causes", 0.90, 100, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_diabetes", "disease_neuropathy",
                                           "causes", 0.88, 75, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_diabetes", "disease_retinopathy",
                                           "causes", 0.85, 80, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_diabetes", "disease_kidney",
                                           "causes", 0.83, 70, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_diabetes", "disease_chd",
                                           "increases_risk", 0.82, 120, "pubmed"))
    
    # Neuropathy causes symptoms
    kg.add_relationship(MedicalRelationship("disease_neuropathy", "symptom_numbness",
                                           "causes", 0.92, 60, "pubmed"))
    kg.add_relationship(MedicalRelationship("disease_neuropathy", "symptom_fatigue",
                                           "causes", 0.75, 40, "pubmed"))
    
    # Retinopathy causes symptoms
    kg.add_relationship(MedicalRelationship("disease_retinopathy", "symptom_blurred_vision",
                                           "causes", 0.90, 55, "pubmed"))
    
    # CHD causes symptoms
    kg.add_relationship(MedicalRelationship("disease_chd", "symptom_chest_pain",
                                           "causes", 0.93, 100, "pubmed"))
    
    # Treatments
    kg.add_relationship(MedicalRelationship("treatment_metformin", "disease_diabetes",
                                           "treats", 0.97, 200, "pubmed"))
    kg.add_relationship(MedicalRelationship("treatment_metformin", "symptom_hyperglycemia",
                                           "reduces", 0.94, 180, "pubmed"))
    kg.add_relationship(MedicalRelationship("treatment_insulin", "disease_diabetes",
                                           "treats", 0.98, 250, "pubmed"))
    kg.add_relationship(MedicalRelationship("treatment_statin", "disease_chd",
                                           "treats", 0.91, 150, "pubmed"))
    kg.add_relationship(MedicalRelationship("treatment_ace_inhibitor", "disease_hypertension",
                                           "treats", 0.93, 160, "pubmed"))
    
    # Biomarker relationships
    kg.add_relationship(MedicalRelationship("biomarker_hba1c", "disease_diabetes",
                                           "diagnoses", 0.96, 300, "pubmed"))
    kg.add_relationship(MedicalRelationship("biomarker_glucose", "symptom_hyperglycemia",
                                           "indicates", 0.98, 250, "pubmed"))
    
    return kg


def demo_query(kg, retrieval, query):
    """Process and display a query"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    # Retrieve knowledge
    result = retrieval.retrieve(query)
    
    if not result.paths:
        print("[FAIL] No knowledge paths found")
        return
    
    # Display results
    print(f"\n[OK] Found {len(result.paths)} knowledge paths")
    print(f"[OK] Confidence: {result.confidence:.2%}")
    print(f"[OK] Query entities: {', '.join(result.query_entities)}")
    
    print(f"\nTop Knowledge Paths:")
    for i, path in enumerate(result.paths[:5], 1):
        print(f"\n{i}. {path.to_string()}")
        print(f"   Confidence: {path.confidence:.2%}")
    
    # Generate context
    context = result.to_context(max_paths=3)
    print(f"\n📝 Generated Context for LLM:")
    print(context)


def main():
    print("="*80)
    print("Medical Knowledge Graph Demo")
    print("Graph-based reasoning for medical QA")
    print("="*80)
    
    # Create knowledge graph
    print("\nInitializing Medical Knowledge Graph...")
    kg = create_sample_medical_graph()
    
    stats = kg.get_statistics()
    print(f"[OK] Loaded {stats['num_nodes']} nodes and {stats['num_edges']} edges")
    print(f"[OK] Entity types: {stats['entity_types']}")
    print(f"[OK] Relationship types: {stats['relationship_types']}")
    
    # Initialize retrieval agent
    print("\nInitializing Graph Retrieval Agent...")
    retrieval = GraphConditionalRetrieval(
        knowledge_graph=kg,
        max_hops=3,
        min_confidence=0.5,
        top_k_paths=10
    )
    print("[OK] Retrieval agent ready")
    
    # Test queries
    test_queries = [
        "What causes numbness in diabetic patients?",
        "How does diabetes affect vision?",
        "What are the treatments for diabetes?",
        "What is the relationship between diabetes and heart disease?",
        "How is diabetes diagnosed?"
    ]
    
    print("\n" + "="*80)
    print("Running Test Queries")
    print("="*80)
    
    for query in test_queries:
        demo_query(kg, retrieval, query)
    
    # Final statistics
    print("\n" + "="*80)
    print("System Statistics")
    print("="*80)
    print(f"Knowledge Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    print(f"Retrieval Config: max_hops={retrieval.max_hops}, min_confidence={retrieval.min_confidence}")
    print(f"Multi-hop reasoning with confidence scoring")
    print("="*80)
    
    print("\n[OK] Demo completed successfully!")


if __name__ == "__main__":
    main()
