#!/usr/bin/env python3
"""
Quick test script for AMG-RAG components
"""

print("="*80)
print("AMG-RAG Component Test")
print("="*80)

# Test 1: Knowledge Graph
print("\n[1/3] Testing Medical Knowledge Graph...")
try:
    from medassist.models.knowledge_graph import MedicalKnowledgeGraph, MedicalEntity, MedicalRelationship
    
    kg = MedicalKnowledgeGraph(use_memory=True)
    kg.add_entity(MedicalEntity("d1", "Diabetes", "disease", ["DM"]))
    kg.add_entity(MedicalEntity("s1", "Numbness", "symptom", []))
    kg.add_relationship(MedicalRelationship("d1", "s1", "causes", 0.88, 75, "pubmed"))
    
    stats = kg.get_statistics()
    print(f"[OK] Knowledge Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
except Exception as e:
    print(f"[FAIL] Knowledge Graph: {e}")

# Test 2: Medical NER
print("\n[2/3] Testing Medical NER...")
try:
    from medassist.tools.medical_ner import BioBERTNER
    
    ner = BioBERTNER()
    entities = ner.extract("diabetes causes numbness")
    print(f"[OK] Medical NER: Found {len(entities)} entities")
    for e in entities[:3]:
        print(f"  - [{e.entity_type}] {e.text}")
except Exception as e:
    print(f"[FAIL] Medical NER: {e}")

# Test 3: Graph Retrieval
print("\n[3/3] Testing Graph Retrieval...")
try:
    from medassist.tools.graph_retrieval import GraphConditionalRetrieval
    
    retrieval = GraphConditionalRetrieval(kg)
    result = retrieval.retrieve("What causes numbness?")
    print(f"[OK] Graph Retrieval: Found {len(result.paths)} paths, confidence={result.confidence:.2%}")
except Exception as e:
    print(f"[FAIL] Graph Retrieval: {e}")

print("\n" + "="*80)
print("Component Test Complete")
print("="*80)
print("\nRun full demo: python demo_amg_rag.py")
