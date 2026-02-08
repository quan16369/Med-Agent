"""
Test SaraCoder-inspired optimization components.
"""

import sys
sys.path.insert(0, '/home/quan/MedGemma')

from medassist.core.hierarchical_optimizer import HierarchicalOptimizer
from medassist.core.medical_disambiguator import MedicalDisambiguator
from medassist.core.knowledge_graph import MedicalKnowledgeGraph
from medassist.models.entities import MedicalEntity


def test_disambiguator():
    """Test medical term disambiguation."""
    print("\n" + "="*70)
    print("TEST 1: Medical Term Disambiguation")
    print("="*70)
    
    disambiguator = MedicalDisambiguator()
    
    # Test 1: MI in cardiac context
    query1 = "Patient with chest pain, elevated troponin, ST elevation on ECG. Diagnosis: MI"
    sense1, conf1 = disambiguator.disambiguate("MI", query1)
    
    print(f"\n✓ Test 1.1: MI in cardiac context")
    print(f"  Query: {query1[:80]}...")
    print(f"  Disambiguation: MI → {sense1.full_name if sense1 else 'None'}")
    print(f"  Confidence: {conf1:.2f}")
    assert sense1 and "Myocardial" in sense1.full_name, "Should be Myocardial Infarction"
    
    # Test 2: MI in valvular context
    query2 = "Patient with heart murmur, echocardiography shows valve regurgitation. Diagnosis: MI"
    sense2, conf2 = disambiguator.disambiguate("MI", query2)
    
    print(f"\n✓ Test 1.2: MI in valvular context")
    print(f"  Query: {query2[:80]}...")
    print(f"  Disambiguation: MI → {sense2.full_name if sense2 else 'None'}")
    print(f"  Confidence: {conf2:.2f}")
    assert sense2 and "Mitral" in sense2.full_name, "Should be Mitral Insufficiency"
    
    # Test 3: Multiple abbreviations
    query3 = "Patient with RA and elevated CRP. PE shows no emboli. RA factor positive."
    expansions = disambiguator.expand_abbreviations(query3)
    
    print(f"\n✓ Test 1.3: Multiple abbreviations")
    print(f"  Query: {query3}")
    print(f"  Expansions: {expansions}")
    assert "RA" in expansions, "Should expand RA"
    assert "PE" in expansions, "Should expand PE"
    
    print("\n✅ Disambiguator tests PASSED")


def test_hierarchical_optimizer():
    """Test hierarchical entity optimization."""
    print("\n" + "="*70)
    print("TEST 2: Hierarchical Entity Optimization")
    print("="*70)
    
    optimizer = HierarchicalOptimizer(
        similarity_threshold=0.85,
        diversity_lambda=0.5,
        max_cluster_size=3
    )
    
    # Create test entities with duplicates
    entities = [
        {'name': 'Type 2 Diabetes', 'entity_type': 'disease', 'confidence': 0.9, 'relevance': 10},
        {'name': 'Diabetes Mellitus', 'entity_type': 'disease', 'confidence': 0.85, 'relevance': 9},  # Duplicate
        {'name': 'Hyperglycemia', 'entity_type': 'symptom', 'confidence': 0.8, 'relevance': 8},
        {'name': 'High Blood Sugar', 'entity_type': 'symptom', 'confidence': 0.75, 'relevance': 7},  # Similar
        {'name': 'Metformin', 'entity_type': 'drug', 'confidence': 0.9, 'relevance': 9},
        {'name': 'Insulin', 'entity_type': 'drug', 'confidence': 0.85, 'relevance': 8},
        {'name': 'HbA1c', 'entity_type': 'lab_test', 'confidence': 0.9, 'relevance': 9},
        {'name': 'Glucose Test', 'entity_type': 'lab_test', 'confidence': 0.7, 'relevance': 6},
        {'name': 'Diabetic Neuropathy', 'entity_type': 'disease', 'confidence': 0.8, 'relevance': 8},
        {'name': 'Retinopathy', 'entity_type': 'disease', 'confidence': 0.75, 'relevance': 7},
    ]
    
    print(f"\n✓ Test 2.1: Deduplication")
    print(f"  Input: {len(entities)} entities")
    deduplicated = optimizer.deduplicate_entities(entities)
    print(f"  Output: {len(deduplicated)} entities (deduplicated)")
    assert len(deduplicated) <= len(entities), "Should have same or fewer entities"
    
    print(f"\n✓ Test 2.2: Semantic Clustering")
    clusters = optimizer.semantic_clustering(deduplicated)
    print(f"  Clusters: {len(clusters)}")
    for cluster_id, cluster_entities in clusters.items():
        types = set(e['entity_type'] for e in cluster_entities)
        print(f"    Cluster {cluster_id}: {len(cluster_entities)} entities, types: {types}")
    
    print(f"\n✓ Test 2.3: Full Optimization")
    kg = MedicalKnowledgeGraph()
    optimized = optimizer.optimize_entities(
        deduplicated,
        knowledge_graph=kg,
        max_entities=6
    )
    print(f"  Input: {len(deduplicated)} entities")
    print(f"  Output: {len(optimized)} diverse entities")
    print(f"  Final entities:")
    for e in optimized:
        print(f"    - {e['name']} ({e['entity_type']})")
    
    assert len(optimized) <= 6, "Should not exceed max_entities"
    
    # Check diversity: should have multiple entity types
    types = set(e['entity_type'] for e in optimized)
    print(f"  Entity type diversity: {len(types)} types")
    assert len(types) >= 3, "Should have at least 3 different entity types"
    
    print("\n✅ Hierarchical optimizer tests PASSED")


def test_evidence_optimization():
    """Test evidence diversity optimization."""
    print("\n" + "="*70)
    print("TEST 3: Evidence Diversity Optimization")
    print("="*70)
    
    optimizer = HierarchicalOptimizer(diversity_lambda=0.5)
    
    # Create test evidence with redundant articles
    evidence = [
        {'title': 'Metformin in Type 2 Diabetes', 'source': 'pubmed', 'pmid': '001', 'relevance': 0.95},
        {'title': 'Metformin Mechanisms', 'source': 'pubmed', 'pmid': '002', 'relevance': 0.93},
        {'title': 'Metformin Side Effects', 'source': 'pubmed', 'pmid': '003', 'relevance': 0.90},
        {'title': 'Insulin Therapy Guidelines', 'source': 'pubmed', 'pmid': '004', 'relevance': 0.85},
        {'title': 'GLP-1 Agonists Review', 'source': 'pubmed', 'pmid': '005', 'relevance': 0.80},
        {'title': 'SGLT2 Inhibitors', 'source': 'pubmed', 'pmid': '006', 'relevance': 0.78},
        {'title': 'Diabetic Diet', 'source': 'pubmed', 'pmid': '007', 'relevance': 0.75},
        {'title': 'Exercise in Diabetes', 'source': 'pubmed', 'pmid': '008', 'relevance': 0.70},
    ]
    
    print(f"\n✓ Test 3.1: Evidence Optimization")
    print(f"  Input: {len(evidence)} articles")
    
    optimized = optimizer.optimize_evidence(evidence, max_evidence=5)
    
    print(f"  Output: {len(optimized)} diverse articles")
    print(f"  Selected:")
    for e in optimized:
        print(f"    - {e['title']}")
    
    assert len(optimized) == 5, "Should return exactly 5 articles"
    
    # Check that we don't have too many redundant Metformin articles
    metformin_count = sum(1 for e in optimized if 'Metformin' in e['title'])
    print(f"  Metformin articles: {metformin_count}/5")
    assert metformin_count <= 2, "Should not have more than 2 Metformin articles (diversity)"
    
    print("\n✅ Evidence optimization tests PASSED")


def test_integration_with_kg():
    """Test integration with knowledge graph."""
    print("\n" + "="*70)
    print("TEST 4: Integration with Knowledge Graph")
    print("="*70)
    
    # Create knowledge graph with entities
    kg = MedicalKnowledgeGraph()
    
    entities_data = [
        ('Type 2 Diabetes', 'disease'),
        ('Metformin', 'drug'),
        ('Hyperglycemia', 'symptom'),
        ('Insulin Resistance', 'condition'),
    ]
    
    for name, entity_type in entities_data:
        entity = MedicalEntity(name=name, entity_type=entity_type, confidence=0.9)
        kg.add_entity(entity)
    
    # Add relationships
    from medassist.models.entities import MedicalRelation
    
    relations = [
        ('Type 2 Diabetes', 'Hyperglycemia', 'causes'),
        ('Metformin', 'Type 2 Diabetes', 'treats'),
        ('Type 2 Diabetes', 'Insulin Resistance', 'associated_with'),
    ]
    
    for source, target, rel_type in relations:
        relation = MedicalRelation(
            source=source,
            target=target,
            relation_type=rel_type,
            confidence=0.8
        )
        kg.add_relation(relation)
    
    print(f"\n✓ Test 4.1: Knowledge Graph Setup")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relations: {len(kg.relations)}")
    
    # Test structural importance
    optimizer = HierarchicalOptimizer()
    
    entity_dicts = [
        {'name': name, 'entity_type': etype, 'confidence': 0.9, 'relevance': 8}
        for name, etype in entities_data
    ]
    
    importance = optimizer.compute_structural_importance(entity_dicts, kg)
    
    print(f"\n✓ Test 4.2: Structural Importance Scores")
    for name, score in importance.items():
        print(f"  {name}: {score:.3f}")
    
    # Type 2 Diabetes should have highest importance (most connected)
    assert importance['Type 2 Diabetes'] > importance['Hyperglycemia'], \
        "Type 2 Diabetes should be more important (more connections)"
    
    print("\n✅ Knowledge graph integration tests PASSED")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING SARACODER-INSPIRED OPTIMIZATION COMPONENTS")
    print("="*70)
    
    try:
        test_disambiguator()
        test_hierarchical_optimizer()
        test_evidence_optimization()
        test_integration_with_kg()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSaraCoder optimization components are working correctly:")
        print("  ✓ Medical term disambiguation")
        print("  ✓ Hierarchical entity optimization")
        print("  ✓ Evidence diversity optimization")
        print("  ✓ Knowledge graph integration")
        print("\nReady for integration with AMG-RAG workflow!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
