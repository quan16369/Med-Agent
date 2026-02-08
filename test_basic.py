"""
Test basic data structures
"""

from medassist.models import MedicalEntity, MedicalRelation, MEDICAL_RELATION_TYPES


def test_medical_entity():
    """Test MedicalEntity creation"""
    entity = MedicalEntity(
        name="Metformin",
        description="First-line medication for type 2 diabetes",
        entity_type="drug",
        confidence=0.9,
        sources=["PubMed"]
    )
    
    assert entity.name == "Metformin"
    assert entity.confidence == 0.9
    print(f"✓ Created entity: {entity.name}")


def test_medical_relation():
    """Test MedicalRelation creation"""
    relation = MedicalRelation(
        source="Metformin",
        target="Type 2 Diabetes",
        relation_type="treats",
        confidence=0.95,
        evidence="Metformin is the first-line treatment for type 2 diabetes",
        sources=["PubMed"]
    )
    
    assert relation.relation_type == "treats"
    assert relation.confidence == 0.95
    print(f"✓ Created relation: {relation.source} --[{relation.relation_type}]--> {relation.target}")


def test_relation_types():
    """Test available relationship types"""
    assert "treats" in MEDICAL_RELATION_TYPES
    assert "causes" in MEDICAL_RELATION_TYPES
    print(f"✓ Available relation types: {len(MEDICAL_RELATION_TYPES)}")
    for rel_type in MEDICAL_RELATION_TYPES:
        print(f"  - {rel_type}")


if __name__ == "__main__":
    print("="*50)
    print("AMG-RAG Data Structures Test")
    print("="*50)
    print()
    
    test_medical_entity()
    test_medical_relation()
    test_relation_types()
    
    print()
    print("="*50)
    print("All tests passed!")
    print("="*50)
