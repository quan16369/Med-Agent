"""
Example: Multimodal Medical Image Analysis with MedGemma
Demonstrates X-ray, CT, MRI analysis with AMG-RAG.
"""

from medassist import get_medgemma_llm
from medassist.core.multimodal_chains import MedicalImageAnalyzer, MultimodalReportGenerator
from medassist.core.knowledge_graph import MedicalKnowledgeGraph


def example_chest_xray_analysis():
    """
    Example: Analyze chest X-ray with MedGemma multimodal.
    """
    print("="*70)
    print("Example 1: Chest X-ray Analysis")
    print("="*70)
    
    # Initialize multimodal MedGemma
    llm = get_medgemma_llm(
        model="medgemma-1.5-4b",  # Multimodal model
        temperature=0.0
    )
    
    # Create image analyzer
    analyzer = MedicalImageAnalyzer(llm)
    
    # Analyze chest X-ray
    # NOTE: Replace with actual image path
    image_path = "path/to/chest_xray.jpg"
    
    result = analyzer.analyze_image(
        image_path=image_path,
        clinical_context="55-year-old male, chronic cough, fever x 3 days",
        modality_hint="chest_xray"
    )
    
    print(f"\nModality: {result['modality']}")
    print(f"Region: {result['anatomical_region']}")
    print(f"\nFindings:")
    for finding in result['findings']:
        print(f"  - {finding}")
    print(f"\nImpression: {result['impression']}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2f}")


def example_multimodal_kg():
    """
    Example: Build knowledge graph from medical image + clinical text.
    """
    print("\n" + "="*70)
    print("Example 2: Multimodal Knowledge Graph")
    print("="*70)
    
    # Initialize components
    llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
    analyzer = MedicalImageAnalyzer(llm)
    kg = MedicalKnowledgeGraph()
    
    # Extract entities from image
    image_path = "path/to/chest_xray.jpg"
    
    entities = analyzer.extract_entities_from_image(
        image_path=image_path,
        clinical_context="Patient with type 2 diabetes, suspected pneumonia",
        modality_hint="chest_xray"
    )
    
    # Add to knowledge graph
    for entity in entities:
        kg.add_entity(entity)
    
    # Show statistics
    stats = kg.get_statistics()
    print(f"\nKnowledge Graph:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Entity types: {stats['entity_types']}")
    
    print("\nExtracted entities:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type})")
        if entity.image_findings:
            print(f"    Findings: {', '.join(entity.image_findings[:3])}")


def example_longitudinal_comparison():
    """
    Example: Compare baseline vs follow-up images.
    """
    print("\n" + "="*70)
    print("Example 3: Longitudinal Image Comparison")
    print("="*70)
    
    llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
    analyzer = MedicalImageAnalyzer(llm)
    
    # Compare baseline and follow-up
    baseline_path = "path/to/baseline_xray.jpg"
    followup_path = "path/to/followup_xray.jpg"
    
    comparison = analyzer.compare_images(
        image_paths=[baseline_path, followup_path],
        comparison_type="temporal"
    )
    
    print("\nComparison Results:")
    print(f"Differences: {comparison.get('differences', [])}")
    print(f"Progression: {comparison.get('progression', 'N/A')}")
    print(f"Clinical Significance: {comparison.get('clinical_significance', 'N/A')}")
    print(f"Recommendation: {comparison.get('recommendation', 'N/A')}")


def example_comprehensive_report():
    """
    Example: Generate comprehensive report from multimodal data.
    """
    print("\n" + "="*70)
    print("Example 4: Comprehensive Multimodal Report")
    print("="*70)
    
    llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
    analyzer = MedicalImageAnalyzer(llm)
    report_gen = MultimodalReportGenerator(llm)
    
    # Analyze images
    chest_xray = analyzer.analyze_image(
        "path/to/chest_xray.jpg",
        clinical_context="Suspected pneumonia",
        modality_hint="chest_xray"
    )
    
    ct_scan = analyzer.analyze_image(
        "path/to/chest_ct.jpg",
        clinical_context="Follow-up CT",
        modality_hint="ct_scan"
    )
    
    # Generate comprehensive report
    report = report_gen.generate_report(
        clinical_history="""
        55-year-old male with history of type 2 diabetes.
        Presents with fever, productive cough x 3 days.
        Vitals: Temp 38.5°C, HR 95, RR 22, SpO2 94% on room air.
        """,
        imaging_findings=[chest_xray, ct_scan],
        other_data={
            "WBC": "15,000/μL",
            "CRP": "85 mg/L",
            "Procalcitonin": "2.5 ng/mL"
        }
    )
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MEDICAL REPORT")
    print("="*70)
    print(report)


def example_simple_image_qa():
    """
    Example: Simple Q&A about medical image.
    """
    print("\n" + "="*70)
    print("Example 5: Simple Image Q&A")
    print("="*70)
    
    from medassist.llm.medgemma import create_image_message
    
    llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
    
    # Ask question about image
    message = create_image_message(
        text="Is there evidence of pneumonia in this chest X-ray?",
        image_paths="path/to/chest_xray.jpg"
    )
    
    response = llm.invoke([message])
    print(f"\nAnswer: {response.content}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   Multimodal Medical Image Analysis with MedGemma               ║
║   Examples: X-ray, CT, MRI Analysis + Knowledge Graphs          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

NOTE: These examples require:
1. MedGemma multimodal model (medgemma-1.5-4b or medgemma-1-27b)
2. Actual medical image files (replace path/to/*.jpg with real paths)
3. Configured LLM provider (GOOGLE_API_KEY or Vertex AI)

To run:
1. Replace image paths with actual medical images
2. Configure: export GOOGLE_API_KEY='your-key'
3. Run: python example_multimodal.py

Available examples:
- example_chest_xray_analysis(): Single image analysis
- example_multimodal_kg(): Build KG from images
- example_longitudinal_comparison(): Compare baseline vs follow-up
- example_comprehensive_report(): Full multimodal report
- example_simple_image_qa(): Simple Q&A about images
""")
    
    # Uncomment to run (after adding real image paths):
    # example_chest_xray_analysis()
    # example_multimodal_kg()
    # example_longitudinal_comparison()
    # example_comprehensive_report()
    # example_simple_image_qa()
