# 🎨 Multimodal Medical Image Analysis

MedAssist now supports **multimodal medical image analysis** using MedGemma's vision capabilities!

## 🔬 Supported Imaging Modalities

- **Chest X-ray (CXR)**: Pneumonia, tuberculosis, lung cancer detection
- **CT Scan**: Tumor analysis, organ assessment, trauma evaluation
- **MRI**: Brain imaging, soft tissue analysis, spinal cord evaluation
- **Ultrasound**: Fetal imaging, organ assessment, vascular studies
- **PET Scan**: Cancer staging, metabolic activity
- **Mammography**: Breast cancer screening
- **Dermatology**: Skin lesion classification, melanoma detection
- **Histopathology**: Cancer diagnosis, tissue analysis
- **Fundoscopy**: Retinal imaging, diabetic retinopathy
- **Endoscopy**: GI tract visualization

## 🚀 Quick Start

### 1. Analyze Single Medical Image

```python
from medassist import get_medgemma_llm
from medassist.core.multimodal_chains import MedicalImageAnalyzer

# Initialize multimodal MedGemma (1.5 4B or 1 27B)
llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
analyzer = MedicalImageAnalyzer(llm)

# Analyze chest X-ray
result = analyzer.analyze_image(
    image_path="chest_xray.jpg",
    clinical_context="55-year-old male, fever, productive cough x 3 days",
    modality_hint="chest_xray"
)

print(f"Modality: {result['modality']}")
print(f"Region: {result['anatomical_region']}")
print(f"Findings: {result['findings']}")
print(f"Impression: {result['impression']}")
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']}")
```

### 2. Build Knowledge Graph from Images

```python
from medassist.core.knowledge_graph import MedicalKnowledgeGraph

# Extract entities from medical image
entities = analyzer.extract_entities_from_image(
    image_path="chest_xray.jpg",
    clinical_context="Patient with diabetes, suspected pneumonia"
)

# Add to knowledge graph
kg = MedicalKnowledgeGraph()
for entity in entities:
    kg.add_entity(entity)

# Show statistics
stats = kg.get_statistics()
print(f"Entities: {stats['num_entities']}")
print(f"Relations: {stats['num_relations']}")
```

### 3. Compare Baseline vs Follow-up Images

```python
# Longitudinal comparison
comparison = analyzer.compare_images(
    image_paths=["baseline_xray.jpg", "followup_xray.jpg"],
    comparison_type="temporal"
)

print(f"Differences: {comparison['differences']}")
print(f"Progression: {comparison['progression']}")
print(f"Clinical Significance: {comparison['clinical_significance']}")
print(f"Recommendation: {comparison['recommendation']}")
```

### 4. Generate Comprehensive Multimodal Report

```python
from medassist.core.multimodal_chains import MultimodalReportGenerator

# Analyze multiple imaging studies
chest_xray = analyzer.analyze_image("chest_xray.jpg", modality_hint="chest_xray")
ct_scan = analyzer.analyze_image("chest_ct.jpg", modality_hint="ct_scan")

# Generate report
report_gen = MultimodalReportGenerator(llm)
report = report_gen.generate_report(
    clinical_history="""
    55-year-old male with history of type 2 diabetes.
    Presents with fever, productive cough x 3 days.
    """,
    imaging_findings=[chest_xray, ct_scan],
    other_data={
        "WBC": "15,000/μL",
        "CRP": "85 mg/L"
    }
)

print(report)
```

### 5. Simple Image Q&A

```python
from medassist.llm.medgemma import create_image_message

llm = get_medgemma_llm(model="medgemma-1.5-4b")

# Ask question about image
message = create_image_message(
    text="Is there evidence of pneumonia in this chest X-ray?",
    image_paths="chest_xray.jpg"
)

response = llm.invoke([message])
print(response.content)
```

## 🏗️ Architecture

### Multimodal Entity Types

```python
from medassist.models.entities import MedicalEntity

entity = MedicalEntity(
    name="Right lower lobe consolidation",
    description="Dense opacity in right lower lobe",
    entity_type="imaging_finding",  # New type for image findings
    confidence=0.8,
    sources=["image:chest_xray.jpg"],
    image_path="chest_xray.jpg",  # Link to source image
    image_findings=["consolidation", "air bronchograms"]  # Specific findings
)
```

### Multimodal Relationships

```python
from medassist.models.entities import MedicalRelation

relation = MedicalRelation(
    source="pneumonia",
    target="chest_xray",
    relation_type="visualized_in",  # New relation type
    confidence=0.9,
    evidence="Right lower lobe consolidation consistent with pneumonia",
    sources=["image:chest_xray.jpg"]
)
```

## 🎯 Use Cases

### Clinical Diagnosis

```python
# Case: 65yo female with chronic cough
llm = get_medgemma_llm(model="medgemma-1.5-4b")
analyzer = MedicalImageAnalyzer(llm)

# Analyze chest X-ray
result = analyzer.analyze_image(
    image_path="patient_cxr.jpg",
    clinical_context="65yo female, chronic cough x 6 months, weight loss",
    modality_hint="chest_xray"
)

# Extract entities and build knowledge graph
entities = analyzer.extract_entities_from_image("patient_cxr.jpg")
kg = MedicalKnowledgeGraph()
for entity in entities:
    kg.add_entity(entity)

# Explore reasoning paths
paths = kg.explore_paths("lung_mass", "lung_cancer", max_length=3)
```

### Treatment Monitoring

```python
# Compare baseline vs follow-up after treatment
baseline = analyzer.analyze_image("baseline.jpg", modality_hint="chest_xray")
month1 = analyzer.analyze_image("month1.jpg", modality_hint="chest_xray")
month3 = analyzer.analyze_image("month3.jpg", modality_hint="chest_xray")

comparison = analyzer.compare_images(
    image_paths=["baseline.jpg", "month1.jpg", "month3.jpg"],
    comparison_type="temporal"
)

print(f"Treatment response: {comparison['progression']}")
```

### Screening Programs

```python
# Batch analyze screening images
screening_images = [
    "patient001_mammo.jpg",
    "patient002_mammo.jpg",
    "patient003_mammo.jpg"
]

for image_path in screening_images:
    result = analyzer.analyze_image(
        image_path=image_path,
        modality_hint="mammography"
    )
    
    if result['severity'] in ['moderate', 'severe']:
        print(f"⚠️  {image_path}: {result['impression']}")
        print(f"   Findings: {result['findings']}")
```

## 📊 Competition Advantages

### Main Track
- **Multimodal clinical decision support**: Text + images
- **Real clinical workflow**: 80% of diagnostics involve imaging
- **Evidence-based**: Direct visual evidence integrated into knowledge graph

### Agentic Workflow Prize
- **Image analysis agents**: Autonomous radiological interpretation
- **Multi-stage reasoning**: Extract → Analyze → Compare → Reason
- **Dynamic decision-making**: Adapts based on image findings

### Edge AI Prize
- **Quantized models**: MedGemma 1.5 4B runs on tablets/edge devices
- **Local inference**: HIPAA-compliant, privacy-preserving
- **Point-of-care**: Emergency room, rural clinics, ambulances

### Novel Task Prize
- **Multimodal KG reasoning**: First AMG-RAG with vision
- **Longitudinal analysis**: Temporal image comparison
- **Cross-modal synthesis**: Text + image + labs integrated

## 🔬 Best Practices

### Image Quality
```python
# Check image quality before analysis
from PIL import Image

img = Image.open("chest_xray.jpg")
width, height = img.size

if width < 512 or height < 512:
    print("⚠️  Warning: Low resolution image may affect accuracy")
```

### Clinical Context
```python
# Always provide clinical context for better analysis
result = analyzer.analyze_image(
    image_path="chest_xray.jpg",
    clinical_context="""
    Age: 65
    Sex: Male
    Symptoms: Fever (38.5°C), productive cough x 3 days
    PMH: Type 2 diabetes, hypertension
    Medications: Metformin, lisinopril
    """,
    modality_hint="chest_xray"
)
```

### Confidence Thresholding
```python
# Only use high-confidence findings for clinical decisions
if result['confidence'] >= 0.8:
    print(f"High confidence: {result['impression']}")
else:
    print("Low confidence - recommend radiologist review")
```

## 🎓 Model Selection

| Model | Parameters | Use Case | Speed | Accuracy |
|-------|-----------|----------|-------|----------|
| **MedGemma 1.5 4B** | 4B | Best multimodal, fast | Fast | High |
| **MedGemma 1 4B** | 4B | Legacy multimodal | Fast | Good |
| **MedGemma 1 27B** | 27B | Advanced reasoning | Slower | Best |

### Recommended Setup

**Development/Testing**:
```python
llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
```

**Production (High Accuracy)**:
```python
llm = get_medgemma_llm(model="medgemma-1-27b", temperature=0.0)
```

**Edge Deployment**:
```python
# Quantized MedGemma 1.5 4B (8-bit or 4-bit)
llm = get_medgemma_llm(
    model="medgemma-1.5-4b",
    provider="ollama",  # Local Ollama with quantization
    temperature=0.0
)
```

## 📚 Medical Imaging Datasets

For testing and evaluation:

1. **ChestX-ray14** - 100K+ chest X-rays with 14 disease labels
2. **MIMIC-CXR** - 377K chest X-rays with radiology reports
3. **CheXpert** - 224K chest radiographs
4. **RSNA Pneumonia Detection** - Chest X-ray pneumonia dataset
5. **ISIC** - Dermatology skin lesion images
6. **PathMNIST** - Histopathology patches

## 🚨 Important Notes

### Clinical Use Warning
```
⚠️  This system is for research and development purposes only.
Not approved for clinical decision-making without physician review.
Always validate findings with board-certified radiologists.
```

### HIPAA Compliance
- Use local/edge deployment for patient data
- Never send PHI to cloud APIs without proper consent
- Implement appropriate access controls and audit logging

### Limitations
- Image quality affects accuracy
- Context is critical for proper interpretation
- Some rare conditions may not be recognized
- Always provide clinical correlation

## 🔮 Future Enhancements

- [ ] **MedSigLIP integration**: Medical image embeddings for similarity search
- [ ] **3D image support**: Full CT/MRI volume analysis
- [ ] **Region-of-interest**: Highlight abnormal areas in images
- [ ] **DICOM support**: Native medical image format handling
- [ ] **Automated measurements**: Tumor size, organ volumes
- [ ] **Structured reports**: Automatic BIRADS/RADS scoring

## 📖 Examples

See [example_multimodal.py](example_multimodal.py) for complete examples.

## 🤝 Contributing

After competition (Feb 24, 2026), PRs welcome for:
- Additional imaging modalities
- Improved prompts
- Clinical validation studies
- Edge deployment optimizations

---

**Competition**: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)  
**Deadline**: February 24, 2026  
**Category**: Main Track + Agentic Workflow + Edge AI + Novel Task
