"""
Multimodal medical image processing chains for MedGemma.
Supports X-ray, CT, MRI, histopathology, dermatology image analysis.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from medassist.models.entities import MedicalEntity, MedicalRelation, IMAGING_MODALITIES
from medassist.llm.medgemma import create_image_message, is_multimodal_model


class ImagingFindingOutput(BaseModel):
    """Structured output for medical image analysis."""
    modality: str = Field(description="Imaging modality: chest_xray, ct_scan, mri, etc.")
    anatomical_region: str = Field(description="Anatomical region shown")
    findings: List[str] = Field(description="List of visible findings/abnormalities")
    impression: str = Field(description="Clinical impression/diagnosis")
    severity: str = Field(description="Severity: normal, mild, moderate, severe")
    confidence: float = Field(description="Confidence score from 0-1")


class MedicalImageAnalyzer:
    """
    Analyze medical images (X-ray, CT, MRI, etc.) using MedGemma multimodal models.
    Extracts findings, impressions, and creates medical entities from images.
    """
    
    IMAGE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an expert radiologist and medical image analyst. Analyze the provided medical image and provide a detailed report.

For the medical image, provide:
- modality: Type of imaging (chest_xray, ct_scan, mri, ultrasound, dermatology, histopathology, etc.)
- anatomical_region: Body part/region shown
- findings: List of all visible abnormalities or significant findings
- impression: Clinical impression and probable diagnosis
- severity: Assessment of severity (normal, mild, moderate, severe)
- confidence: Your confidence in the analysis (0-1)

Guidelines:
1. Be specific about anatomical locations
2. Use standard radiological terminology
3. Note both positive and clinically significant negative findings
4. Provide differential diagnoses when appropriate
5. State confidence level based on image quality and clarity of findings

Output as JSON."""),
        ("human", "{prompt}")
    ])
    
    def __init__(self, llm):
        """
        Initialize medical image analyzer.
        
        Args:
            llm: LangChain multimodal chat model (MedGemma 1.5 or MedGemma 1)
        """
        if not hasattr(llm, "model_name") or not is_multimodal_model(getattr(llm, "model_name", "")):
            print("Warning: LLM may not support multimodal input. Use medgemma-1.5-4b or medgemma-1-27b.")
        
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ImagingFindingOutput)
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        clinical_context: Optional[str] = None,
        modality_hint: Optional[str] = None
    ) -> Dict:
        """
        Analyze a medical image and extract findings.
        
        Args:
            image_path: Path to medical image file
            clinical_context: Optional clinical context/history
            modality_hint: Optional hint about imaging modality
            
        Returns:
            Dictionary with findings, impression, entities
        """
        try:
            # Build prompt
            prompt = "Analyze this medical image and provide a detailed radiological report."
            
            if modality_hint:
                prompt += f"\nImaging modality: {modality_hint}"
            
            if clinical_context:
                prompt += f"\nClinical context: {clinical_context}"
            
            # Create multimodal message
            message = create_image_message(
                text=prompt,
                image_paths=str(image_path),
                detail="high"
            )
            
            # Invoke model
            response = self.llm.invoke([message])
            
            # Parse response
            content = response.content if hasattr(response, "content") else str(response)
            
            # Try to parse as JSON
            import json
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Extract JSON from markdown or text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: return raw content
                    result = {
                        "modality": modality_hint or "unknown",
                        "anatomical_region": "unknown",
                        "findings": [content],
                        "impression": content,
                        "severity": "unknown",
                        "confidence": 0.5
                    }
            
            # Add image path to result
            result["image_path"] = str(image_path)
            
            return result
            
        except Exception as e:
            print(f"Image analysis failed: {e}")
            return {
                "modality": modality_hint or "unknown",
                "anatomical_region": "unknown",
                "findings": [],
                "impression": f"Analysis failed: {e}",
                "severity": "unknown",
                "confidence": 0.0,
                "image_path": str(image_path)
            }
    
    def extract_entities_from_image(
        self,
        image_path: Union[str, Path],
        clinical_context: Optional[str] = None,
        modality_hint: Optional[str] = None
    ) -> List[MedicalEntity]:
        """
        Extract medical entities from image analysis.
        
        Args:
            image_path: Path to medical image
            clinical_context: Optional clinical context
            modality_hint: Optional imaging modality hint
            
        Returns:
            List of MedicalEntity objects with imaging findings
        """
        # Analyze image first
        analysis = self.analyze_image(image_path, clinical_context, modality_hint)
        
        entities = []
        
        # Create entity for anatomical region
        if analysis.get("anatomical_region") and analysis["anatomical_region"] != "unknown":
            region_entity = MedicalEntity(
                name=analysis["anatomical_region"],
                description=f"Anatomical region visualized in {analysis['modality']}",
                entity_type="body_part",
                confidence=analysis.get("confidence", 0.5),
                sources=[f"image:{Path(image_path).name}"],
                image_path=str(image_path),
                image_findings=analysis.get("findings", [])
            )
            entities.append(region_entity)
        
        # Create entities for each finding
        for finding in analysis.get("findings", []):
            finding_entity = MedicalEntity(
                name=finding,
                description=f"Imaging finding: {finding}",
                entity_type="imaging_finding",
                confidence=analysis.get("confidence", 0.5),
                sources=[f"image:{Path(image_path).name}"],
                image_path=str(image_path),
                image_findings=[finding]
            )
            entities.append(finding_entity)
        
        # Create entity for impression/diagnosis if available
        if analysis.get("impression") and analysis["impression"] not in ["unknown", ""]:
            impression_entity = MedicalEntity(
                name=analysis["impression"],
                description=f"Clinical impression from {analysis['modality']} imaging",
                entity_type="disease",
                confidence=analysis.get("confidence", 0.5),
                sources=[f"image:{Path(image_path).name}"],
                image_path=str(image_path),
                image_findings=analysis.get("findings", [])
            )
            entities.append(impression_entity)
        
        return entities
    
    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        comparison_type: str = "temporal"
    ) -> Dict:
        """
        Compare multiple medical images (e.g., baseline vs follow-up).
        
        Args:
            image_paths: List of image paths to compare
            comparison_type: Type of comparison (temporal, different_views, different_modalities)
            
        Returns:
            Dictionary with comparison results
        """
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for comparison")
        
        try:
            prompt = f"Compare these medical images ({comparison_type} comparison):\n"
            prompt += "1. Identify key differences\n"
            prompt += "2. Note progression or changes\n"
            prompt += "3. Clinical significance of changes\n"
            prompt += "Output as JSON with: differences, progression, clinical_significance, recommendation"
            
            # Create message with multiple images
            message = create_image_message(
                text=prompt,
                image_paths=[str(p) for p in image_paths],
                detail="high"
            )
            
            response = self.llm.invoke([message])
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "differences": [content],
                    "progression": "Unable to parse",
                    "clinical_significance": "Unable to assess",
                    "recommendation": "Review by radiologist"
                }
            
            return result
            
        except Exception as e:
            print(f"Image comparison failed: {e}")
            return {
                "differences": [],
                "progression": f"Comparison failed: {e}",
                "clinical_significance": "Unable to assess",
                "recommendation": "Manual review required"
            }


class MultimodalReportGenerator:
    """
    Generate comprehensive medical reports combining text and image analysis.
    """
    
    REPORT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a medical report generator. Create a comprehensive clinical report combining:
1. Patient clinical history
2. Imaging findings
3. Laboratory results (if available)
4. Assessment and impression
5. Recommendations

Use standard medical report format."""),
        ("human", "Clinical History:\n{history}\n\nImaging Findings:\n{imaging}\n\nOther Data:\n{other}\n\nGenerate comprehensive report:")
    ])
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_report(
        self,
        clinical_history: str,
        imaging_findings: List[Dict],
        other_data: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive medical report.
        
        Args:
            clinical_history: Patient clinical history
            imaging_findings: List of imaging analysis results
            other_data: Optional additional data (labs, vitals, etc.)
            
        Returns:
            Formatted medical report
        """
        # Format imaging findings
        imaging_text = ""
        for i, finding in enumerate(imaging_findings, 1):
            imaging_text += f"\nStudy {i}: {finding.get('modality', 'Unknown')}\n"
            imaging_text += f"Region: {finding.get('anatomical_region', 'N/A')}\n"
            imaging_text += f"Findings:\n"
            for f in finding.get('findings', []):
                imaging_text += f"  - {f}\n"
            imaging_text += f"Impression: {finding.get('impression', 'N/A')}\n"
        
        # Format other data
        other_text = "None provided"
        if other_data:
            other_text = "\n".join([f"{k}: {v}" for k, v in other_data.items()])
        
        try:
            response = self.llm.invoke(
                self.REPORT_PROMPT.format(
                    history=clinical_history,
                    imaging=imaging_text,
                    other=other_text
                )
            )
            
            return response.content if hasattr(response, "content") else str(response)
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"
