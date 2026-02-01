"""
Example usage of MedAssist system
Demonstrates how to use the agentic workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medassist import MedAssistOrchestrator
import json


def example_1_simple_case():
    """Example 1: Simple respiratory case"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Respiratory Case")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MedAssistOrchestrator(
        model_name="google/medgemma-2b",  # Will fall back to mock if not available
        device="auto",
        load_in_8bit=True
    )
    
    # Define patient case
    case = {
        "age": 45,
        "gender": "female",
        "symptoms": "persistent cough for 3 weeks, low-grade fever, night sweats",
        "history": "non-smoker, no chronic conditions",
        "medications": "",
        "allergies": ""
    }
    
    # Process case
    result = orchestrator.process_case(case, workflow="standard_diagnosis")
    
    # Display results
    print("\n📋 CASE SUMMARY:")
    print(f"Patient: {case['age']} year old {case['gender']}")
    print(f"Chief Complaint: {case['symptoms']}")
    
    print("\n🔍 DIAGNOSIS:")
    assessment = result['final_assessment']
    if assessment['diagnosis']:
        print(f"Primary: {assessment['diagnosis'].get('most_likely', 'N/A')}")
        print("\nDifferential:")
        for i, dx in enumerate(assessment['diagnosis'].get('differential', [])[:3], 1):
            print(f"  {i}. {dx}")
    
    print("\nTREATMENT PLAN:")
    if assessment['treatment_plan']:
        meds = assessment['treatment_plan'].get('medications', [])
        if meds:
            print("Medications:")
            for med in meds:
                if isinstance(med, dict):
                    print(f"  - {med.get('name', 'Unknown')}")
                else:
                    print(f"  - {med}")
    
    print(f"\n⏱️  Processing Time: {result['processing_time']:.2f}s")
    print()


def example_2_complex_case():
    """Example 2: Complex cardiac case"""
    print("=" * 60)
    print("EXAMPLE 2: Complex Cardiac Case")
    print("=" * 60)
    
    orchestrator = MedAssistOrchestrator()
    
    case = {
        "age": 67,
        "gender": "male",
        "symptoms": "chest pain radiating to left arm, shortness of breath, diaphoresis",
        "history": "hypertension, diabetes mellitus type 2, hyperlipidemia, former smoker (quit 5 years ago)",
        "medications": "Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily",
        "allergies": "None known",
        "vital_signs": "BP 150/95, HR 102, RR 22, O2 Sat 94% on RA"
    }
    
    # Use complex workflow
    result = orchestrator.process_case(case, workflow="complex_case")
    
    print("\n📋 CASE SUMMARY:")
    print(f"Patient: {case['age']} year old {case['gender']}")
    print(f"Chief Complaint: {case['symptoms']}")
    print(f"Vitals: {case['vital_signs']}")
    
    print("\n🔍 ASSESSMENT:")
    assessment = result['final_assessment']
    print(assessment.get('summary', 'No summary available'))
    
    print("\nURGENCY:")
    if 'diagnosis' in assessment and 'urgency' in assessment['diagnosis']:
        print(assessment['diagnosis']['urgency'])
    
    print(f"\n⏱️  Processing Time: {result['processing_time']:.2f}s")
    print()


def example_3_using_tools():
    """Example 3: Using medical calculation tools"""
    print("=" * 60)
    print("EXAMPLE 3: Medical Calculation Tools")
    print("=" * 60)
    
    from medassist.tools import MedicalCalculators, LabInterpreter
    
    # BMI Calculation
    print("\n📏 BMI Calculation:")
    bmi_result = MedicalCalculators.bmi(weight_kg=85, height_m=1.75)
    print(f"BMI: {bmi_result['bmi']}")
    print(f"Category: {bmi_result['category']}")
    print(f"Risk: {bmi_result['risk_assessment']}")
    
    # eGFR Calculation
    print("\n🩺 eGFR Calculation:")
    egfr_result = MedicalCalculators.egfr(
        creatinine_mg_dl=1.5,
        age=65,
        is_female=False,
        is_black=False
    )
    print(f"eGFR: {egfr_result['egfr']} {egfr_result['unit']}")
    print(f"CKD Stage: {egfr_result['ckd_stage']}")
    print(f"Description: {egfr_result['description']}")
    
    # Lab Interpretation
    print("\n🧪 Lab Panel Interpretation:")
    labs = {
        "wbc": 12.5,
        "hemoglobin": 11.0,
        "platelets": 180,
        "sodium": 138,
        "potassium": 4.2,
    }
    
    results = LabInterpreter.interpret_panel(labs, gender="male")
    for result in results:
        status = "OK" if result['interpretation'] == "Normal" else "ALERT"
        print(f"{status} {result['test']}: {result['value']} {result['unit']} - {result['interpretation']}")
    
    print()


def example_4_batch_processing():
    """Example 4: Batch processing multiple cases"""
    print("=" * 60)
    print("EXAMPLE 4: Batch Case Processing")
    print("=" * 60)
    
    orchestrator = MedAssistOrchestrator()
    
    cases = [
        {
            "id": "P001",
            "age": 35,
            "gender": "female",
            "symptoms": "severe headache, photophobia, nausea"
        },
        {
            "id": "P002",
            "age": 52,
            "gender": "male",
            "symptoms": "lower back pain radiating to right leg"
        },
        {
            "id": "P003",
            "age": 28,
            "gender": "female",
            "symptoms": "frequent urination, increased thirst, fatigue"
        }
    ]
    
    print(f"\nProcessing {len(cases)} cases...\n")
    
    results = []
    for case in cases:
        result = orchestrator.process_case(case, workflow="triage")
        results.append({
            "id": case["id"],
            "diagnosis": result['final_assessment']['diagnosis'].get('most_likely', 'Unknown'),
            "urgency": result['final_assessment']['diagnosis'].get('urgency', 'Routine'),
            "time": result['processing_time']
        })
    
    print("RESULTS:")
    print("-" * 60)
    for r in results:
        print(f"{r['id']}: {r['diagnosis']}")
        print(f"  Urgency: {r['urgency']}")
        print(f"  Time: {r['time']:.2f}s")
        print()
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average processing time: {avg_time:.2f}s per case")
    print()


def example_5_json_export():
    """Example 5: Exporting results as JSON"""
    print("=" * 60)
    print("EXAMPLE 5: JSON Export")
    print("=" * 60)
    
    orchestrator = MedAssistOrchestrator()
    
    case = {
        "age": 55,
        "gender": "male",
        "symptoms": "productive cough with yellow sputum, fever, chest pain on deep breathing",
        "history": "COPD, current smoker"
    }
    
    result = orchestrator.process_case(case)
    
    # Export to JSON
    output_file = "case_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults exported to {output_file}")
    print(f"  Case ID: {result['case_id']}")
    print(f"  Workflow: {result['workflow']}")
    print(f"  Steps completed: {len(result['steps'])}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MedAssist - Agentic Medical Workflow System")
    print("Example Usage Demonstrations")
    print("=" * 60 + "\n")
    
    try:
        example_1_simple_case()
        input("Press Enter to continue to next example...")
        
        example_2_complex_case()
        input("Press Enter to continue to next example...")
        
        example_3_using_tools()
        input("Press Enter to continue to next example...")
        
        example_4_batch_processing()
        input("Press Enter to continue to next example...")
        
        example_5_json_export()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
