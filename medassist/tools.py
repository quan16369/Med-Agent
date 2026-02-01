"""
Medical Tools and Utilities
Tools that agents can use for calculations, lookups, and specialized functions
"""

from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class MedicalCalculators:
    """Collection of medical calculators"""
    
    @staticmethod
    def bmi(weight_kg: float, height_m: float) -> Dict:
        """Calculate Body Mass Index"""
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
            risk = "Increased risk of malnutrition"
        elif bmi < 25:
            category = "Normal weight"
            risk = "Low risk"
        elif bmi < 30:
            category = "Overweight"
            risk = "Increased risk of cardiovascular disease"
        elif bmi < 35:
            category = "Obese Class I"
            risk = "Moderate risk"
        elif bmi < 40:
            category = "Obese Class II"
            risk = "High risk"
        else:
            category = "Obese Class III"
            risk = "Very high risk"
        
        return {
            "bmi": round(bmi, 1),
            "category": category,
            "risk_assessment": risk
        }
    
    @staticmethod
    def egfr(
        creatinine_mg_dl: float,
        age: int,
        is_female: bool,
        is_black: bool = False
    ) -> Dict:
        """Calculate eGFR using CKD-EPI equation"""
        kappa = 0.7 if is_female else 0.9
        alpha = -0.329 if is_female else -0.411
        
        min_val = min(creatinine_mg_dl / kappa, 1)
        max_val = max(creatinine_mg_dl / kappa, 1)
        
        egfr = 141 * (min_val ** alpha) * (max_val ** -1.209) * (0.993 ** age)
        
        if is_female:
            egfr *= 1.018
        if is_black:
            egfr *= 1.159
        
        # CKD stage
        if egfr >= 90:
            stage = "G1"
            description = "Normal or high kidney function"
        elif egfr >= 60:
            stage = "G2"
            description = "Mildly decreased kidney function"
        elif egfr >= 45:
            stage = "G3a"
            description = "Mild to moderate decrease"
        elif egfr >= 30:
            stage = "G3b"
            description = "Moderate to severe decrease"
        elif egfr >= 15:
            stage = "G4"
            description = "Severely decreased"
        else:
            stage = "G5"
            description = "Kidney failure"
        
        return {
            "egfr": round(egfr, 1),
            "unit": "mL/min/1.73m²",
            "ckd_stage": stage,
            "description": description
        }
    
    @staticmethod
    def chads2_vasc(
        age: int,
        is_female: bool,
        has_chf: bool = False,
        has_hypertension: bool = False,
        has_diabetes: bool = False,
        has_stroke_tia: bool = False,
        has_vascular_disease: bool = False
    ) -> Dict:
        """Calculate CHA2DS2-VASc score for stroke risk in atrial fibrillation"""
        score = 0
        
        # Age
        if age >= 75:
            score += 2
        elif age >= 65:
            score += 1
        
        # Female
        if is_female:
            score += 1
        
        # Conditions
        if has_chf:
            score += 1
        if has_hypertension:
            score += 1
        if has_diabetes:
            score += 1
        if has_stroke_tia:
            score += 2
        if has_vascular_disease:
            score += 1
        
        # Risk interpretation
        if score == 0:
            risk = "Low"
            recommendation = "No antithrombotic therapy recommended"
        elif score == 1:
            risk = "Low-Moderate"
            recommendation = "Consider oral anticoagulation or aspirin"
        else:
            risk = "Moderate-High"
            recommendation = "Oral anticoagulation recommended"
        
        return {
            "score": score,
            "risk_category": risk,
            "recommendation": recommendation
        }
    
    @staticmethod
    def wells_dvt(
        active_cancer: bool = False,
        paralysis_or_immobilization: bool = False,
        bedridden_or_surgery: bool = False,
        tenderness_along_veins: bool = False,
        entire_leg_swollen: bool = False,
        calf_swelling: bool = False,
        pitting_edema: bool = False,
        collateral_veins: bool = False,
        alternative_diagnosis: bool = False
    ) -> Dict:
        """Wells score for DVT probability"""
        score = 0
        
        if active_cancer:
            score += 1
        if paralysis_or_immobilization:
            score += 1
        if bedridden_or_surgery:
            score += 1
        if tenderness_along_veins:
            score += 1
        if entire_leg_swollen:
            score += 1
        if calf_swelling:
            score += 1
        if pitting_edema:
            score += 1
        if collateral_veins:
            score += 1
        if alternative_diagnosis:
            score -= 2
        
        if score <= 0:
            probability = "Low"
            dvt_risk = "5%"
        elif score <= 2:
            probability = "Moderate"
            dvt_risk = "17%"
        else:
            probability = "High"
            dvt_risk = "53%"
        
        return {
            "score": score,
            "probability": probability,
            "dvt_risk": dvt_risk,
            "recommendation": "Consider D-dimer if low, imaging if moderate/high"
        }
    
    @staticmethod
    def framingham_risk(
        age: int,
        is_female: bool,
        total_cholesterol: float,
        hdl_cholesterol: float,
        systolic_bp: int,
        is_smoker: bool = False,
        has_diabetes: bool = False,
        on_bp_meds: bool = False
    ) -> Dict:
        """
        Simplified Framingham Risk Score for 10-year cardiovascular risk
        Note: This is a simplified version
        """
        # This is a simplified calculation
        # In production, use the full Framingham equation
        
        risk_points = 0
        
        # Age
        if is_female:
            if age < 35:
                risk_points -= 7
            elif age < 40:
                risk_points -= 3
            elif age < 45:
                risk_points += 0
            elif age < 50:
                risk_points += 3
            elif age < 55:
                risk_points += 6
            elif age < 60:
                risk_points += 8
            elif age < 65:
                risk_points += 10
            elif age < 70:
                risk_points += 12
            else:
                risk_points += 14
        else:
            if age < 35:
                risk_points -= 9
            elif age < 40:
                risk_points -= 4
            elif age < 45:
                risk_points += 0
            elif age < 50:
                risk_points += 3
            elif age < 55:
                risk_points += 6
            elif age < 60:
                risk_points += 8
            elif age < 65:
                risk_points += 10
            elif age < 70:
                risk_points += 11
            else:
                risk_points += 12
        
        # Simplified cholesterol scoring
        if total_cholesterol > 240:
            risk_points += 2
        elif total_cholesterol > 200:
            risk_points += 1
        
        if hdl_cholesterol < 40:
            risk_points += 2
        elif hdl_cholesterol > 60:
            risk_points -= 1
        
        # Blood pressure
        if systolic_bp > 160:
            risk_points += 2
        elif systolic_bp > 140:
            risk_points += 1
        
        # Risk factors
        if is_smoker:
            risk_points += 2
        if has_diabetes:
            risk_points += 2
        
        # Estimate risk percentage (simplified)
        if risk_points < 0:
            risk_percent = "<1%"
            category = "Low"
        elif risk_points < 5:
            risk_percent = "1-5%"
            category = "Low"
        elif risk_points < 10:
            risk_percent = "5-10%"
            category = "Intermediate"
        elif risk_points < 15:
            risk_percent = "10-20%"
            category = "High"
        else:
            risk_percent = ">20%"
            category = "Very High"
        
        return {
            "risk_points": risk_points,
            "ten_year_risk": risk_percent,
            "risk_category": category,
            "recommendation": "Consider statin therapy if intermediate or high risk"
        }


class LabInterpreter:
    """Interpret lab values"""
    
    # Reference ranges (example values)
    REFERENCE_RANGES = {
        "wbc": {"low": 4.5, "high": 11.0, "unit": "K/μL"},
        "hemoglobin": {
            "male": {"low": 13.5, "high": 17.5},
            "female": {"low": 12.0, "high": 15.5},
            "unit": "g/dL"
        },
        "platelets": {"low": 150, "high": 400, "unit": "K/μL"},
        "sodium": {"low": 135, "high": 145, "unit": "mEq/L"},
        "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L"},
        "glucose": {"low": 70, "high": 100, "unit": "mg/dL (fasting)"},
        "creatinine": {
            "male": {"low": 0.7, "high": 1.3},
            "female": {"low": 0.6, "high": 1.1},
            "unit": "mg/dL"
        },
        "alt": {"low": 7, "high": 56, "unit": "U/L"},
        "ast": {"low": 10, "high": 40, "unit": "U/L"},
    }
    
    @staticmethod
    def interpret_lab(
        test_name: str,
        value: float,
        gender: str = "male"
    ) -> Dict:
        """Interpret a lab value"""
        test_name = test_name.lower()
        
        if test_name not in LabInterpreter.REFERENCE_RANGES:
            return {
                "test": test_name,
                "value": value,
                "interpretation": "Reference range not available"
            }
        
        ref_range = LabInterpreter.REFERENCE_RANGES[test_name]
        
        # Handle gender-specific ranges
        if isinstance(ref_range.get("male"), dict):
            if gender.lower() == "female":
                low = ref_range["female"]["low"]
                high = ref_range["female"]["high"]
            else:
                low = ref_range["male"]["low"]
                high = ref_range["male"]["high"]
            unit = ref_range["unit"]
        else:
            low = ref_range["low"]
            high = ref_range["high"]
            unit = ref_range["unit"]
        
        # Interpret
        if value < low:
            interpretation = "Low"
            significance = f"Below reference range ({low}-{high} {unit})"
        elif value > high:
            interpretation = "High"
            significance = f"Above reference range ({low}-{high} {unit})"
        else:
            interpretation = "Normal"
            significance = f"Within reference range ({low}-{high} {unit})"
        
        return {
            "test": test_name.upper(),
            "value": value,
            "unit": unit,
            "reference_range": f"{low}-{high}",
            "interpretation": interpretation,
            "significance": significance
        }
    
    @staticmethod
    def interpret_panel(labs: Dict[str, float], gender: str = "male") -> List[Dict]:
        """Interpret multiple lab values"""
        results = []
        for test_name, value in labs.items():
            result = LabInterpreter.interpret_lab(test_name, value, gender)
            results.append(result)
        return results


class DrugInteractionChecker:
    """Check for drug interactions (simplified database)"""
    
    # Simplified interaction database
    INTERACTIONS = {
        ("warfarin", "aspirin"): {
            "severity": "Major",
            "description": "Increased bleeding risk"
        },
        ("warfarin", "nsaid"): {
            "severity": "Major",
            "description": "Increased bleeding risk"
        },
        ("ace_inhibitor", "potassium_supplement"): {
            "severity": "Moderate",
            "description": "Risk of hyperkalemia"
        },
        ("statin", "gemfibrozil"): {
            "severity": "Major",
            "description": "Increased risk of rhabdomyolysis"
        },
    }
    
    @staticmethod
    def check_interactions(medications: List[str]) -> List[Dict]:
        """Check for interactions between medications"""
        interactions = []
        
        # Normalize medication names
        meds = [m.lower().strip() for m in medications]
        
        # Check all pairs
        for i, med1 in enumerate(meds):
            for med2 in meds[i+1:]:
                # Check both orders
                interaction = (
                    DrugInteractionChecker.INTERACTIONS.get((med1, med2)) or
                    DrugInteractionChecker.INTERACTIONS.get((med2, med1))
                )
                
                if interaction:
                    interactions.append({
                        "drug1": med1,
                        "drug2": med2,
                        "severity": interaction["severity"],
                        "description": interaction["description"]
                    })
        
        return interactions


class ClinicalGuidelines:
    """Access to clinical practice guidelines (simplified)"""
    
    GUIDELINES = {
        "hypertension": {
            "source": "ACC/AHA 2017",
            "recommendations": [
                "BP target <130/80 mmHg for most adults",
                "Lifestyle modifications for all",
                "Medication if BP ≥140/90 or high CV risk"
            ]
        },
        "diabetes": {
            "source": "ADA 2023",
            "recommendations": [
                "HbA1c target <7% for most adults",
                "Individualize glycemic targets",
                "Metformin as first-line therapy"
            ]
        },
        "pneumonia": {
            "source": "IDSA/ATS",
            "recommendations": [
                "Assess severity using CURB-65 or PSI",
                "Empiric antibiotics based on risk factors",
                "Consider outpatient vs inpatient treatment"
            ]
        }
    }
    
    @staticmethod
    def get_guideline(condition: str) -> Optional[Dict]:
        """Get clinical guideline for a condition"""
        condition = condition.lower().strip()
        return ClinicalGuidelines.GUIDELINES.get(condition)
    
    @staticmethod
    def search_guidelines(query: str) -> List[Dict]:
        """Search guidelines by query"""
        query = query.lower()
        results = []
        
        for condition, guideline in ClinicalGuidelines.GUIDELINES.items():
            if query in condition or query in str(guideline).lower():
                results.append({
                    "condition": condition,
                    **guideline
                })
        
        return results
