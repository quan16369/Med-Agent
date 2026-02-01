"""
Unit tests for MedAssist system
"""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.tools import (
    MedicalCalculators,
    LabInterpreter,
    DrugInteractionChecker,
    ClinicalGuidelines
)


class TestMedicalCalculators(unittest.TestCase):
    """Test medical calculator functions"""
    
    def test_bmi_normal(self):
        """Test BMI calculation for normal weight"""
        result = MedicalCalculators.bmi(70, 1.75)
        self.assertEqual(result['bmi'], 22.9)
        self.assertEqual(result['category'], "Normal weight")
    
    def test_bmi_overweight(self):
        """Test BMI calculation for overweight"""
        result = MedicalCalculators.bmi(90, 1.75)
        self.assertGreater(result['bmi'], 25)
        self.assertEqual(result['category'], "Overweight")
    
    def test_egfr_normal(self):
        """Test eGFR calculation for normal kidney function"""
        result = MedicalCalculators.egfr(
            creatinine_mg_dl=1.0,
            age=45,
            is_female=False
        )
        self.assertGreater(result['egfr'], 90)
        self.assertEqual(result['ckd_stage'], "G1")
    
    def test_egfr_female(self):
        """Test eGFR calculation accounts for gender"""
        result_male = MedicalCalculators.egfr(1.0, 45, False)
        result_female = MedicalCalculators.egfr(1.0, 45, True)
        # Female typically has slightly lower eGFR for same creatinine
        self.assertNotEqual(result_male['egfr'], result_female['egfr'])
    
    def test_chads2_vasc_low_risk(self):
        """Test CHA2DS2-VASc score for low risk"""
        result = MedicalCalculators.chads2_vasc(
            age=55,
            is_female=False,
            has_chf=False,
            has_hypertension=False,
            has_diabetes=False,
            has_stroke_tia=False
        )
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['risk_category'], "Low")
    
    def test_chads2_vasc_high_risk(self):
        """Test CHA2DS2-VASc score for high risk"""
        result = MedicalCalculators.chads2_vasc(
            age=78,
            is_female=True,
            has_chf=True,
            has_hypertension=True,
            has_diabetes=True,
            has_stroke_tia=True
        )
        self.assertGreater(result['score'], 5)
        self.assertEqual(result['risk_category'], "Moderate-High")
    
    def test_wells_dvt(self):
        """Test Wells DVT score"""
        result = MedicalCalculators.wells_dvt(
            active_cancer=True,
            bedridden_or_surgery=True,
            entire_leg_swollen=True
        )
        self.assertGreater(result['score'], 2)
        self.assertEqual(result['probability'], "High")


class TestLabInterpreter(unittest.TestCase):
    """Test lab value interpretation"""
    
    def test_interpret_normal_wbc(self):
        """Test interpretation of normal WBC"""
        result = LabInterpreter.interpret_lab("wbc", 8.0)
        self.assertEqual(result['interpretation'], "Normal")
    
    def test_interpret_low_hemoglobin(self):
        """Test interpretation of low hemoglobin"""
        result = LabInterpreter.interpret_lab("hemoglobin", 10.0, "male")
        self.assertEqual(result['interpretation'], "Low")
    
    def test_interpret_high_glucose(self):
        """Test interpretation of high glucose"""
        result = LabInterpreter.interpret_lab("glucose", 150)
        self.assertEqual(result['interpretation'], "High")
    
    def test_gender_specific_ranges(self):
        """Test gender-specific reference ranges"""
        hgb_male = LabInterpreter.interpret_lab("hemoglobin", 13.0, "male")
        hgb_female = LabInterpreter.interpret_lab("hemoglobin", 13.0, "female")
        # 13.0 is low-normal for males, normal for females
        self.assertIn(hgb_male['interpretation'], ["Low", "Normal"])
        self.assertEqual(hgb_female['interpretation'], "Normal")
    
    def test_interpret_panel(self):
        """Test interpretation of multiple labs"""
        labs = {
            "wbc": 10.0,
            "hemoglobin": 14.0,
            "platelets": 250
        }
        results = LabInterpreter.interpret_panel(labs, "male")
        self.assertEqual(len(results), 3)
        self.assertTrue(all('interpretation' in r for r in results))


class TestDrugInteractionChecker(unittest.TestCase):
    """Test drug interaction checking"""
    
    def test_no_interactions(self):
        """Test when no interactions present"""
        meds = ["metformin", "lisinopril"]
        interactions = DrugInteractionChecker.check_interactions(meds)
        # These don't have interactions in our simplified database
        self.assertEqual(len(interactions), 0)
    
    def test_major_interaction(self):
        """Test detection of major interaction"""
        meds = ["warfarin", "aspirin"]
        interactions = DrugInteractionChecker.check_interactions(meds)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]['severity'], "Major")
    
    def test_case_insensitive(self):
        """Test that checking is case-insensitive"""
        meds1 = ["Warfarin", "Aspirin"]
        meds2 = ["warfarin", "aspirin"]
        interactions1 = DrugInteractionChecker.check_interactions(meds1)
        interactions2 = DrugInteractionChecker.check_interactions(meds2)
        self.assertEqual(len(interactions1), len(interactions2))


class TestClinicalGuidelines(unittest.TestCase):
    """Test clinical guideline access"""
    
    def test_get_existing_guideline(self):
        """Test retrieval of existing guideline"""
        guideline = ClinicalGuidelines.get_guideline("hypertension")
        self.assertIsNotNone(guideline)
        self.assertIn('source', guideline)
        self.assertIn('recommendations', guideline)
    
    def test_get_nonexistent_guideline(self):
        """Test retrieval of non-existent guideline"""
        guideline = ClinicalGuidelines.get_guideline("nonexistent_condition")
        self.assertIsNone(guideline)
    
    def test_search_guidelines(self):
        """Test guideline search"""
        results = ClinicalGuidelines.search_guidelines("diabetes")
        self.assertGreater(len(results), 0)
        self.assertTrue(any('diabetes' in r['condition'] for r in results))


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_clinical_workflow(self):
        """Test a complete clinical calculation workflow"""
        # Patient: 65yo male, creatinine 1.5, weight 85kg, height 1.75m
        
        # Calculate BMI
        bmi = MedicalCalculators.bmi(85, 1.75)
        self.assertEqual(bmi['category'], "Overweight")
        
        # Calculate eGFR
        egfr = MedicalCalculators.egfr(1.5, 65, False)
        self.assertLess(egfr['egfr'], 90)
        
        # Check labs
        labs = {"creatinine": 1.5, "potassium": 4.8}
        lab_results = LabInterpreter.interpret_panel(labs, "male")
        self.assertEqual(len(lab_results), 2)
        
        # Check medications
        meds = ["lisinopril", "potassium_supplement"]
        interactions = DrugInteractionChecker.check_interactions(meds)
        # Should find ACE inhibitor + potassium interaction
        self.assertGreater(len(interactions), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
