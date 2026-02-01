"""
Offline RAG System for Medical Guidelines
Enables retrieval-augmented generation using local medical knowledge base
Optimized for rural/offline deployment
Now with automatic knowledge sync when internet available
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import pickle
import logging

logger = logging.getLogger(__name__)


class MedicalKnowledgeBase:
    """
    Local medical knowledge base for offline RAG
    Contains WHO guidelines, drug databases, clinical protocols
    
    Now supports:
    - Automatic updates when internet available
    - SQLite backend for scalability
    - Full-text search
    - Version control
    """
    
    def __init__(
        self,
        kb_path: str = "./data/medical_kb",
        use_database: bool = True,
        auto_sync: bool = True
    ):
        self.kb_path = Path(kb_path)
        self.use_database = use_database
        self.auto_sync = auto_sync
        
        # Initialize storage backend
        if use_database:
            from .knowledge_sync import KnowledgeDatabase
            self.db = KnowledgeDatabase(
                db_path=str(self.kb_path / "knowledge.db"),
                auto_sync=auto_sync
            )
            logger.info("Using SQLite database backend with auto-sync")
        else:
            self.db = None
            self.documents = []
            self.embeddings = None
            self.index = None
            # Load knowledge base from files
            self._load_kb()
            logger.info("Using file-based knowledge backend")
    
    def _load_kb(self):
        """Load medical knowledge base from disk"""
        # In production, these would be pre-computed embeddings
        self.guidelines = self._load_guidelines()
        self.drug_database = self._load_drug_database()
        self.protocols = self._load_protocols()
        
    def _load_guidelines(self) -> Dict:
        """Load WHO and clinical guidelines"""
        return {
            "who_essential_medicines": {
                "source": "WHO Essential Medicines List 2023",
                "categories": {
                    "antimalarials": [
                        {
                            "name": "Artemether-Lumefantrine",
                            "indication": "Malaria (uncomplicated)",
                            "dosing": "Based on weight, twice daily for 3 days",
                            "contraindications": ["First trimester pregnancy"],
                            "cost": "Low"
                        },
                        {
                            "name": "Artesunate",
                            "indication": "Severe malaria",
                            "dosing": "IV 2.4 mg/kg at 0, 12, 24h then daily",
                            "contraindications": ["Known hypersensitivity"],
                            "cost": "Moderate"
                        }
                    ],
                    "antibiotics": [
                        {
                            "name": "Amoxicillin",
                            "indication": "Respiratory infections, UTI",
                            "dosing": "500mg TID for 7 days",
                            "contraindications": ["Penicillin allergy"],
                            "cost": "Very low"
                        },
                        {
                            "name": "Doxycycline",
                            "indication": "Respiratory infections, rickettsia",
                            "dosing": "100mg BID",
                            "contraindications": ["Pregnancy", "Children <8yo"],
                            "cost": "Low"
                        }
                    ],
                    "antihypertensives": [
                        {
                            "name": "Amlodipine",
                            "indication": "Hypertension",
                            "dosing": "5-10mg once daily",
                            "contraindications": ["Severe hypotension"],
                            "cost": "Low"
                        },
                        {
                            "name": "Enalapril",
                            "indication": "Hypertension, heart failure",
                            "dosing": "5-20mg once or twice daily",
                            "contraindications": ["Pregnancy", "Bilateral renal artery stenosis"],
                            "cost": "Low"
                        }
                    ]
                }
            },
            
            "imci_protocols": {
                "source": "WHO IMCI (Integrated Management of Childhood Illness)",
                "age_groups": {
                    "2mo_to_5yr": {
                        "danger_signs": [
                            "Unable to drink or breastfeed",
                            "Vomits everything",
                            "Convulsions",
                            "Lethargic or unconscious"
                        ],
                        "fever_assessment": {
                            "malaria_risk": {
                                "high": "Test all fevers with RDT or microscopy",
                                "low": "Consider other causes first"
                            },
                            "measles_check": "Rash + one of: cough/runny nose/red eyes",
                            "meningitis_signs": "Stiff neck, bulging fontanelle"
                        }
                    }
                }
            },
            
            "emergency_protocols": {
                "source": "WHO Basic Emergency Care",
                "conditions": {
                    "shock": {
                        "recognition": ["Cold extremities", "Capillary refill >3s", "Weak pulse"],
                        "immediate_action": [
                            "Lay flat, elevate legs",
                            "Give oxygen if available",
                            "IV fluid bolus 20ml/kg rapidly",
                            "Identify and treat cause"
                        ],
                        "referral": "URGENT - Life threatening"
                    },
                    "severe_dehydration": {
                        "recognition": ["Sunken eyes", "Skin pinch goes back slowly", "Lethargic"],
                        "immediate_action": [
                            "IV fluid: Ringer's lactate or normal saline",
                            "100ml/kg over 6 hours (30ml/kg in first hour)",
                            "Monitor closely"
                        ],
                        "referral": "If no improvement or unable to give IV"
                    },
                    "respiratory_distress": {
                        "recognition": ["Respiratory rate >60/min (infant)", "Chest indrawing", "Stridor"],
                        "immediate_action": [
                            "Sit upright",
                            "Give oxygen",
                            "Assess for foreign body",
                            "Consider bronchodilator if wheeze"
                        ],
                        "referral": "URGENT if severe"
                    }
                }
            }
        }
    
    def _load_drug_database(self) -> Dict:
        """Load comprehensive drug information"""
        return {
            "interactions": {
                "warfarin": {
                    "major_interactions": [
                        "NSAIDs (bleeding risk)",
                        "Antibiotics (INR changes)",
                        "Antifungals (INR increase)"
                    ],
                    "monitoring": "Check INR regularly"
                },
                "metformin": {
                    "major_interactions": [
                        "Contrast dye (lactic acidosis risk)",
                        "Alcohol (lactic acidosis)"
                    ],
                    "contraindications": ["eGFR <30", "Acute illness"]
                }
            },
            "pregnancy_categories": {
                "safe": ["Amoxicillin", "Cephalexin", "Methyldopa", "Insulin"],
                "avoid": ["ACE inhibitors", "Warfarin", "Tetracyclines", "NSAIDs (3rd trimester)"],
                "contraindicated": ["Isotretinoin", "Misoprostol", "Thalidomide"]
            }
        }
    
    def _load_protocols(self) -> Dict:
        """Load clinical decision protocols"""
        return {
            "triage": {
                "emergency": [
                    "Airway obstruction",
                    "Severe respiratory distress",
                    "Shock",
                    "Unconscious",
                    "Severe bleeding",
                    "Severe burns >20% body"
                ],
                "urgent": [
                    "Moderate respiratory distress",
                    "Dehydration with ongoing losses",
                    "High fever with altered mental status",
                    "Severe pain",
                    "Major fracture"
                ],
                "non_urgent": [
                    "Minor illness >3 days",
                    "Routine follow-up",
                    "Medication refill",
                    "Health education"
                ]
            },
            
            "referral_criteria": {
                "maternal_health": [
                    "Vaginal bleeding in pregnancy",
                    "Severe headache + blurred vision (pre-eclampsia)",
                    "Fever in pregnancy >38.5C",
                    "Decreased fetal movement",
                    "Labor <37 weeks"
                ],
                "pediatric": [
                    "Any danger sign",
                    "Persistent vomiting >24h",
                    "Not eating/drinking",
                    "Fever >7 days",
                    "Blood in stool"
                ],
                "adult": [
                    "Chest pain",
                    "Stroke symptoms",
                    "Severe headache (worst ever)",
                    "Abdominal pain + guarding",
                    "Uncontrolled bleeding"
                ]
            }
        }
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search knowledge base for relevant information
        
        Args:
            query: Search query
            category: Optional category filter (guidelines, drugs, protocols)
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with metadata
        """
        # Use database if available
        if self.db:
            try:
                tables = None
                if category:
                    tables = [category if category.endswith('s') else category + 's']
                
                results = self.db.search(query, tables=tables, limit=top_k)
                return self._format_db_results(results)
            except Exception as e:
                logger.error(f"Database search failed: {e}, falling back to file search")
        
        # Fallback to file-based search
        results = []
        
        # Simple keyword search (in production, use embeddings + vector search)
        query_lower = query.lower()
        
        # Search guidelines
        if category in [None, "guidelines"]:
            results.extend(self._search_guidelines(query_lower))
        
        # Search drug database
        if category in [None, "drugs"]:
            results.extend(self._search_drugs(query_lower))
        
        # Search protocols
        if category in [None, "protocols"]:
            results.extend(self._search_protocols(query_lower))
        
        # Rank by relevance (simple keyword matching for demo)
        ranked = self._rank_results(results, query_lower)
        
        return ranked[:top_k]
    
    def _search_guidelines(self, query: str) -> List[Dict]:
        """Search clinical guidelines"""
        results = []
        
        # Check antimalarials
        if any(kw in query for kw in ["malaria", "fever", "chills"]):
            for drug in self.guidelines["who_essential_medicines"]["categories"]["antimalarials"]:
                results.append({
                    "type": "guideline",
                    "category": "antimalarial",
                    "content": drug,
                    "source": "WHO Essential Medicines List"
                })
        
        # Check antibiotics
        if any(kw in query for kw in ["infection", "bacteria", "antibiotic"]):
            for drug in self.guidelines["who_essential_medicines"]["categories"]["antibiotics"]:
                results.append({
                    "type": "guideline",
                    "category": "antibiotic",
                    "content": drug,
                    "source": "WHO Essential Medicines List"
                })
        
        # Check emergency protocols
        if any(kw in query for kw in ["emergency", "urgent", "shock", "severe"]):
            for condition, info in self.guidelines["emergency_protocols"]["conditions"].items():
                if any(kw in query for kw in condition.split("_")):
                    results.append({
                        "type": "guideline",
                        "category": "emergency",
                        "condition": condition,
                        "content": info,
                        "source": "WHO Basic Emergency Care"
                    })
        
        return results
    
    def _search_drugs(self, query: str) -> List[Dict]:
        """Search drug database"""
        results = []
        
        # Check drug interactions
        for drug, info in self.drug_database["interactions"].items():
            if drug in query:
                results.append({
                    "type": "drug_interaction",
                    "drug": drug,
                    "content": info,
                    "source": "Drug Interaction Database"
                })
        
        # Check pregnancy safety
        if "pregnan" in query:
            results.append({
                "type": "pregnancy_safety",
                "content": self.drug_database["pregnancy_categories"],
                "source": "FDA Pregnancy Categories"
            })
        
        return results
    
    def _search_protocols(self, query: str) -> List[Dict]:
        """Search clinical protocols"""
        results = []
        
        # Triage protocols
        if any(kw in query for kw in ["triage", "urgent", "emergency"]):
            results.append({
                "type": "protocol",
                "category": "triage",
                "content": self.protocols["triage"],
                "source": "Clinical Triage Protocol"
            })
        
        # Referral criteria
        if "refer" in query or "hospital" in query:
            results.append({
                "type": "protocol",
                "category": "referral",
                "content": self.protocols["referral_criteria"],
                "source": "Referral Guidelines"
            })
        
        return results
    
    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Rank results by relevance (simple keyword scoring)"""
        scored = []
        query_words = set(query.split())
        
        for result in results:
            # Simple scoring: count matching keywords
            content_str = json.dumps(result).lower()
            score = sum(1 for word in query_words if word in content_str)
            scored.append((score, result))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [result for score, result in scored]
    
    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get detailed information about a specific drug"""
        drug_lower = drug_name.lower()
        
        # Search all categories
        for category, drugs in self.guidelines["who_essential_medicines"]["categories"].items():
            for drug in drugs:
                if drug_lower in drug["name"].lower():
                    return {
                        "category": category,
                        "info": drug,
                        "source": "WHO Essential Medicines List"
                    }
        
        return None
    
    def check_contraindication(
        self,
        drug: str,
        patient_conditions: List[str]
    ) -> Dict:
        """
        Check if drug is contraindicated for patient
        
        Returns:
            {
                "contraindicated": bool,
                "reasons": List[str],
                "alternatives": List[str]
            }
        """
        drug_info = self.get_drug_info(drug)
        
        if not drug_info:
            return {
                "contraindicated": False,
                "reasons": [],
                "alternatives": [],
                "warning": "Drug not in database"
            }
        
        contraindications = drug_info["info"].get("contraindications", [])
        
        # Check for matches
        reasons = []
        for condition in patient_conditions:
            for contra in contraindications:
                if condition.lower() in contra.lower():
                    reasons.append(f"{contra} (patient has {condition})")
        
        return {
            "contraindicated": len(reasons) > 0,
            "reasons": reasons,
            "alternatives": self._suggest_alternatives(drug_info["category"], contraindications)
        }
    
    def _suggest_alternatives(
        self,
        category: str,
        avoid_contras: List[str]
    ) -> List[str]:
        """Suggest alternative drugs in same category"""
        alternatives = []
        
        if category in self.guidelines["who_essential_medicines"]["categories"]:
            for drug in self.guidelines["who_essential_medicines"]["categories"][category]:
                # Check if alternative doesn't have same contraindications
                drug_contras = drug.get("contraindications", [])
                if not any(c in drug_contras for c in avoid_contras):
                    alternatives.append(drug["name"])
        
        return alternatives


class OfflineRAG:
    """
    Retrieval-Augmented Generation for medical queries
    Uses local knowledge base (no internet required)
    """
    
    def __init__(self, kb_path: str = "./data/medical_kb"):
        self.kb = MedicalKnowledgeBase(kb_path)
    
    def augment_query(
        self,
        query: str,
        patient_context: Optional[Dict] = None,
        max_context_length: int = 1000
    ) -> Tuple[str, List[Dict]]:
        """
        Augment query with relevant information from knowledge base
        
        Args:
            query: Original query
            patient_context: Patient information
            max_context_length: Maximum length of context to add
        
        Returns:
            (augmented_prompt, retrieved_documents)
        """
        # Retrieve relevant documents
        retrieved = self.kb.search(query, top_k=3)
        
        # Check drug interactions if medications mentioned
        if patient_context and "medications" in patient_context:
            for med in patient_context["medications"]:
                drug_info = self.kb.get_drug_info(med)
                if drug_info:
                    retrieved.append({
                        "type": "drug_info",
                        "content": drug_info
                    })
        
        # Check contraindications
        if patient_context and "medications" in patient_context and "conditions" in patient_context:
            for med in patient_context["medications"]:
                contra_check = self.kb.check_contraindication(
                    med,
                    patient_context["conditions"]
                )
                if contra_check["contraindicated"]:
                    retrieved.append({
                        "type": "contraindication_warning",
                        "content": contra_check
                    })
        
        # Build augmented prompt
        context_parts = []
        for doc in retrieved[:3]:  # Limit to top 3 to avoid context overflow
            context_parts.append(self._format_context(doc))
        
        context_str = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "..."
        
        augmented_prompt = f"""Relevant Medical Knowledge:
{context_str}

Patient Query: {query}

Based on the medical knowledge above and best practices, provide your assessment:"""
        
        return augmented_prompt, retrieved
    
    def _format_context(self, doc: Dict) -> str:
        """Format retrieved document for context"""
        doc_type = doc.get("type", "unknown")
        
        if doc_type == "guideline":
            content = doc["content"]
            if isinstance(content, dict) and "name" in content:
                return f"""Guideline ({doc['source']}):
- {content['name']}: {content.get('indication', 'N/A')}
- Dosing: {content.get('dosing', 'See guidelines')}
- Contraindications: {', '.join(content.get('contraindications', []))}"""
            else:
                return f"Guideline: {json.dumps(content, indent=2)}"
        
        elif doc_type == "drug_interaction":
            content = doc["content"]
            return f"""Drug Interaction Alert - {doc['drug']}:
Major interactions: {', '.join(content.get('major_interactions', []))}
Monitoring: {content.get('monitoring', 'Standard')}"""
        
        elif doc_type == "protocol":
            return f"Protocol ({doc['category']}): {json.dumps(doc['content'], indent=2)}"
        
        else:
            return json.dumps(doc, indent=2)
    
    def _format_db_results(self, results: List[Dict]) -> List[Dict]:
        """Format database results to match expected structure"""
        formatted = []
        
        for result in results:
            doc_type = result.get("type")
            
            if doc_type == "guideline":
                formatted.append({
                    "type": "guideline",
                    "category": result.get("category"),
                    "content": json.loads(result.get("content", "{}")),
                    "source": result.get("source")
                })
            elif doc_type == "drug":
                formatted.append({
                    "type": "drug",
                    "name": result.get("name"),
                    "content": {
                        "name": result.get("name"),
                        "indication": result.get("indication"),
                        "dosing": result.get("dosing"),
                        "contraindications": json.loads(result.get("contraindications", "[]")),
                        "interactions": json.loads(result.get("interactions", "[]")),
                        "pregnancy_category": result.get("pregnancy_category")
                    },
                    "source": "Drug Database"
                })
            elif doc_type == "protocol":
                formatted.append({
                    "type": "protocol",
                    "category": result.get("category"),
                    "content": json.loads(result.get("content", "{}")),
                    "source": "Clinical Protocol"
                })
        
        return formatted
    
    def get_sync_status(self) -> Optional[Dict]:
        """Get knowledge base sync status"""
        if self.db:
            return self.db.get_stats()
        return None
    
    def force_sync(self) -> Dict:
        """Manually trigger knowledge base sync"""
        if self.db:
            return self.db.sync_from_remote(force=True)
        return {"status": "error", "message": "Database backend not enabled"}


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Offline RAG System Demo")
    print("="*60)
    
    rag = OfflineRAG()
    
    # Test query
    query = "Patient with fever, chills, and body aches for 3 days in malaria-endemic area"
    
    print(f"\nQuery: {query}\n")
    
    augmented, docs = rag.augment_query(query)
    
    print("Retrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Type: {doc['type']}")
        print(f"   Content: {json.dumps(doc, indent=2)[:200]}...")
    
    print(f"\n\nAugmented prompt length: {len(augmented)} chars")
    print(f"\nFirst 500 chars of augmented prompt:")
    print(augmented[:500] + "...")
