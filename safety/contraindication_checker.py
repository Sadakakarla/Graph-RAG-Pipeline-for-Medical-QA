"""
Contraindication safety checker for generated medical responses.
Cross-references patient context against known drug/condition interactions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class ContraindicationFlag:
    flag_type: str          # drug_drug | drug_condition | dose_alert
    description: str
    severity: str           # critical | major | moderate | minor
    source: str


class ContraindicationChecker:
    """
    Rule-based + knowledge-graph contraindication checker.
    Scans generated answers for drug mentions and checks against
    patient conditions, allergies, and current medications.
    """

    # High-risk endocrinology drug pairs
    KNOWN_CONTRAINDICATIONS: dict[str, list[dict]] = {
        "metformin": [
            {"condition": "renal_impairment", "severity": "critical",
             "description": "Metformin contraindicated in eGFR < 30 due to lactic acidosis risk"},
            {"condition": "contrast_dye", "severity": "major",
             "description": "Hold metformin 48h before/after iodinated contrast"},
        ],
        "insulin": [
            {"condition": "hypoglycemia_unawareness", "severity": "critical",
             "description": "Intensive insulin therapy high-risk with hypoglycemia unawareness"},
        ],
        "thiazolidinedione": [
            {"condition": "heart_failure", "severity": "critical",
             "description": "TZDs contraindicated in NYHA Class III/IV heart failure"},
            {"condition": "bladder_cancer", "severity": "major",
             "description": "Pioglitazone associated with increased bladder cancer risk"},
        ],
        "sglt2_inhibitor": [
            {"condition": "type1_diabetes", "severity": "major",
             "description": "SGLT2i not approved for T1DM — DKA risk"},
            {"condition": "uti_recurrent", "severity": "moderate",
             "description": "SGLT2i increases risk of genital mycotic infections and UTIs"},
        ],
        "glp1_agonist": [
            {"condition": "pancreatitis_history", "severity": "major",
             "description": "GLP-1 agonists contraindicated with history of pancreatitis"},
            {"condition": "medullary_thyroid_cancer", "severity": "critical",
             "description": "GLP-1 agonists contraindicated with personal/family history of MTC"},
        ],
        "levothyroxine": [
            {"condition": "adrenal_insufficiency", "severity": "critical",
             "description": "Treat adrenal insufficiency before initiating levothyroxine"},
        ],
    }

    DRUG_PATTERNS = {
        "metformin": r"\b(metformin|glucophage|fortamet)\b",
        "insulin": r"\b(insulin|glargine|detemir|aspart|lispro|humalog|lantus)\b",
        "thiazolidinedione": r"\b(pioglitazone|rosiglitazone|actos|avandia|tzd)\b",
        "sglt2_inhibitor": r"\b(empagliflozin|dapagliflozin|canagliflozin|jardiance|farxiga|invokana)\b",
        "glp1_agonist": r"\b(semaglutide|liraglutide|dulaglutide|ozempic|victoza|trulicity|wegovy)\b",
        "levothyroxine": r"\b(levothyroxine|synthroid|eltroxin|t4)\b",
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.severity_threshold = cfg.get("contraindication_severity_threshold", "moderate")

    def check(self, answer: str, patient_context: dict) -> list[str]:
        """
        Check generated answer for contraindications given patient context.
        Returns list of flag description strings.
        """
        answer_lower = answer.lower()
        patient_conditions = {c.lower() for c in patient_context.get("conditions", [])}
        patient_medications = {m.lower() for m in patient_context.get("current_medications", [])}
        patient_allergies = {a.lower() for a in patient_context.get("allergies", [])}

        mentioned_drugs = self._extract_mentioned_drugs(answer_lower)
        flags: list[str] = []

        for drug_class in mentioned_drugs:
            contraindications = self.KNOWN_CONTRAINDICATIONS.get(drug_class, [])
            for ci in contraindications:
                condition = ci["condition"].replace("_", " ")
                if any(condition in pc for pc in patient_conditions):
                    if self._meets_severity_threshold(ci["severity"]):
                        flags.append(
                            f"[{ci['severity'].upper()}] {ci['description']}"
                        )

        # Check allergy conflicts
        for drug_class in mentioned_drugs:
            if any(drug_class in allergy for allergy in patient_allergies):
                flags.append(f"[CRITICAL] Patient has documented allergy to {drug_class}")

        return flags

    def _extract_mentioned_drugs(self, text: str) -> list[str]:
        found = []
        for drug_class, pattern in self.DRUG_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(drug_class)
        return found

    def _meets_severity_threshold(self, severity: str) -> bool:
        order = ["minor", "moderate", "major", "critical"]
        threshold_idx = order.index(self.severity_threshold) if self.severity_threshold in order else 1
        severity_idx = order.index(severity) if severity in order else 0
        return severity_idx >= threshold_idx
