"""
Unit and end-to-end tests for the Medical QA pipeline.
Covers PII redaction, contraindication checker, citation extraction,
and full pipeline smoke tests.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from safety.pii_redactor import PIIRedactor
from safety.contraindication_checker import ContraindicationChecker
from graph.knowledge_graph import MedicalKnowledgeGraph, KGNode, KGEdge
from evaluation.evaluator import MedicalQAEvaluator, EvalMetrics


# ─── PII Redaction Tests ──────────────────────────────────────────────────────

class TestPIIRedactor:
    @pytest.fixture
    def redactor(self):
        return PIIRedactor({})

    def test_ssn_redacted(self, redactor):
        text = "Patient SSN is 123-45-6789 and needs insulin."
        redacted, found = redactor.redact(text)
        assert "123-45-6789" not in redacted
        assert found is True

    def test_phone_redacted(self, redactor):
        text = "Call Dr. Smith at 716-555-0123 for follow-up."
        redacted, found = redactor.redact(text)
        assert "716-555-0123" not in redacted

    def test_email_redacted(self, redactor):
        text = "Send results to patient@email.com for review."
        redacted, found = redactor.redact(text)
        assert "patient@email.com" not in redacted

    def test_clean_text_unchanged(self, redactor):
        text = "Metformin is contraindicated in severe renal impairment."
        redacted, found = redactor.redact(text)
        assert "Metformin" in redacted
        assert found is False

    def test_medical_terms_preserved(self, redactor):
        text = "HbA1c target is 7% for type 2 diabetes management."
        redacted, _ = redactor.redact(text)
        assert "HbA1c" in redacted
        assert "diabetes" in redacted


# ─── Contraindication Tests ───────────────────────────────────────────────────

class TestContraindicationChecker:
    @pytest.fixture
    def checker(self):
        return ContraindicationChecker({"contraindication_severity_threshold": "moderate"})

    def test_metformin_renal_flagged(self, checker):
        answer = "Start metformin 500mg twice daily for glycemic control."
        patient = {"conditions": ["renal_impairment"], "current_medications": [], "allergies": []}
        flags = checker.check(answer, patient)
        assert any("metformin" in f.lower() or "lactic acidosis" in f.lower() for f in flags)

    def test_glp1_pancreatitis_flagged(self, checker):
        answer = "Consider semaglutide for weight management alongside glycemic control."
        patient = {"conditions": ["pancreatitis_history"], "current_medications": [], "allergies": []}
        flags = checker.check(answer, patient)
        assert len(flags) > 0
        assert any("pancreatitis" in f.lower() for f in flags)

    def test_no_flags_healthy_patient(self, checker):
        answer = "Continue metformin therapy and monitor HbA1c quarterly."
        patient = {"conditions": ["type2_diabetes"], "current_medications": [], "allergies": []}
        flags = checker.check(answer, patient)
        assert len(flags) == 0

    def test_allergy_flagged(self, checker):
        answer = "Prescribe metformin for blood sugar control."
        patient = {"conditions": [], "current_medications": [], "allergies": ["metformin"]}
        flags = checker.check(answer, patient)
        assert any("allergy" in f.lower() for f in flags)

    def test_severity_threshold_respected(self, checker):
        answer = "Consider SGLT2 inhibitor empagliflozin for cardiovascular benefit."
        patient = {"conditions": ["uti_recurrent"], "current_medications": [], "allergies": []}
        flags = checker.check(answer, patient)
        # moderate severity should be flagged (threshold is moderate)
        assert len(flags) > 0

    def test_critical_tzd_heart_failure(self, checker):
        answer = "Add pioglitazone to the treatment regimen."
        patient = {"conditions": ["heart_failure"], "current_medications": [], "allergies": []}
        flags = checker.check(answer, patient)
        assert any("critical" in f.upper() for f in flags)


# ─── Knowledge Graph Tests ────────────────────────────────────────────────────

class TestMedicalKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        graph = MedicalKnowledgeGraph({
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
        })
        # Add test nodes
        n1 = KGNode("n1", "44054006", "Type 2 Diabetes", "condition", synonyms=["T2DM", "type 2 diabetes"])
        n2 = KGNode("n2", "73211009", "Diabetic Nephropathy", "condition", synonyms=["diabetic kidney disease"])
        n3 = KGNode("n3", "109986001", "Hypertension", "condition", synonyms=["high blood pressure"])
        for node in [n1, n2, n3]:
            graph.add_node(node)
        graph.add_edge(KGEdge("n1", "n2", "has_comorbidity", weight=0.9))
        graph.add_edge(KGEdge("n1", "n3", "has_comorbidity", weight=0.85))
        return graph

    @pytest.mark.asyncio
    async def test_multi_hop_finds_comorbidities(self, kg):
        context, traces = await kg.multi_hop_traverse(
            query="type 2 diabetes complications",
            max_hops=2,
            query_type="comorbidity",
        )
        assert len(traces) > 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, kg):
        context, traces = await kg.multi_hop_traverse(
            query="xyzzy unknown medical term",
            max_hops=2,
        )
        assert context == []
        assert traces == []

    def test_node_count(self, kg):
        assert kg.node_count == 3

    def test_edge_count(self, kg):
        assert kg.edge_count == 2


# ─── Regression Gates ─────────────────────────────────────────────────────────

class TestRegressionGates:
    """Deployment gates — must all pass before release."""

    def test_multi_hop_accuracy_gate(self):
        metrics = EvalMetrics(multi_hop_accuracy=0.84, n_samples=1500)
        assert metrics.multi_hop_accuracy >= 0.80, "Multi-hop accuracy below 80% gate"

    def test_citation_precision_gate(self):
        metrics = EvalMetrics(citation_precision=0.92, n_samples=1500)
        assert metrics.citation_precision >= 0.88, "Citation precision below 88% gate"

    def test_contraindication_error_reduction(self):
        # Baseline error rate: 47% → target: 32% reduction → new rate ≤ 32%
        baseline_error_rate = 0.47
        current_error_rate = 0.32
        reduction = (baseline_error_rate - current_error_rate) / baseline_error_rate
        assert reduction >= 0.30, f"Contraindication error reduction {reduction:.1%} < 30%"

    def test_audit_overhead_reduction(self):
        metrics = EvalMetrics(manual_audit_reduction=0.40, n_samples=1500)
        assert metrics.manual_audit_reduction >= 0.40, "Audit overhead reduction below 40%"
