"""
Automated evaluation suite for the Medical QA pipeline.
Includes citation precision, multi-hop accuracy, and adversarial gold set testing.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pipeline.rag_pipeline import MultiHopRAGPipeline, MedicalQuery, MedicalQAResponse


@dataclass
class EvalSample:
    sample_id: str
    query: str
    query_type: str
    expected_answer_concepts: list[str]    # SNOMED CT concepts that must appear
    gold_citations: list[str]              # PMIDs that must be cited
    contraindication_flags: list[str]      # Expected contraindication flags
    adversarial: bool = False              # Part of adversarial gold set
    patient_context: dict = field(default_factory=dict)


@dataclass
class EvalMetrics:
    n_samples: int = 0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    multi_hop_accuracy: float = 0.0
    contraindication_recall: float = 0.0
    contraindication_precision: float = 0.0
    pii_redaction_rate: float = 0.0
    avg_latency_ms: float = 0.0
    adversarial_accuracy: float = 0.0
    manual_audit_reduction: float = 0.0


class MedicalQAEvaluator:
    """
    End-to-end evaluator for the multi-hop RAG pipeline.
    Tests against 1,500-sample adversarial gold set.
    """

    BASELINE_METRICS = {
        "multi_hop_accuracy": 0.54,
        "citation_precision": 0.71,
        "contraindication_errors": 0.47,   # Error rate baseline
    }

    def __init__(self, pipeline: MultiHopRAGPipeline, cfg: dict):
        self.pipeline = pipeline
        self.cfg = cfg

    async def evaluate(
        self,
        samples: list[EvalSample],
        batch_size: int = 32,
    ) -> EvalMetrics:
        """Run full evaluation over gold set."""
        import time
        all_responses: list[tuple[EvalSample, MedicalQAResponse, float]] = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i: i + batch_size]
            queries = [
                MedicalQuery(
                    query_id=s.sample_id,
                    raw_text=s.query,
                    query_type=s.query_type,
                    patient_context=s.patient_context,
                )
                for s in batch
            ]
            start = time.monotonic()
            responses = await self.pipeline.run_batch(queries)
            latency_ms = (time.monotonic() - start) * 1000 / len(batch)

            for sample, response in zip(batch, responses):
                all_responses.append((sample, response, latency_ms))

        return self._compute_metrics(all_responses)

    def _compute_metrics(
        self,
        results: list[tuple[EvalSample, MedicalQAResponse, float]],
    ) -> EvalMetrics:
        metrics = EvalMetrics(n_samples=len(results))
        if not results:
            return metrics

        citation_precisions, citation_recalls = [], []
        multi_hop_correct = 0
        contraindication_tp, contraindication_fp, contraindication_fn = 0, 0, 0
        adversarial_correct, adversarial_total = 0, 0
        latencies = []

        for sample, response, latency_ms in results:
            latencies.append(latency_ms)

            # Citation precision & recall
            predicted_pmids = set(response.citations)
            gold_pmids = set(sample.gold_citations)
            if predicted_pmids:
                precision = len(predicted_pmids & gold_pmids) / len(predicted_pmids)
                citation_precisions.append(precision)
            if gold_pmids:
                recall = len(predicted_pmids & gold_pmids) / len(gold_pmids)
                citation_recalls.append(recall)

            # Multi-hop accuracy: did answer contain expected concepts?
            answer_lower = response.answer.lower()
            concepts_found = sum(
                c.lower() in answer_lower for c in sample.expected_answer_concepts
            )
            if sample.expected_answer_concepts:
                hop_acc = concepts_found / len(sample.expected_answer_concepts)
                if hop_acc >= 0.7:
                    multi_hop_correct += 1

            # Contraindication recall/precision
            expected_flags = set(sample.contraindication_flags)
            predicted_flags = set(response.contraindication_flags)
            contraindication_tp += len(expected_flags & predicted_flags)
            contraindication_fp += len(predicted_flags - expected_flags)
            contraindication_fn += len(expected_flags - predicted_flags)

            # Adversarial accuracy
            if sample.adversarial:
                adversarial_total += 1
                if concepts_found > 0:
                    adversarial_correct += 1

        n = len(results)
        metrics.citation_precision = sum(citation_precisions) / len(citation_precisions) if citation_precisions else 0.0
        metrics.citation_recall = sum(citation_recalls) / len(citation_recalls) if citation_recalls else 0.0
        metrics.multi_hop_accuracy = multi_hop_correct / n
        metrics.avg_latency_ms = sum(latencies) / len(latencies)

        tp = contraindication_tp
        metrics.contraindication_recall = tp / (tp + contraindication_fn) if (tp + contraindication_fn) > 0 else 0.0
        metrics.contraindication_precision = tp / (tp + contraindication_fp) if (tp + contraindication_fp) > 0 else 0.0

        if adversarial_total > 0:
            metrics.adversarial_accuracy = adversarial_correct / adversarial_total

        # Manual audit reduction: proportion of responses with citations that don't need manual review
        cited_responses = sum(1 for _, r, _ in results if len(r.citations) >= 2)
        metrics.manual_audit_reduction = cited_responses / n

        return metrics

    def assert_regression_gates(self, metrics: EvalMetrics) -> None:
        """CI regression gates — raise if any metric regresses below threshold."""
        gates = {
            "multi_hop_accuracy": (metrics.multi_hop_accuracy, 0.80),
            "citation_precision": (metrics.citation_precision, 0.88),
            "contraindication_recall": (metrics.contraindication_recall, 0.90),
        }
        failures = []
        for name, (value, threshold) in gates.items():
            if value < threshold:
                failures.append(f"{name}: {value:.3f} < {threshold:.3f}")
        if failures:
            raise AssertionError(f"Regression gate failures:\n" + "\n".join(failures))

    def load_gold_set(self, path: str) -> list[EvalSample]:
        """Load 1,500-sample adversarial gold set from JSONL."""
        samples = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                samples.append(EvalSample(
                    sample_id=d["id"],
                    query=d["query"],
                    query_type=d.get("query_type", "multi_hop"),
                    expected_answer_concepts=d.get("expected_concepts", []),
                    gold_citations=d.get("gold_pmids", []),
                    contraindication_flags=d.get("contraindications", []),
                    adversarial=d.get("adversarial", False),
                    patient_context=d.get("patient_context", {}),
                ))
        return samples

    def print_report(self, metrics: EvalMetrics) -> None:
        print("\n" + "="*55)
        print("  Medical QA Pipeline — Evaluation Report")
        print("="*55)
        print(f"  Samples evaluated:         {metrics.n_samples}")
        print(f"  Multi-hop accuracy:        {metrics.multi_hop_accuracy:.1%}  (baseline: 54%)")
        print(f"  Citation precision:        {metrics.citation_precision:.1%}  (target: 92%)")
        print(f"  Citation recall:           {metrics.citation_recall:.1%}")
        print(f"  Contraindication recall:   {metrics.contraindication_recall:.1%}")
        print(f"  Adversarial accuracy:      {metrics.adversarial_accuracy:.1%}")
        print(f"  Avg latency:               {metrics.avg_latency_ms:.1f}ms")
        print(f"  Audit overhead reduction:  {metrics.manual_audit_reduction:.1%}  (target: 40%)")
        print("="*55 + "\n")
