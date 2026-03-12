"""
DSPy-tuned HyDE (Hypothetical Document Embeddings) query rewriter.
Generates hypothetical answers to improve dense retrieval grounding.
"""
from __future__ import annotations

import asyncio
from typing import Any

import dspy
from pydantic import BaseModel


# ─── DSPy Signatures ──────────────────────────────────────────────────────────

class MedicalHyDE(dspy.Signature):
    """
    Generate a hypothetical medical document that would answer this query.
    The document should be written as if it were an excerpt from a clinical
    study or medical textbook, using precise medical terminology.
    """
    query: str = dspy.InputField(desc="Medical question to answer")
    domain: str = dspy.InputField(desc="Medical domain e.g. endocrinology")
    hypothetical_document: str = dspy.OutputField(
        desc="A hypothetical medical passage that answers the query"
    )


class QueryExpansion(dspy.Signature):
    """
    Expand a medical query into multiple semantically diverse sub-queries
    to improve multi-hop retrieval coverage.
    """
    query: str = dspy.InputField(desc="Original medical query")
    expanded_queries: list[str] = dspy.OutputField(
        desc="3-5 diverse sub-queries covering different aspects"
    )


class MedicalHyDEModule(dspy.Module):
    """DSPy module combining HyDE generation with query expansion."""

    def __init__(self):
        super().__init__()
        self.hyde_generator = dspy.ChainOfThought(MedicalHyDE)
        self.query_expander = dspy.ChainOfThought(QueryExpansion)

    def forward(self, query: str, domain: str = "endocrinology") -> dspy.Prediction:
        # Generate hypothetical document
        hyde_result = self.hyde_generator(query=query, domain=domain)

        # Expand original query into sub-queries
        expansion_result = self.query_expander(query=query)

        return dspy.Prediction(
            hypothetical_document=hyde_result.hypothetical_document,
            expanded_queries=expansion_result.expanded_queries,
        )


# ─── Rewriter ─────────────────────────────────────────────────────────────────

class HyDEConfig(BaseModel):
    lm_model: str = "openai/gpt-4o-mini"
    n_hypotheses: int = 3
    domain: str = "endocrinology"
    temperature: float = 0.7
    max_tokens: int = 512
    dspy_optimizer: str = "BootstrapFewShot"  # or MIPROv2


class HyDEQueryRewriter:
    """
    DSPy-optimized HyDE rewriter for medical queries.
    Generates hypothetical documents + expanded sub-queries for richer retrieval.
    """

    def __init__(self, cfg: dict):
        self.cfg = HyDEConfig(**cfg)
        self._module: MedicalHyDEModule | None = None
        self._setup_dspy()

    def _setup_dspy(self) -> None:
        try:
            lm = dspy.LM(
                self.cfg.lm_model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            dspy.configure(lm=lm)
            self._module = MedicalHyDEModule()
        except Exception as e:
            print(f"[HyDE] DSPy setup failed: {e}. Using fallback rewriter.")

    async def rewrite(self, query: str, n_hypotheses: int | None = None) -> list[str]:
        """
        Returns list of queries: [original, hyde_doc, ...expanded_queries].
        All are used as input to BGE-M3 retrieval with RRF fusion.
        """
        n = n_hypotheses or self.cfg.n_hypotheses

        if self._module is None:
            return self._fallback_rewrite(query, n)

        loop = asyncio.get_event_loop()
        try:
            prediction = await loop.run_in_executor(
                None, self._module, query, self.cfg.domain
            )
            queries = [query, prediction.hypothetical_document]
            if isinstance(prediction.expanded_queries, list):
                queries.extend(prediction.expanded_queries[:n])
            return queries[:n + 1]
        except Exception as e:
            print(f"[HyDE] Rewrite failed: {e}. Using fallback.")
            return self._fallback_rewrite(query, n)

    def tune(self, trainset: list[dict], metric: Any = None) -> None:
        """
        Tune the DSPy module using BootstrapFewShot or MIPROv2.
        trainset: list of {"query": ..., "expected_passages": [...]}
        """
        if self._module is None:
            return

        if self.cfg.dspy_optimizer == "MIPROv2":
            optimizer = dspy.MIPROv2(metric=metric or self._default_metric, auto="medium")
        else:
            optimizer = dspy.BootstrapFewShot(metric=metric or self._default_metric, max_bootstrapped_demos=4)

        dspy_trainset = [
            dspy.Example(query=ex["query"], domain=self.cfg.domain).with_inputs("query", "domain")
            for ex in trainset
        ]
        self._module = optimizer.compile(self._module, trainset=dspy_trainset)

    def _default_metric(self, example: Any, prediction: Any, trace: Any = None) -> float:
        """Default metric: check that hypothetical doc contains medical terms."""
        doc = prediction.hypothetical_document.lower()
        medical_terms = ["patient", "treatment", "diagnosis", "mg", "study", "clinical"]
        return sum(term in doc for term in medical_terms) / len(medical_terms)

    def _fallback_rewrite(self, query: str, n: int) -> list[str]:
        """Simple keyword-based fallback when DSPy is unavailable."""
        templates = [
            query,
            f"What is the clinical management of {query}?",
            f"Endocrinology guidelines for {query}",
            f"Comorbidities and contraindications related to {query}",
        ]
        return templates[:n + 1]
