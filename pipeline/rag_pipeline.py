"""
Multi-Hop RAG Pipeline for Medical QA.
Orchestrates GraphRAG + dense retrieval + DSPy HyDE + Llama-3 generation.
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graph.knowledge_graph import MedicalKnowledgeGraph
from retrieval.bge_retriever import BGERetriever
from retrieval.hyde_rewriter import HyDEQueryRewriter
from generation.llama_generator import LlamaGenerator
from safety.pii_redactor import PIIRedactor
from safety.contraindication_checker import ContraindicationChecker


# ─── Data Models ──────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMORBIDITY = "comorbidity"
    CONTRAINDICATION = "contraindication"


class MedicalQuery(BaseModel):
    query_id: str
    raw_text: str
    query_type: QueryType = QueryType.MULTI_HOP
    patient_context: dict = Field(default_factory=dict)
    max_hops: int = 3


class RetrievedContext(BaseModel):
    graph_nodes: list[dict] = Field(default_factory=list)
    dense_passages: list[dict] = Field(default_factory=list)
    hop_traces: list[dict] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class MedicalQAResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[str]
    confidence: float
    contraindication_flags: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    pii_redacted: bool = False
    hop_count: int = 0


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class MultiHopRAGPipeline:
    """
    End-to-end multi-hop RAG pipeline for medical QA.

    Flow:
      query → PII redaction → HyDE rewriting → GraphRAG traversal
           → BGE dense retrieval → context fusion → Llama-3 generation
           → contraindication check → citation grounding → response
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kg = MedicalKnowledgeGraph(cfg["knowledge_graph"])
        self.retriever = BGERetriever(cfg["retrieval"])
        self.rewriter = HyDEQueryRewriter(cfg["hyde"])
        self.generator = LlamaGenerator(cfg["generation"])
        self.pii_redactor = PIIRedactor(cfg["safety"])
        self.contraindication_checker = ContraindicationChecker(cfg["safety"])

    async def run(self, query: MedicalQuery) -> MedicalQAResponse:
        # Step 1: PII redaction
        clean_query, pii_found = self.pii_redactor.redact(query.raw_text)

        # Step 2: HyDE query rewriting for better retrieval grounding
        rewritten_queries = await self.rewriter.rewrite(clean_query, n_hypotheses=3)

        # Step 3: Multi-hop graph traversal over SNOMED CT
        graph_context, hop_traces = await self.kg.multi_hop_traverse(
            query=clean_query,
            max_hops=query.max_hops,
            query_type=query.query_type,
        )

        # Step 4: BGE-M3 dense retrieval over NIH corpora
        dense_passages = await self.retriever.retrieve(
            queries=rewritten_queries,
            top_k=self.cfg["retrieval"].get("top_k", 10),
        )

        # Step 5: Context fusion — merge graph + dense
        fused_context = self._fuse_contexts(graph_context, dense_passages)

        # Step 6: Llama-3 generation with citations
        raw_answer, citations, reasoning = await self.generator.generate(
            query=clean_query,
            context=fused_context,
            hop_traces=hop_traces,
        )

        # Step 7: Contraindication safety check
        flags = self.contraindication_checker.check(
            answer=raw_answer,
            patient_context=query.patient_context,
        )

        return MedicalQAResponse(
            query_id=query.query_id,
            answer=raw_answer,
            citations=citations,
            confidence=self._compute_confidence(dense_passages, graph_context),
            contraindication_flags=flags,
            reasoning_trace=reasoning,
            pii_redacted=pii_found,
            hop_count=len(hop_traces),
        )

    def _fuse_contexts(
        self,
        graph_context: list[dict],
        dense_passages: list[dict],
    ) -> list[dict]:
        """Merge and deduplicate graph nodes and dense passages by relevance score."""
        combined = []
        seen_ids: set[str] = set()

        for item in graph_context + dense_passages:
            item_id = item.get("id") or item.get("pmid") or item.get("node_id", "")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                combined.append(item)

        return sorted(combined, key=lambda x: x.get("score", 0.0), reverse=True)[
            : self.cfg.get("max_context_items", 20)
        ]

    def _compute_confidence(
        self, dense_passages: list[dict], graph_context: list[dict]
    ) -> float:
        if not dense_passages and not graph_context:
            return 0.0
        scores = [p.get("score", 0.0) for p in dense_passages + graph_context]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    async def run_batch(self, queries: list[MedicalQuery]) -> list[MedicalQAResponse]:
        return await asyncio.gather(*[self.run(q) for q in queries])
