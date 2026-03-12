"""
Llama-3 SFT generator for medical QA with citation grounding.
Produces grounded answers with inline citations from retrieved context.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from pydantic import BaseModel


MEDICAL_QA_SYSTEM_PROMPT = """You are a clinical decision support assistant specializing in endocrinology.

Your task is to answer medical questions accurately using ONLY the provided context passages.

Rules:
1. Cite every factual claim using [CITATION:pmid] inline
2. If context is insufficient, say "Insufficient evidence in provided context"
3. Flag any potential contraindications with [CONTRAINDICATION: ...]
4. Structure complex answers as: Diagnosis → Pathophysiology → Treatment → Monitoring
5. Never fabricate drug names, dosages, or study results
6. Use precise medical terminology

Context passages:
{context}

Reasoning trace from knowledge graph:
{hop_traces}
"""


class GeneratorConfig(BaseModel):
    vllm_base_url: str = "http://localhost:8000"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    temperature: float = 0.1       # Low temp for factual medical QA
    max_tokens: int = 1024
    top_p: float = 0.9
    timeout: float = 30.0
    citation_pattern: str = r"\[CITATION:([^\]]+)\]"
    contraindication_pattern: str = r"\[CONTRAINDICATION:([^\]]+)\]"


class LlamaGenerator:
    """
    Llama-3 8B (SFT-tuned on medical QA) via vLLM.
    Extracts citations and contraindication flags from generated text.
    """

    def __init__(self, cfg: dict):
        self.cfg = GeneratorConfig(**cfg)
        self.client = httpx.AsyncClient(
            base_url=self.cfg.vllm_base_url,
            timeout=self.cfg.timeout,
        )

    async def generate(
        self,
        query: str,
        context: list[dict],
        hop_traces: list[dict],
    ) -> tuple[str, list[str], list[str]]:
        """
        Returns (answer, citations, reasoning_steps).
        """
        system = MEDICAL_QA_SYSTEM_PROMPT.format(
            context=self._format_context(context),
            hop_traces=self._format_hop_traces(hop_traces),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            raw_answer = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raw_answer = f"Generation failed: {e}"

        citations = self._extract_citations(raw_answer)
        reasoning = self._extract_reasoning_steps(raw_answer, hop_traces)
        clean_answer = self._clean_answer(raw_answer)

        return clean_answer, citations, reasoning

    def _format_context(self, passages: list[dict]) -> str:
        formatted = []
        for i, p in enumerate(passages[:15]):
            pmid = p.get("pmid", p.get("id", f"doc_{i}"))
            title = p.get("title", "")
            abstract = p.get("abstract", p.get("text", ""))[:500]
            formatted.append(f"[{i+1}] PMID:{pmid} | {title}\n{abstract}")
        return "\n\n".join(formatted)

    def _format_hop_traces(self, traces: list[dict]) -> str:
        if not traces:
            return "No graph traversal performed."
        lines = []
        for t in traces[:10]:
            lines.append(
                f"  Hop {t['hop']}: {t['label']} (SNOMED:{t.get('concept_id','')}) "
                f"[score={t['score']:.3f}]"
            )
        return "\n".join(lines)

    def _extract_citations(self, text: str) -> list[str]:
        matches = re.findall(self.cfg.citation_pattern, text)
        return list(dict.fromkeys(matches))  # Deduplicated, order-preserving

    def _extract_reasoning_steps(self, text: str, hop_traces: list[dict]) -> list[str]:
        steps = []
        # Add graph reasoning steps
        for trace in hop_traces[:5]:
            steps.append(f"Graph hop {trace['hop']}: {trace['label']}")
        # Add any numbered steps in the answer
        numbered = re.findall(r"\d+\.\s+(.+?)(?=\n|$)", text)
        steps.extend(numbered[:5])
        return steps

    def _clean_answer(self, text: str) -> str:
        """Remove citation tags for clean final answer display."""
        clean = re.sub(self.cfg.citation_pattern, "", text)
        return clean.strip()

    async def close(self) -> None:
        await self.client.aclose()
