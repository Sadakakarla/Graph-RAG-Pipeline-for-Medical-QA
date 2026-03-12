"""
BGE-M3 dense retriever over NIH PubMed/PMC corpora.
Supports multi-query retrieval with score fusion (RRF).
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from pydantic import BaseModel


class RetrieverConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    index_path: str = "indices/nih_pubmed_bge_m3.faiss"
    metadata_path: str = "indices/nih_pubmed_metadata.jsonl"
    top_k: int = 10
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 512
    rrf_k: int = 60          # Reciprocal rank fusion constant


class BGERetriever:
    """
    BGE-M3 dense retrieval over NIH corpora (PubMed + PMC).
    Multi-query retrieval with Reciprocal Rank Fusion for score aggregation.
    """

    def __init__(self, cfg: dict):
        self.cfg = RetrieverConfig(**cfg)
        self._model = None
        self._index = None
        self._metadata: list[dict] = []
        self._load_model()

    def _load_model(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.cfg.model_name,
                use_fp16=True,
                device=self.cfg.device,
            )
        except ImportError:
            print("[Retriever] FlagEmbedding not installed. Run: pip install FlagEmbedding")

        try:
            import faiss, json
            self._index = faiss.read_index(self.cfg.index_path)
            with open(self.cfg.metadata_path) as f:
                self._metadata = [json.loads(line) for line in f]
        except Exception as e:
            print(f"[Retriever] Index load failed: {e}. Using mock retrieval.")

    async def retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve passages for multiple queries and fuse with RRF.
        Returns deduplicated, re-ranked passages.
        """
        k = top_k or self.cfg.top_k
        all_results: list[list[dict]] = await asyncio.gather(
            *[self._retrieve_single(q, k * 2) for q in queries]
        )
        return self._reciprocal_rank_fusion(all_results, k)

    async def _retrieve_single(self, query: str, top_k: int) -> list[dict]:
        if self._model is None or self._index is None:
            return self._mock_retrieve(query, top_k)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_retrieve, query, top_k)

    def _sync_retrieve(self, query: str, top_k: int) -> list[dict]:
        embeddings = self._model.encode(
            [query],
            batch_size=1,
            max_length=self.cfg.max_seq_length,
            return_dense=True,
        )["dense_vecs"]

        query_vec = np.array(embeddings, dtype=np.float32)
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._metadata):
                passage = {**self._metadata[idx], "score": float(score), "id": str(idx)}
                results.append(passage)
        return results

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[dict]], top_k: int
    ) -> list[dict]:
        """
        RRF: score(d) = sum(1 / (k + rank)) across all query result lists.
        Handles multi-query result fusion without score normalization issues.
        """
        rrf_scores: dict[str, float] = {}
        doc_registry: dict[str, dict] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get("id", doc.get("pmid", str(rank)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
                doc_registry[doc_id] = doc

        top_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_k]
        return [{**doc_registry[doc_id], "score": rrf_scores[doc_id]} for doc_id in top_ids]

    def _mock_retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "id": f"mock_{i}",
                "pmid": f"PMC{3000000 + i}",
                "title": f"Mock NIH article {i} for: {query[:40]}",
                "abstract": f"This study examines endocrinological findings relevant to: {query}",
                "score": 0.9 - i * 0.05,
            }
            for i in range(top_k)
        ]

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """Batch encode corpus passages for index building."""
        if self._model is None:
            return np.random.rand(len(texts), 1024).astype(np.float32)
        all_embeddings = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i: i + self.cfg.batch_size]
            embs = self._model.encode(batch, batch_size=self.cfg.batch_size, return_dense=True)["dense_vecs"]
            all_embeddings.append(embs)
        return np.vstack(all_embeddings).astype(np.float32)
