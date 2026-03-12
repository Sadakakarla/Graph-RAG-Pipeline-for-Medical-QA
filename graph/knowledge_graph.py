"""
SNOMED CT Endocrinology Knowledge Graph.
150K nodes covering conditions, medications, procedures, and relationships.
Supports multi-hop traversal for comorbidity and contraindication queries.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from pydantic import BaseModel


@dataclass
class KGNode:
    node_id: str
    concept_id: str          # SNOMED CT concept ID
    label: str
    category: str            # condition | medication | procedure | finding
    synonyms: list[str] = field(default_factory=list)
    icd10_codes: list[str] = field(default_factory=list)
    score: float = 1.0


@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str            # e.g. "has_comorbidity", "contraindicated_with", "treats"
    weight: float = 1.0
    evidence_pmids: list[str] = field(default_factory=list)


class GraphRAGConfig(BaseModel):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    max_nodes: int = 150_000
    endocrinology_root_id: str = "362969004"   # SNOMED CT endocrinology root
    hop_decay: float = 0.85                    # Score decay per hop


class MedicalKnowledgeGraph:
    """
    GraphRAG over SNOMED CT endocrinology subgraph.
    Uses NetworkX in-memory graph for fast multi-hop traversal;
    Neo4j as persistent store for full 150K-node graph.
    """

    HOP_RELATIONS = {
        "comorbidity": ["has_comorbidity", "associated_with", "co_occurs_with"],
        "contraindication": ["contraindicated_with", "interacts_with", "aggravates"],
        "treatment": ["treats", "manages", "indicated_for"],
        "finding": ["is_finding_of", "manifestation_of"],
    }

    def __init__(self, cfg: dict):
        self.cfg = GraphRAGConfig(**cfg)
        self.graph = nx.DiGraph()
        self._node_index: dict[str, KGNode] = {}

    async def multi_hop_traverse(
        self,
        query: str,
        max_hops: int = 3,
        query_type: str = "multi_hop",
    ) -> tuple[list[dict], list[dict]]:
        """
        Traverse the SNOMED CT graph from seed nodes identified in query.
        Returns (context_nodes, hop_traces).
        """
        seed_nodes = await self._identify_seed_nodes(query)
        if not seed_nodes:
            return [], []

        visited: dict[str, float] = {}
        hop_traces: list[dict] = []
        queue = [(node, 0, 1.0) for node in seed_nodes]

        target_relations = self._get_relations_for_query_type(query_type)

        while queue:
            current_node, hop, score = queue.pop(0)
            node_id = current_node.node_id

            if node_id in visited or hop > max_hops:
                continue
            visited[node_id] = score

            hop_traces.append({
                "node_id": node_id,
                "label": current_node.label,
                "hop": hop,
                "score": score,
                "concept_id": current_node.concept_id,
            })

            if hop < max_hops:
                neighbors = self._get_neighbors(node_id, target_relations)
                for neighbor, relation, weight in neighbors:
                    if neighbor.node_id not in visited:
                        next_score = score * self.cfg.hop_decay * weight
                        queue.append((neighbor, hop + 1, next_score))

        context_nodes = [
            {**self._node_index[nid].__dict__, "score": sc}
            for nid, sc in sorted(visited.items(), key=lambda x: -x[1])
            if nid in self._node_index
        ]
        return context_nodes, hop_traces

    async def _identify_seed_nodes(self, query: str) -> list[KGNode]:
        """
        Identify SNOMED CT concepts mentioned in the query via string matching.
        In production, replace with a medical NER model (e.g. SciSpacy).
        """
        query_lower = query.lower()
        matches = []
        for node in self._node_index.values():
            terms = [node.label.lower()] + [s.lower() for s in node.synonyms]
            if any(term in query_lower for term in terms):
                matches.append(node)
        return matches[:5]  # Top 5 seed nodes

    def _get_neighbors(
        self, node_id: str, target_relations: list[str]
    ) -> list[tuple[KGNode, str, float]]:
        if node_id not in self.graph:
            return []
        neighbors = []
        for _, neighbor_id, data in self.graph.out_edges(node_id, data=True):
            relation = data.get("relation", "")
            if relation in target_relations and neighbor_id in self._node_index:
                neighbors.append((
                    self._node_index[neighbor_id],
                    relation,
                    data.get("weight", 1.0),
                ))
        return neighbors

    def _get_relations_for_query_type(self, query_type: str) -> list[str]:
        if "comorbidity" in query_type:
            return self.HOP_RELATIONS["comorbidity"]
        if "contraindication" in query_type:
            return self.HOP_RELATIONS["contraindication"]
        # Default: all relations for general multi-hop
        return [r for rels in self.HOP_RELATIONS.values() for r in rels]

    def add_node(self, node: KGNode) -> None:
        self._node_index[node.node_id] = node
        self.graph.add_node(node.node_id, **node.__dict__)

    def add_edge(self, edge: KGEdge) -> None:
        self.graph.add_edge(
            edge.source_id, edge.target_id,
            relation=edge.relation,
            weight=edge.weight,
            evidence_pmids=edge.evidence_pmids,
        )

    def load_from_neo4j(self) -> None:
        """Load SNOMED CT endocrinology subgraph from Neo4j into memory."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.cfg.neo4j_uri,
                auth=(self.cfg.neo4j_user, self.cfg.neo4j_password),
            )
            with driver.session() as session:
                # Load nodes
                nodes_result = session.run(
                    "MATCH (n:Concept) WHERE n.domain = 'endocrinology' RETURN n LIMIT $limit",
                    limit=self.cfg.max_nodes,
                )
                for record in nodes_result:
                    n = record["n"]
                    self.add_node(KGNode(
                        node_id=n["id"],
                        concept_id=n.get("snomedId", ""),
                        label=n["label"],
                        category=n.get("category", "condition"),
                        synonyms=n.get("synonyms", []),
                        icd10_codes=n.get("icd10", []),
                    ))
                # Load edges
                edges_result = session.run(
                    "MATCH (a:Concept)-[r]->(b:Concept) WHERE a.domain='endocrinology' RETURN a.id, type(r), b.id, r.weight"
                )
                for record in edges_result:
                    self.add_edge(KGEdge(
                        source_id=record["a.id"],
                        target_id=record["b.id"],
                        relation=record["type(r)"].lower(),
                        weight=record.get("r.weight", 1.0),
                    ))
            driver.close()
        except Exception as e:
            print(f"[KG] Neo4j load failed: {e}. Using in-memory graph only.")

    @property
    def node_count(self) -> int:
        return len(self._node_index)

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()
