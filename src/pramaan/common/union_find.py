"""Generic union-find (disjoint set), shared by anything that clusters
items via pairwise edges: canonical-identity resolution
(`ingest/identity.py`) and, in Phase 1's benchmark builder, the combined
identity+ring clustering that split assignment must respect.
"""

from __future__ import annotations

from collections import defaultdict


class UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent: dict[str, str] = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self) -> dict[str, list[str]]:
        """{root: [members]} - root is arbitrary/order-dependent; callers
        wanting a deterministic canonical id should derive one from the
        member list themselves (e.g. min(members)), not from this root."""
        out: dict[str, list[str]] = defaultdict(list)
        for cid in self.parent:
            out[self.find(cid)].append(cid)
        return out
