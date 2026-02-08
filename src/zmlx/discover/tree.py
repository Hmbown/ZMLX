"""PUCT tree search for kernel optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .candidates import EvalResult, KernelCandidate


@dataclass
class Node:
    """A node in the search tree."""

    candidate: KernelCandidate
    eval_result: EvalResult | None = None
    parent: Node | None = field(default=None, repr=False)
    children: list[Node] = field(default_factory=list)
    visit_count: int = 0
    max_reward: float = 0.0
    prior: float = 1.0
    node_id: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = self.candidate.candidate_id

    def puct_score(self, c_puct: float) -> float:
        """Compute the PUCT score for this node.

        PUCT = max_reward + c_puct * prior * sqrt(parent_visits) / (1 + visits)

        Unexplored nodes (visit_count == 0) get a high score.
        """
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.max_reward + exploration

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evaluated(self) -> bool:
        return self.eval_result is not None


class SearchTree:
    """PUCT-based tree search for kernel candidates.

    Uses max-reward backpropagation (not mean) — the best result found
    in a subtree propagates up to the root.
    """

    def __init__(self, root_candidate: KernelCandidate, c_puct: float = 1.0):
        self.root = Node(candidate=root_candidate, node_id="root")
        self.c_puct = c_puct
        self._all_nodes: dict[str, Node] = {"root": self.root}

    def select(self) -> Node:
        """Walk from root choosing the child with max PUCT score.

        Returns a leaf node for expansion or evaluation.
        """
        node = self.root
        while not node.is_leaf:
            best_child = max(node.children, key=lambda c: c.puct_score(self.c_puct))
            node = best_child
        return node

    def expand(self, parent: Node, candidates: list[KernelCandidate]) -> list[Node]:
        """Create child nodes from generated candidates.

        Returns the newly created nodes.
        """
        new_nodes = []
        for cand in candidates:
            nid = cand.candidate_id
            # Deduplicate by source hash
            if nid in self._all_nodes:
                continue
            child = Node(
                candidate=cand,
                parent=parent,
                prior=1.0 / max(len(candidates), 1),
                node_id=nid,
            )
            parent.children.append(child)
            self._all_nodes[nid] = child
            new_nodes.append(child)
        return new_nodes

    def backpropagate(self, node: Node, reward: float) -> None:
        """Propagate max reward up to root."""
        node.visit_count += 1
        node.max_reward = max(node.max_reward, reward)

        current = node.parent
        while current is not None:
            current.visit_count += 1
            current.max_reward = max(current.max_reward, reward)
            current = current.parent

    def best_node(self) -> Node:
        """Return the evaluated node with the highest reward across the entire tree."""
        best: Node | None = None
        best_reward = -1.0
        for node in self._all_nodes.values():
            if node.eval_result is not None and node.eval_result.reward > best_reward:
                best = node
                best_reward = node.eval_result.reward
        return best if best is not None else self.root

    @property
    def total_nodes(self) -> int:
        return len(self._all_nodes)

    @property
    def evaluated_nodes(self) -> list[Node]:
        return [n for n in self._all_nodes.values() if n.is_evaluated]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tree to a JSON-compatible dict."""

        def _node_dict(node: Node) -> dict[str, Any]:
            d: dict[str, Any] = {
                "node_id": node.node_id,
                "candidate": {
                    "spec": {
                        "name": node.candidate.spec.name,
                        "input_names": list(node.candidate.spec.input_names),
                        "output_names": list(node.candidate.spec.output_names),
                        "source": node.candidate.spec.source,
                        "header": node.candidate.spec.header,
                        "threadgroup": list(node.candidate.spec.threadgroup),
                        "template_params": [
                            list(p) for p in node.candidate.spec.template_params
                        ],
                    },
                    "parent_id": node.candidate.parent_id,
                    "generation": node.candidate.generation,
                    "llm_reasoning": node.candidate.llm_reasoning,
                },
                "visit_count": node.visit_count,
                "max_reward": node.max_reward,
                "prior": node.prior,
            }
            if node.eval_result is not None:
                d["eval_result"] = {
                    "compiled": node.eval_result.compiled,
                    "correct": node.eval_result.correct,
                    "compile_error": node.eval_result.compile_error,
                    "correctness_error": node.eval_result.correctness_error,
                    "timings_us": node.eval_result.timings_us,
                    "median_us": node.eval_result.median_us,
                    "reward": node.eval_result.reward,
                    "speedup": node.eval_result.speedup,
                }
            d["children"] = [_node_dict(c) for c in node.children]
            return d

        return {
            "c_puct": self.c_puct,
            "root": _node_dict(self.root),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchTree:
        """Deserialize a tree from a dict."""
        from .candidates import KernelSpec

        c_puct = data.get("c_puct", 1.0)

        def _build_node(nd: dict[str, Any], parent: Node | None) -> Node:
            cd = nd["candidate"]
            sd = cd["spec"]
            spec = KernelSpec(
                name=sd["name"],
                input_names=tuple(sd["input_names"]),
                output_names=tuple(sd["output_names"]),
                source=sd["source"],
                header=sd.get("header", ""),
                threadgroup=tuple(sd.get("threadgroup", [256, 1, 1])),
                template_params=tuple(
                    tuple(p) for p in sd.get("template_params", [])
                ),
            )
            cand = KernelCandidate(
                spec=spec,
                parent_id=cd.get("parent_id"),
                generation=cd.get("generation", 0),
                llm_reasoning=cd.get("llm_reasoning", ""),
            )

            er_data = nd.get("eval_result")
            er = None
            if er_data is not None:
                er = EvalResult(
                    compiled=er_data["compiled"],
                    correct=er_data["correct"],
                    compile_error=er_data.get("compile_error"),
                    correctness_error=er_data.get("correctness_error"),
                    timings_us=er_data.get("timings_us", []),
                    median_us=er_data.get("median_us", float("inf")),
                    reward=er_data.get("reward", 0.0),
                    speedup=er_data.get("speedup", 0.0),
                )

            node = Node(
                candidate=cand,
                eval_result=er,
                parent=parent,
                visit_count=nd.get("visit_count", 0),
                max_reward=nd.get("max_reward", 0.0),
                prior=nd.get("prior", 1.0),
                node_id=nd.get("node_id", cand.candidate_id),
            )

            for child_d in nd.get("children", []):
                child = _build_node(child_d, node)
                node.children.append(child)

            return node

        root_data = data["root"]
        dummy_spec = KernelSpec(
            name="dummy",
            input_names=(),
            output_names=(),
            source="",
        )
        dummy_cand = KernelCandidate(spec=dummy_spec)
        tree = cls(dummy_cand, c_puct=c_puct)

        # Replace the dummy root
        tree.root = _build_node(root_data, None)
        tree._all_nodes = {}

        # Rebuild the _all_nodes index
        def _index(node: Node) -> None:
            tree._all_nodes[node.node_id] = node
            for c in node.children:
                _index(c)

        _index(tree.root)
        return tree
