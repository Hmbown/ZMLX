"""Tests for ZMLX Discover: LLM-guided kernel search."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from zmlx.discover.candidates import (
    EvalResult,
    KernelCandidate,
    KernelSpec,
)
from zmlx.discover.llm import LLMResponse, MockBackend
from zmlx.discover.prompts import parse_llm_response
from zmlx.discover.reward import compute_reward, geometric_mean
from zmlx.discover.safety import validate_metal_source
from zmlx.discover.session import Session
from zmlx.discover.tree import SearchTree

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_spec() -> KernelSpec:
    return KernelSpec(
        name="kk_test",
        input_names=("inp",),
        output_names=("out",),
        source="uint idx = thread_position_in_grid.x;\nout[idx] = (T)((float)inp[idx]);",
        threadgroup=(256, 1, 1),
    )


@pytest.fixture
def dummy_candidate(dummy_spec: KernelSpec) -> KernelCandidate:
    return KernelCandidate(spec=dummy_spec, generation=0, llm_reasoning="test")


# ---------------------------------------------------------------------------
# Tree tests
# ---------------------------------------------------------------------------

class TestTree:
    def test_puct_selection_unexplored_first(self, dummy_candidate: KernelCandidate):
        """Unexplored nodes should be selected first due to high exploration bonus."""
        # High c_puct so exploration term dominates over the reward of node A
        tree = SearchTree(dummy_candidate, c_puct=10.0)

        # Add two children — one explored, one not
        spec_a = KernelSpec(
            name="a", input_names=("inp",), output_names=("out",),
            source="// variant A",
        )
        spec_b = KernelSpec(
            name="b", input_names=("inp",), output_names=("out",),
            source="// variant B",
        )
        cand_a = KernelCandidate(spec=spec_a, generation=1)
        cand_b = KernelCandidate(spec=spec_b, generation=1)

        nodes = tree.expand(tree.root, [cand_a, cand_b])
        assert len(nodes) == 2

        # Evaluate node A only (low reward so exploration wins)
        nodes[0].eval_result = EvalResult(compiled=True, correct=True, reward=0.5)
        tree.backpropagate(nodes[0], 0.5)

        # Select should pick unexplored node B (higher PUCT due to 0 visits)
        # Node A PUCT: 0.5 + 10 * 0.5 * sqrt(1) / 2 = 0.5 + 2.5 = 3.0
        # Node B PUCT: 0.0 + 10 * 0.5 * sqrt(1) / 1 = 5.0
        selected = tree.select()
        assert selected.node_id == nodes[1].node_id

    def test_max_reward_backprop(self, dummy_candidate: KernelCandidate):
        """Max reward (not mean) should propagate up to root."""
        tree = SearchTree(dummy_candidate, c_puct=1.0)

        spec = KernelSpec(
            name="child", input_names=("inp",), output_names=("out",),
            source="// child",
        )
        cand = KernelCandidate(spec=spec, generation=1)
        nodes = tree.expand(tree.root, [cand])

        # Backprop reward 2.0
        tree.backpropagate(nodes[0], 2.0)
        assert tree.root.max_reward == 2.0

        # Backprop lower reward
        tree.backpropagate(nodes[0], 1.0)
        # Max should stay at 2.0
        assert tree.root.max_reward == 2.0

        # Backprop higher reward
        tree.backpropagate(nodes[0], 3.0)
        assert tree.root.max_reward == 3.0

    def test_tree_serialization(self, dummy_candidate: KernelCandidate):
        """Tree should roundtrip through JSON serialization."""
        tree = SearchTree(dummy_candidate, c_puct=2.0)

        spec = KernelSpec(
            name="child", input_names=("inp",), output_names=("out",),
            source="// test source",
        )
        cand = KernelCandidate(spec=spec, generation=1, llm_reasoning="vectorize")
        nodes = tree.expand(tree.root, [cand])
        nodes[0].eval_result = EvalResult(
            compiled=True, correct=True,
            timings_us=[10.0, 11.0, 9.5],
            median_us=10.0, reward=1.5, speedup=1.5,
        )
        tree.backpropagate(nodes[0], 1.5)

        data = tree.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        tree2 = SearchTree.from_dict(restored_data)

        assert tree2.c_puct == 2.0
        assert tree2.total_nodes == 2
        assert tree2.root.max_reward == 1.5

    def test_best_node(self, dummy_candidate: KernelCandidate):
        tree = SearchTree(dummy_candidate, c_puct=1.0)

        specs = [
            KernelSpec(name=f"v{i}", input_names=("inp",), output_names=("out",),
                       source=f"// variant {i}")
            for i in range(3)
        ]
        cands = [KernelCandidate(spec=s, generation=1) for s in specs]
        nodes = tree.expand(tree.root, cands)

        rewards = [0.5, 2.0, 1.0]
        for node, r in zip(nodes, rewards, strict=True):
            node.eval_result = EvalResult(compiled=True, correct=True, reward=r)
            tree.backpropagate(node, r)

        best = tree.best_node()
        assert best.eval_result is not None
        assert best.eval_result.reward == 2.0


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestReward:
    def test_speedup_scaling(self):
        """Faster timings should give higher reward."""
        baseline = 100.0
        fast = compute_reward([50.0, 50.0, 50.0], baseline)
        slow = compute_reward([200.0, 200.0, 200.0], baseline)
        assert fast > slow
        assert fast > 1.0  # Faster than baseline
        assert slow < 1.0  # Slower than baseline

    def test_zero_for_empty(self):
        assert compute_reward([], 100.0) == 0.0

    def test_zero_for_zero_baseline(self):
        assert compute_reward([10.0], 0.0) == 0.0

    def test_clamped_to_10(self):
        # Baseline 1000, candidate 0.001 -> very high ratio, should clamp
        result = compute_reward([0.001], 1000.0)
        assert result == 10.0

    def test_geometric_mean(self):
        assert abs(geometric_mean([4.0, 16.0]) - 8.0) < 1e-6
        assert geometric_mean([]) == 0.0
        assert geometric_mean([-1.0]) == 0.0


# ---------------------------------------------------------------------------
# Safety tests
# ---------------------------------------------------------------------------

class TestSafety:
    def test_infinite_loop_detection(self):
        source = "while(true) { /* spin */ }"
        warnings = validate_metal_source(source)
        assert any("infinite loop" in w.lower() for w in warnings)

    def test_large_alloc_detection(self):
        source = "threadgroup float buf[999999];"
        warnings = validate_metal_source(source)
        assert any("threadgroup allocation" in w.lower() for w in warnings)

    def test_clean_source_no_warnings(self):
        source = """
        uint idx = thread_position_in_grid.x;
        threadgroup float buf[256];
        out[idx] = (T)((float)inp[idx]);
        """
        warnings = validate_metal_source(source)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSession:
    def test_save_load_roundtrip(self):
        session = Session.new("moe_combine", "mock", {"chip": "M4 Max", "memory_gb": 36})
        session.metadata.total_steps = 5
        session.metadata.best_speedup = 1.5
        session.eval_history = [{"step": 1, "reward": 1.0}]
        session.candidate_sources["node1"] = "// source"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_session.json"
            session.save(path)

            loaded = Session.load(path)
            assert loaded.metadata.target_name == "moe_combine"
            assert loaded.metadata.total_steps == 5
            assert loaded.metadata.best_speedup == 1.5
            assert loaded.eval_history == [{"step": 1, "reward": 1.0}]
            assert loaded.candidate_sources["node1"] == "// source"
            assert loaded.schema_version == "1.0"


# ---------------------------------------------------------------------------
# Prompt parsing tests
# ---------------------------------------------------------------------------

class TestPromptParsing:
    def test_parse_variant_format(self):
        raw = """
Here are some variants:

---VARIANT---
REASONING: Use float4 vectorization for better memory throughput
THREADGROUP: 128, 1, 1
SOURCE:
uint idx = thread_position_in_grid.x * 4;
float4 v = *((device float4*)(inp + idx));
*((device float4*)(out + idx)) = float4((T)v.x, (T)v.y, (T)v.z, (T)v.w);

---VARIANT---
REASONING: Loop unrolling with 2x factor
THREADGROUP: 256, 1, 1
SOURCE:
uint idx = thread_position_in_grid.x;
uint stride = threads_per_grid.x;
for (uint i = idx; i < N; i += stride * 2) {
    out[i] = (T)((float)inp[i]);
    out[i + stride] = (T)((float)inp[i + stride]);
}
"""
        results = parse_llm_response(raw)
        assert len(results) == 2

        assert results[0]["reasoning"] == "Use float4 vectorization for better memory throughput"
        assert results[0]["threadgroup"] == (128, 1, 1)
        assert "float4" in results[0]["source"]

        assert results[1]["threadgroup"] == (256, 1, 1)
        assert "stride" in results[1]["source"]

    def test_parse_empty_response(self):
        assert parse_llm_response("") == []

    def test_parse_no_separators(self):
        # If no separators, should not parse any variants
        raw = "Just some random text with no variants"
        assert parse_llm_response(raw) == []


# ---------------------------------------------------------------------------
# LLM backend tests
# ---------------------------------------------------------------------------

class TestMockBackend:
    def test_mock_generates_candidates(self):
        backend = MockBackend()
        response = backend.generate_candidates("system", "user", n_candidates=3)
        assert len(response.candidates) == 3
        assert response.model_id == "mock"

    def test_mock_with_preloaded_responses(self):
        preloaded = LLMResponse(
            candidates=[{"source": "// custom", "reasoning": "test", "threadgroup": (32, 1, 1)}],
            raw_response="preloaded",
            model_id="mock",
        )
        backend = MockBackend(responses=[preloaded])
        response = backend.generate_candidates("system", "user")
        assert len(response.candidates) == 1
        assert response.candidates[0]["source"] == "// custom"


# ---------------------------------------------------------------------------
# Candidates tests
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_source_hash_deterministic(self, dummy_spec: KernelSpec):
        h1 = dummy_spec.source_hash
        h2 = dummy_spec.source_hash
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_candidate_id(self, dummy_candidate: KernelCandidate):
        cid = dummy_candidate.candidate_id
        assert cid.startswith("gen0_")
        assert len(cid) > 5


# ---------------------------------------------------------------------------
# Metal-dependent tests (auto-skip on non-Apple-Silicon)
# ---------------------------------------------------------------------------

@pytest.mark.metal
class TestEvaluate:
    def test_evaluate_correct_kernel(self):
        """A known-good identity kernel should pass evaluation."""
        import mlx.core as mx

        from zmlx.discover.evaluate import evaluate_candidate

        spec = KernelSpec(
            name="kk_identity_test",
            input_names=("inp",),
            output_names=("out",),
            source="uint idx = thread_position_in_grid.x;\nout[idx] = inp[idx];",
            template_params=(("T", "float32"),),
        )
        candidate = KernelCandidate(spec=spec, generation=0)

        test_input = mx.array([1.0, 2.0, 3.0, 4.0])

        def ref_fn(*inputs):
            return inputs[0]

        result = evaluate_candidate(
            candidate, ref_fn, [test_input],
            baseline_us=100.0,
            warmup=2, iters=5, timeout_s=5.0,
            output_shapes=[(4,)],
            output_dtypes=[mx.float32],
            grid=(4, 1, 1),
            threadgroup=(4, 1, 1),
            template=[("T", mx.float32)],
        )
        assert result.compiled
        assert result.correct
        assert result.median_us > 0
        assert result.reward > 0

    def test_evaluate_bad_kernel(self):
        """A kernel with syntax errors should fail gracefully."""
        import mlx.core as mx

        from zmlx.discover.evaluate import evaluate_candidate

        spec = KernelSpec(
            name="kk_bad_test",
            input_names=("inp",),
            output_names=("out",),
            source="THIS IS NOT VALID METAL CODE !!!",
        )
        candidate = KernelCandidate(spec=spec, generation=0)

        test_input = mx.array([1.0, 2.0])

        def ref_fn(*inputs):
            return inputs[0]

        result = evaluate_candidate(
            candidate, ref_fn, [test_input],
            baseline_us=100.0,
            warmup=1, iters=2, timeout_s=5.0,
        )
        # Should not crash — should record failure
        assert not result.correct
        assert result.compile_error is not None or result.correctness_error is not None


# ---------------------------------------------------------------------------
# E2E mock search test
# ---------------------------------------------------------------------------

class TestE2EMock:
    def test_mock_search(self, dummy_candidate: KernelCandidate):
        """Full loop with MockBackend, 2 steps, 2 candidates per step."""
        from zmlx.discover.llm import MockBackend
        from zmlx.discover.tree import SearchTree

        tree = SearchTree(dummy_candidate, c_puct=1.0)
        backend = MockBackend()

        # Simulate 2 steps
        for step in range(2):
            parent = tree.select()

            response = backend.generate_candidates(
                "system", "user", n_candidates=2,
            )

            new_cands = []
            for i, cd in enumerate(response.candidates):
                spec = KernelSpec(
                    name=f"mock_step{step}_{i}",
                    input_names=("inp",),
                    output_names=("out",),
                    source=cd["source"],
                )
                new_cands.append(KernelCandidate(
                    spec=spec, parent_id=parent.node_id, generation=step + 1,
                ))

            new_nodes = tree.expand(parent, new_cands)
            for node in new_nodes:
                # Simulate evaluation
                reward = 0.5 + step * 0.3
                node.eval_result = EvalResult(
                    compiled=True, correct=True,
                    timings_us=[100.0], median_us=100.0,
                    reward=reward, speedup=reward,
                )
                tree.backpropagate(node, reward)

        assert tree.total_nodes >= 5  # root + 4 children
        assert tree.root.max_reward > 0
        best = tree.best_node()
        assert best.eval_result is not None
