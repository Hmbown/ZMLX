"""Tests for the kernel-discovery subsystem."""

from __future__ import annotations

import json

import pytest

from zmlx.kd import registry as kd_registry
from zmlx.kd.archive import RunArchive
from zmlx.kd.cli import _discrete_search_space_size, build_parser, main
from zmlx.kd.eval import evaluate_candidate
from zmlx.kd.features import candidate_vector, collect_feature_keys
from zmlx.kd.graph import build_knn_graph
from zmlx.kd.model_shapes import derive_shape_suite_from_config, parse_decode_rows
from zmlx.kd.mutations import initial_population
from zmlx.kd.ops import get_op
from zmlx.kd.promotion import PromotionPolicy, select_promoted_entries
from zmlx.kd.registry import get_kernel
from zmlx.kd.report import best_kernels_payload
from zmlx.kd.tour import build_union_tours, schedule_batch
from zmlx.kd.types import KernelCandidate


def _candidate_ids(seed: int) -> list[str]:
    op = get_op("rmsnorm")
    shape = op.normalize_shape({"rows": 2, "D": 64})
    pop = initial_population(
        op_module=op,
        shape=shape,
        dtype_name="float16",
        seed=seed,
        count=8,
    )
    return [cand.candidate_id for cand in pop]


def test_candidate_generation_is_deterministic() -> None:
    ids_a = _candidate_ids(11)
    ids_b = _candidate_ids(11)
    ids_c = _candidate_ids(13)
    assert ids_a == ids_b
    assert ids_a != ids_c


def test_candidate_id_is_stable_from_source_and_params() -> None:
    op = get_op("swiglu")
    shape = op.normalize_shape({"rows": 2, "D": 64})
    candidate = op.make_candidate(
        template_params=op.seed_template_params(),
        launch_params=op.seed_launch_params(),
        shape=shape,
        dtype_name="float16",
        parent_id=None,
    )

    rebuilt = KernelCandidate.build_candidate_id(
        op_name=candidate.op_name,
        metal_source=candidate.metal_source,
        func_name=candidate.func_name,
        inputs_spec=candidate.inputs_spec,
        outputs_spec=candidate.outputs_spec,
        template_params=candidate.template_params,
        launch_params=candidate.launch_params,
    )
    assert rebuilt == candidate.candidate_id

    mutated = KernelCandidate.build_candidate_id(
        op_name=candidate.op_name,
        metal_source=candidate.metal_source,
        func_name=candidate.func_name,
        inputs_spec=candidate.inputs_spec,
        outputs_spec=candidate.outputs_spec,
        template_params={**candidate.template_params, "unroll": 4},
        launch_params=candidate.launch_params,
    )
    assert mutated != candidate.candidate_id


def test_schedule_batch_is_deterministic_with_seed() -> None:
    op = get_op("rmsnorm")
    shape = op.normalize_shape({"rows": 2, "D": 64})
    candidates = initial_population(
        op_module=op,
        shape=shape,
        dtype_name="float16",
        seed=5,
        count=12,
    )
    candidates[0].status = "benchmarked"
    candidates[0].metrics = {"speedup_vs_ref": 1.05, "latency_us": 1.0}

    keys = collect_feature_keys(candidates)
    vectors = [candidate_vector(c, keys) for c in candidates]
    graph = build_knn_graph(vectors, k=4)
    tours = build_union_tours(graph, len(candidates), n_tours=3, seed=42)

    kwargs = {
        "candidates": candidates,
        "vectors": vectors,
        "graph": graph,
        "tours": tours,
        "batch_size": 5,
        "step": 2,
        "seed": 123,
        "exploit_fraction": 0.3,
        "novelty_fraction": 0.3,
        "min_tour_gap": 2,
    }
    batch_a = schedule_batch(**kwargs)
    batch_b = schedule_batch(**kwargs)
    assert batch_a == batch_b
    assert len(set(batch_a)) == len(batch_a)


def test_discrete_search_space_size_for_phase0_ops() -> None:
    assert _discrete_search_space_size(get_op("rope")) == 8
    assert _discrete_search_space_size(get_op("swiglu")) == 96
    assert _discrete_search_space_size(get_op("rmsnorm_residual")) == 72


def test_registry_launch_shapes_for_all_supported_kinds() -> None:
    class _Array:
        def __init__(self, size: int):
            self.size = size

    grid, tg = kd_registry._compute_launch(  # noqa: SLF001
        {"launch_kind": "rmsnorm_rows_tg", "threadgroup_x": 128},
        {"rows": 3, "D": 64},
        [_Array(192)],
    )
    assert grid == (384, 1, 1)
    assert tg == (128, 1, 1)

    grid, tg = kd_registry._compute_launch(  # noqa: SLF001
        {"launch_kind": "rmsnorm_residual_rows_tg", "threadgroup_x": 64},
        {"rows": 4, "D": 64},
        [_Array(256)],
    )
    assert grid == (256, 1, 1)
    assert tg == (64, 1, 1)

    grid, tg = kd_registry._compute_launch(  # noqa: SLF001
        {"launch_kind": "swiglu_flat", "threadgroup_x": 256, "vec_width": 2, "unroll": 4},
        {"N": 100},
        [_Array(100)],
    )
    assert grid == (13, 1, 1)
    assert tg == (13, 1, 1)

    grid, tg = kd_registry._compute_launch(  # noqa: SLF001
        {"launch_kind": "rope_decode_pos", "threadgroup_x": 256},
        {"B": 2, "H_Q": 8, "D_OUT": 96},
        [_Array(0)],
    )
    assert grid == (1728, 1, 1)
    assert tg == (256, 1, 1)

    grid, tg = kd_registry._compute_launch(  # noqa: SLF001
        {"threadgroup_x": 32},
        {},
        [_Array(20)],
    )
    assert grid == (20, 1, 1)
    assert tg == (20, 1, 1)


@pytest.mark.metal
def test_correctness_harness_catches_broken_kernel() -> None:
    op = get_op("rmsnorm")
    shape = op.normalize_shape({"rows": 2, "D": 64})

    seed_tpl = op.seed_template_params()
    seed_launch = op.seed_launch_params()
    base = op.make_candidate(
        template_params=seed_tpl,
        launch_params=seed_launch,
        shape=shape,
        dtype_name="float32",
        parent_id=None,
    )
    broken_source = base.metal_source.replace(
        "out[base + idx] = (T)(x * inv * w);",
        "out[base + idx] = (T)(0);",
    )

    broken = KernelCandidate(
        op_name=base.op_name,
        candidate_id="",
        metal_source=broken_source,
        func_name=f"{base.func_name}_broken",
        inputs_spec=base.inputs_spec,
        outputs_spec=base.outputs_spec,
        template_params=base.template_params,
        launch_params=base.launch_params,
        features=base.features,
        parent_id=base.candidate_id,
        notes=base.notes,
    )

    result = evaluate_candidate(
        candidate=broken,
        op_module=op,
        dtype_name="float32",
        shape_suite=[shape],
        seed=0,
        warmup=1,
        iters=2,
        baseline_cache={},
    )

    assert result.status == "failed"
    assert result.metrics.get("failure") == "correctness_error"


def test_registry_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    payload = {
        "schema_version": "2",
        "runtime": {
            "mlx_version": "unknown",
            "device_arch": "unknown",
            "device_name": "unknown",
        },
        "entries": [
            {
                "key": {
                    "op_name": "rmsnorm",
                    "mlx_version": "unknown",
                    "device_arch": "unknown",
                    "device_name": "unknown",
                    "dtype": "float16",
                    "shape_signature": {"rows": 1, "D": 64},
                },
                "candidate_id": "rmsnorm_deadbeef",
                "func_name": "kk_test",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 1.0},
            }
        ],
    }

    path = tmp_path / "pins.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("ZMLX_USE_DISCOVERED_KERNELS", "0")
    assert get_kernel("rmsnorm", "float16", {"rows": 1, "D": 64}, pinned_path=path) is None

    monkeypatch.setenv("ZMLX_USE_DISCOVERED_KERNELS", "1")
    entry = get_kernel("rmsnorm", "float16", {"rows": 1, "D": 64}, pinned_path=path)
    assert entry is not None
    assert entry["candidate_id"] == "rmsnorm_deadbeef"

    assert get_kernel("rmsnorm", "float16", {"rows": 2, "D": 64}, pinned_path=path) is None


def test_registry_runtime_key_matching_prefers_exact(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    key_shape = {"rows": 1, "D": 64}
    payload = {
        "schema_version": "2",
        "runtime": {
            "mlx_version": "0.0.0",
            "device_arch": "unknown",
            "device_name": "unknown",
        },
        "entries": [
            {
                "key": {
                    "op_name": "rmsnorm",
                    "mlx_version": "unknown",
                    "device_arch": "unknown",
                    "device_name": "unknown",
                    "dtype": "float16",
                    "shape_signature": key_shape,
                },
                "candidate_id": "wildcard",
                "func_name": "kk_wildcard",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 1.0},
            },
            {
                "key": {
                    "op_name": "rmsnorm",
                    "mlx_version": "0.30.5",
                    "device_arch": "apple-gpu-g16",
                    "device_name": "M4 Max",
                    "dtype": "float16",
                    "shape_signature": key_shape,
                },
                "candidate_id": "exact",
                "func_name": "kk_exact",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 3.0},
            },
            {
                "key": {
                    "op_name": "rmsnorm",
                    "mlx_version": "0.30.5",
                    "device_arch": "other-arch",
                    "device_name": "M4 Max",
                    "dtype": "float16",
                    "shape_signature": key_shape,
                },
                "candidate_id": "wrong_arch",
                "func_name": "kk_wrong_arch",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 0.2},
            },
        ],
    }
    path = tmp_path / "pins_runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("ZMLX_USE_DISCOVERED_KERNELS", "1")

    exact_runtime = {
        "mlx_version": "0.30.5",
        "device_arch": "apple-gpu-g16",
        "device_name": "M4 Max",
    }
    entry = get_kernel(
        "rmsnorm",
        "float16",
        key_shape,
        pinned_path=path,
        runtime=exact_runtime,
    )
    assert entry is not None
    assert entry["candidate_id"] == "exact"

    wildcard_runtime = {
        "mlx_version": "0.31.0",
        "device_arch": "unknown-arch",
        "device_name": "unknown-device",
    }
    fallback = get_kernel(
        "rmsnorm",
        "float16",
        key_shape,
        pinned_path=path,
        runtime=wildcard_runtime,
    )
    assert fallback is not None
    assert fallback["candidate_id"] == "wildcard"

    assert (
        get_kernel(
            "rmsnorm",
            "float16",
            {"rows": 2, "D": 64},
            pinned_path=path,
            runtime=exact_runtime,
        )
        is None
    )
    assert get_kernel("rmsnorm", "float32", key_shape, pinned_path=path, runtime=exact_runtime) is None


def test_install_and_pack_merge_by_runtime_key(tmp_path) -> None:
    runtime = {
        "mlx_version": "0.30.5",
        "device_arch": "apple-gpu-g16",
        "device_name": "M4 Max",
    }
    rms_key = {
        "op_name": "rmsnorm",
        "mlx_version": runtime["mlx_version"],
        "device_arch": runtime["device_arch"],
        "device_name": runtime["device_name"],
        "dtype": "float16",
        "shape_signature": {"rows": 1, "D": 64},
    }
    rope_key = {
        "op_name": "rope",
        "mlx_version": runtime["mlx_version"],
        "device_arch": runtime["device_arch"],
        "device_name": runtime["device_name"],
        "dtype": "float16",
        "shape_signature": {"B": 1, "H_Q": 8, "D_NOPE": 64, "D_ROPE": 32, "D_OUT": 96},
    }

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    payload_a = {
        "schema_version": "2",
        "runtime": runtime,
        "entries": [
            {
                "key": rms_key,
                "candidate_id": "rms_slow",
                "func_name": "kk_rms_slow",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 3.0},
            }
        ],
    }
    payload_b = {
        "schema_version": "2",
        "runtime": runtime,
        "entries": [
            {
                "key": rms_key,
                "candidate_id": "rms_fast",
                "func_name": "kk_rms_fast",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 1.0},
            },
            {
                "key": rope_key,
                "candidate_id": "rope_fast",
                "func_name": "kk_rope_fast",
                "metal_source": "",
                "inputs_spec": [],
                "outputs_spec": [],
                "template_params": {},
                "launch_params": {},
                "source_hash": "",
                "metrics": {"latency_us": 2.0},
            },
        ],
    }

    (run_a / "best_kernels.json").write_text(json.dumps(payload_a), encoding="utf-8")
    (run_b / "best_kernels.json").write_text(json.dumps(payload_b), encoding="utf-8")

    installed = tmp_path / "configs" / "discovered_kernels.json"
    main(["install", "--run", str(run_a), "--output", str(installed)])
    main(["install", "--run", str(run_b), "--output", str(installed)])

    merged_install = json.loads(installed.read_text(encoding="utf-8"))
    by_id = {entry["candidate_id"]: entry for entry in merged_install["entries"]}
    assert "rms_fast" in by_id
    assert "rms_slow" not in by_id
    assert "rope_fast" in by_id

    pack_out = tmp_path / "kernelpacks" / "pack.json"
    main(["pack", "--runs", str(run_a), str(run_b), "--out", str(pack_out)])
    merged_pack = json.loads(pack_out.read_text(encoding="utf-8"))
    by_pack_id = {entry["candidate_id"]: entry for entry in merged_pack["entries"]}
    assert "rms_fast" in by_pack_id
    assert "rms_slow" not in by_pack_id
    assert "rope_fast" in by_pack_id


def test_best_payload_key_fields_are_pinning_tuple() -> None:
    op = get_op("rmsnorm")
    shape = op.normalize_shape({"rows": 1, "D": 64})
    candidate = op.make_candidate(
        template_params=op.seed_template_params(),
        launch_params=op.seed_launch_params(),
        shape=shape,
        dtype_name="float16",
        parent_id=None,
    )
    candidate.status = "benchmarked"
    candidate.metrics = {
        "latency_us": 1.0,
        "speedup_vs_ref": 1.0,
        "correctness_max_abs_err": 0.0,
        "correctness_max_rel_err": 0.0,
        "dtype": "float16",
        "per_shape": [
            {
                "shape": op.shape_signature(shape),
            }
        ],
    }
    payload = best_kernels_payload(
        [candidate],
        runtime_env={
            "mlx_version": "0.30.6",
            "device_arch": "applegpu_g16s",
            "device_name": "Apple M4 Max",
        },
    )
    entry = payload["entries"][0]
    assert set(entry["key"]) == {
        "op_name",
        "mlx_version",
        "device_arch",
        "device_name",
        "dtype",
        "shape_signature",
    }


def test_cli_help() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "--help"])
    assert exc.value.code == 0


def test_archive_candidate_order_is_deterministic(tmp_path) -> None:
    archive = RunArchive(
        out_dir=tmp_path,
        op_name="rmsnorm",
        seed=0,
        budget=2,
        dtype_name="float16",
        shape_suite="default",
        runtime_env={},
    )
    cand_b = KernelCandidate(
        op_name="rmsnorm",
        candidate_id="",
        metal_source="source_b",
        func_name="fn_b",
        inputs_spec=[],
        outputs_spec=[],
        template_params={},
        launch_params={},
    )
    cand_a = KernelCandidate(
        op_name="rmsnorm",
        candidate_id="",
        metal_source="source_a",
        func_name="fn_a",
        inputs_spec=[],
        outputs_spec=[],
        template_params={},
        launch_params={},
    )
    archive.register_candidate(cand_b)
    archive.register_candidate(cand_a)

    ids = [cand.candidate_id for cand in archive.all_candidates()]
    assert ids == sorted(ids)


def test_promotion_selects_best_per_shape() -> None:
    shape_sig = {"rows": 1, "D": 64}
    cand_fast = KernelCandidate(
        op_name="swiglu",
        candidate_id="fast",
        metal_source="src_fast",
        func_name="fn_fast",
        inputs_spec=[],
        outputs_spec=[],
        template_params={},
        launch_params={},
    )
    cand_fast.status = "benchmarked"
    cand_fast.metrics = {
        "dtype": "float16",
        "per_shape": [
            {
                "shape": shape_sig,
                "latency_us": 1.0,
                "speedup_vs_ref": 1.12,
                "speedup_p10": 1.08,
                "speedup_p90": 1.18,
            }
        ],
    }

    cand_slow = KernelCandidate(
        op_name="swiglu",
        candidate_id="slow",
        metal_source="src_slow",
        func_name="fn_slow",
        inputs_spec=[],
        outputs_spec=[],
        template_params={},
        launch_params={},
    )
    cand_slow.status = "benchmarked"
    cand_slow.metrics = {
        "dtype": "float16",
        "per_shape": [
            {
                "shape": shape_sig,
                "latency_us": 1.2,
                "speedup_vs_ref": 1.08,
                "speedup_p10": 1.04,
                "speedup_p90": 1.11,
            }
        ],
    }

    policy = PromotionPolicy(min_speedup_p10=1.05, noise_guard=0.5, max_noise_pct=0.5)
    selection = select_promoted_entries([cand_slow, cand_fast], runtime_env={}, policy=policy)

    assert selection.promoted_count == 1
    entry = selection.payload["entries"][0]
    assert entry["candidate_id"] == "fast"


def test_promotion_gate_rejects_unclear_speedup() -> None:
    shape_sig = {"rows": 1, "D": 64}
    cand = KernelCandidate(
        op_name="swiglu",
        candidate_id="maybe",
        metal_source="src_maybe",
        func_name="fn_maybe",
        inputs_spec=[],
        outputs_spec=[],
        template_params={},
        launch_params={},
    )
    cand.status = "benchmarked"
    cand.metrics = {
        "dtype": "float16",
        "per_shape": [
            {
                "shape": shape_sig,
                "latency_us": 1.0,
                "speedup_vs_ref": 1.01,
                "speedup_p10": 1.00,
                "speedup_p90": 1.02,
            }
        ],
    }

    policy = PromotionPolicy(min_speedup_p10=1.02, noise_guard=0.5, max_noise_pct=0.5)
    selection = select_promoted_entries([cand], runtime_env={}, policy=policy)
    assert selection.promoted_count == 0


def test_parse_decode_rows_is_stable() -> None:
    assert parse_decode_rows("1,2,4") == (1, 2, 4)
    assert parse_decode_rows("4, 2, 2, 1") == (4, 2, 1)
    with pytest.raises(ValueError):
        parse_decode_rows("")
    with pytest.raises(ValueError):
        parse_decode_rows("1,0,2")


def test_model_shapes_from_glm_config() -> None:
    config = {
        "model_type": "glm4_moe_lite",
        "hidden_size": 2048,
        "moe_intermediate_size": 1536,
        "n_shared_experts": 1,
        "num_attention_heads": 20,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
    }
    rows = (1, 2, 4)
    assert derive_shape_suite_from_config("rmsnorm_residual", config, decode_rows=rows) == [
        {"rows": 1, "D": 2048},
        {"rows": 2, "D": 2048},
        {"rows": 4, "D": 2048},
    ]
    assert derive_shape_suite_from_config("swiglu", config, decode_rows=rows) == [
        {"rows": 1, "D": 1536},
        {"rows": 2, "D": 1536},
        {"rows": 4, "D": 1536},
    ]
    assert derive_shape_suite_from_config("rope", config, decode_rows=rows) == [
        {"B": 1, "H_Q": 20, "D_NOPE": 192, "D_ROPE": 64},
        {"B": 2, "H_Q": 20, "D_NOPE": 192, "D_ROPE": 64},
        {"B": 4, "H_Q": 20, "D_NOPE": 192, "D_ROPE": 64},
    ]


def test_best_payload_picks_per_shape_winner() -> None:
    shared = {
        "op_name": "swiglu",
        "inputs_spec": [],
        "outputs_spec": [],
        "template_params": {},
        "launch_params": {},
        "features": {},
        "status": "benchmarked",
        "notes": {},
    }
    shape_a = {"rows": 1, "D": 1536, "N": 1536}
    shape_b = {"rows": 2, "D": 1536, "N": 3072}
    cand_a = KernelCandidate(
        candidate_id="cand_a",
        metal_source="kernel_a",
        func_name="kk_a",
        metrics={
            "dtype": "float16",
            "latency_us": 40.0,
            "per_shape": [
                {"shape": shape_a, "latency_us": 10.0},
                {"shape": shape_b, "latency_us": 80.0},
            ],
        },
        **shared,
    )
    cand_b = KernelCandidate(
        candidate_id="cand_b",
        metal_source="kernel_b",
        func_name="kk_b",
        metrics={
            "dtype": "float16",
            "latency_us": 35.0,
            "per_shape": [
                {"shape": shape_a, "latency_us": 20.0},
                {"shape": shape_b, "latency_us": 30.0},
            ],
        },
        **shared,
    )
    payload = best_kernels_payload(
        [cand_a, cand_b],
        runtime_env={
            "mlx_version": "0.30.6",
            "device_arch": "applegpu_g16s",
            "device_name": "Apple M4 Max",
        },
    )
    by_shape = {
        json.dumps(entry["key"]["shape_signature"], sort_keys=True): entry for entry in payload["entries"]
    }
    assert by_shape[json.dumps(shape_a, sort_keys=True)]["candidate_id"] == "cand_a"
    assert by_shape[json.dumps(shape_b, sort_keys=True)]["candidate_id"] == "cand_b"
