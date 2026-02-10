"""CLI for Hamiltonian-guided MLX Metal kernel discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .archive import RunArchive
from .env import runtime_fingerprint
from .eval import evaluate_candidate
from .features import candidate_vector, collect_feature_keys
from .graph import build_knn_graph
from .model_shapes import (
    derive_shape_suite_from_config,
    derive_shape_suite_from_log,
    load_hf_config,
    parse_decode_rows,
)
from .mutations import initial_population, neighbor_mutations
from .ops import OPS, get_op
from .report import (
    load_ndjson,
    merge_best_payloads,
    write_best_kernels,
    write_markdown_report,
)
from .tour import build_union_tours, schedule_batch
from .types import KernelCandidate

_PHASE0_OPS = ("rmsnorm_residual", "rope", "swiglu")
_OP_CHOICES = [*(_PHASE0_OPS), *(op for op in OPS if op not in _PHASE0_OPS)]


def _shape_suite(op_module: Any, suite_name: str | None) -> tuple[str, list[dict[str, Any]]]:
    suite = suite_name or op_module.DEFAULT_SHAPE_SUITE
    if suite not in op_module.SHAPE_SUITES:
        raise ValueError(
            f"Unknown shape suite {suite!r} for {op_module.OP_NAME}. "
            f"Expected one of: {', '.join(sorted(op_module.SHAPE_SUITES))}"
        )
    return suite, [op_module.normalize_shape(s) for s in op_module.SHAPE_SUITES[suite]]


def _shape_suite_from_model(args: argparse.Namespace, op_module: Any) -> tuple[str, list[dict[str, Any]]]:
    if not args.model_id:
        raise ValueError("--model-id is required for model-derived shape selection")

    config = load_hf_config(
        args.model_id,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
    )
    decode_rows = parse_decode_rows(args.decode_rows)
    shapes = derive_shape_suite_from_config(op_module.OP_NAME, config, decode_rows=decode_rows)
    return f"hf:{args.model_id}", [op_module.normalize_shape(s) for s in shapes]


def _shape_suite_from_log(args: argparse.Namespace, op_module: Any) -> tuple[str, list[dict[str, Any]]]:
    if not args.shape_log:
        raise ValueError("--shape-log is required for log-derived shapes")
    shapes = derive_shape_suite_from_log(
        op_module.OP_NAME,
        args.shape_log,
        dtype_name=args.dtype,
        max_shapes=int(args.log_max_shapes),
        min_count=int(args.log_min_count),
    )
    label = f"log:{Path(args.shape_log).name}"
    return label, [op_module.normalize_shape(s) for s in shapes]


def _discrete_search_space_size(op_module: Any) -> int | None:
    total = 1
    for attr in ("TEMPLATE_PARAM_SPACE", "LAUNCH_PARAM_SPACE"):
        space = getattr(op_module, attr, None)
        if not isinstance(space, dict):
            return None
        for values in space.values():
            count = len(tuple(values))
            if count <= 0:
                return None
            total *= count
    return total


def _run_search(args: argparse.Namespace) -> None:
    op_module = get_op(args.op)
    if args.shape_log:
        shape_suite_name, shape_suite = _shape_suite_from_log(args, op_module)
    else:
        if args.shapes == "auto" and not args.model_id:
            raise ValueError("--shapes auto requires --model-id")
        use_model_shapes = bool(args.model_id) and (args.shapes in {None, "auto"})
        if use_model_shapes:
            shape_suite_name, shape_suite = _shape_suite_from_model(args, op_module)
        else:
            shape_suite_name, shape_suite = _shape_suite(op_module, args.shapes)
    search_space_size = _discrete_search_space_size(op_module)
    effective_budget = min(args.budget, search_space_size) if search_space_size is not None else args.budget

    out_dir = Path(args.out)
    runtime_env = runtime_fingerprint()
    archive = RunArchive(
        out_dir=out_dir,
        op_name=args.op,
        seed=args.seed,
        budget=effective_budget,
        dtype_name=args.dtype,
        shape_suite=shape_suite_name,
        runtime_env=runtime_env,
    )

    compile_shape = shape_suite[0]
    initial_count = min(max(args.batch * 2, 8), effective_budget)
    population = initial_population(
        op_module=op_module,
        shape=compile_shape,
        dtype_name=args.dtype,
        seed=args.seed,
        count=initial_count,
    )

    for cand in population:
        archive.register_candidate(cand)

    baseline_cache: dict[tuple[str, str], float] = {}
    evaluated = 0
    step = 0

    while evaluated < effective_budget:
        pool = archive.all_candidates()
        if not pool:
            break

        keys = collect_feature_keys(pool)
        vectors = [candidate_vector(c, keys) for c in pool]
        graph = build_knn_graph(vectors, k=max(1, min(args.knn, len(pool) - 1)))
        tours = build_union_tours(
            graph,
            len(pool),
            n_tours=max(1, args.tours),
            seed=args.seed + step * 101,
        )

        selected_indices = schedule_batch(
            candidates=pool,
            vectors=vectors,
            graph=graph,
            tours=tours,
            batch_size=min(args.batch, effective_budget - evaluated),
            step=step,
            seed=args.seed,
            exploit_fraction=args.exploit_fraction,
            novelty_fraction=args.novelty_fraction,
            min_tour_gap=args.min_tour_gap,
        )

        if not selected_indices:
            break

        progress = False
        for idx in selected_indices:
            if idx >= len(pool):
                continue
            candidate = pool[idx]
            if candidate.status != "new":
                continue

            updated = evaluate_candidate(
                candidate=candidate,
                op_module=op_module,
                dtype_name=args.dtype,
                shape_suite=shape_suite,
                seed=args.seed + step * 1009 + idx,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.bench_repeats,
                bench_mode=args.bench_mode,
                baseline_cache=baseline_cache,
            )
            archive.candidates[updated.candidate_id] = updated
            archive.log_evaluation(step=step, candidate=updated)
            evaluated += 1
            progress = True

            if updated.status in {"compiled", "correct", "benchmarked", "failed"}:
                neighbors = neighbor_mutations(
                    parent=updated,
                    op_module=op_module,
                    shape=compile_shape,
                    dtype_name=args.dtype,
                    seed=args.seed + step * 6151 + idx,
                    count=args.neighbors,
                )
                for nb in neighbors:
                    archive.register_candidate(nb)

            if evaluated >= effective_budget:
                break

        if not progress:
            break
        step += 1

    candidates = archive.all_candidates()
    report_path = out_dir / "report.md"
    write_markdown_report(
        out_path=report_path,
        op_name=args.op,
        seed=args.seed,
        budget=effective_budget,
        dtype_name=args.dtype,
        shape_suite=shape_suite_name,
        candidates=candidates,
    )

    best_path = out_dir / "best_kernels.json"
    write_best_kernels(best_path, candidates, runtime_env=runtime_env)

    print(f"Run complete: {evaluated} / {effective_budget} candidates evaluated")
    if effective_budget < args.budget:
        print(
            "Budget capped by finite parameter space: "
            f"requested={args.budget}, available={effective_budget}"
        )
    print(f"NDJSON log: {archive.ndjson_path}")
    print(f"Report: {report_path}")
    print(f"Best kernels: {best_path}")
    print(
        "Runtime key: "
        f"mlx={runtime_env.get('mlx_version', 'unknown')} "
        f"device={runtime_env.get('device_name', 'unknown')} "
        f"arch={runtime_env.get('device_arch', 'unknown')}"
    )


def _run_suggest_shapes(args: argparse.Namespace) -> None:
    op_names = [args.op] if args.op else list(_PHASE0_OPS)
    shapes: dict[str, list[dict[str, int]]] = {}
    errors: dict[str, str] = {}

    if args.shape_log:
        for op_name in op_names:
            try:
                shapes[op_name] = derive_shape_suite_from_log(
                    op_name,
                    args.shape_log,
                    max_shapes=int(args.log_max_shapes),
                    min_count=int(args.log_min_count),
                )
            except Exception as exc:
                errors[op_name] = str(exc)
        payload: dict[str, Any] = {
            "shape_log": str(args.shape_log),
            "shape_suites": shapes,
            "errors": errors,
        }
    else:
        if not args.model_id:
            raise ValueError("--model-id is required unless --shape-log is provided")
        config = load_hf_config(
            args.model_id,
            revision=args.revision,
            local_files_only=bool(args.local_files_only),
        )
        decode_rows = parse_decode_rows(args.decode_rows)
        for op_name in op_names:
            try:
                shapes[op_name] = derive_shape_suite_from_config(
                    op_name,
                    config,
                    decode_rows=decode_rows,
                )
            except Exception as exc:
                errors[op_name] = str(exc)

        payload = {
            "model_id": args.model_id,
            "revision": args.revision or "main",
            "model_type": str(config.get("model_type", "unknown")),
            "decode_rows": list(decode_rows),
            "shape_suites": shapes,
            "errors": errors,
        }

    out = Path(args.output) if args.output else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote shape suggestions to {out}")
    else:
        print(json.dumps(payload, indent=2))


def _load_candidates_from_records(records: list[dict[str, Any]]) -> list[KernelCandidate]:
    candidates: dict[str, KernelCandidate] = {}
    for rec in records:
        if rec.get("event") != "evaluation":
            continue
        cid = str(rec.get("candidate_id", ""))
        if not cid:
            continue

        candidate = KernelCandidate(
            op_name=str(rec.get("op_name", "unknown")),
            candidate_id=cid,
            metal_source=str(rec.get("metal_source", "")),
            func_name=str(rec.get("func_name", cid)),
            inputs_spec=list(rec.get("inputs_spec", [])),
            outputs_spec=list(rec.get("outputs_spec", [])),
            template_params=dict(rec.get("template_params", {})),
            launch_params=dict(rec.get("launch_params", {})),
            features=dict(rec.get("features", {})),
            status=rec.get("status", "new"),  # type: ignore[arg-type]
            metrics=dict(rec.get("metrics", {})),
            parent_id=rec.get("parent_id"),
            notes=dict(rec.get("notes", {})),
        )
        candidates[cid] = candidate
    return list(candidates.values())


def _run_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    meta_path = run_dir / "run_meta.json"
    ndjson_path = run_dir / "run.ndjson"

    if not meta_path.exists() or not ndjson_path.exists():
        raise FileNotFoundError(f"Missing run files under {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = load_ndjson(ndjson_path)
    candidates = _load_candidates_from_records(records)

    report_path = run_dir / "report.md"
    write_markdown_report(
        out_path=report_path,
        op_name=str(meta.get("op_name", "unknown")),
        seed=int(meta.get("seed", 0)),
        budget=int(meta.get("budget", 0)),
        dtype_name=str(meta.get("dtype", "float16")),
        shape_suite=str(meta.get("shape_suite", "default")),
        candidates=candidates,
    )

    best_path = run_dir / "best_kernels.json"
    write_best_kernels(best_path, candidates, runtime_env=dict(meta.get("runtime", {})))

    print(f"Report regenerated: {report_path}")
    print(f"Best kernels regenerated: {best_path}")


def _run_install(args: argparse.Namespace) -> None:
    run_file = Path(args.run)
    if run_file.is_dir():
        run_file = run_file / "best_kernels.json"
    if not run_file.exists():
        raise FileNotFoundError(f"Missing best kernels file: {run_file}")

    incoming = json.loads(run_file.read_text(encoding="utf-8"))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
    else:
        existing = {"schema_version": "2", "entries": []}

    payload = merge_best_payloads([existing, incoming])
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Installed discovered kernels to {target}")
    incoming_entries = incoming.get("entries", [])
    if not incoming_entries and "ops" in incoming:
        incoming_entries = [
            {"key": {"op_name": op_name}}
            for op_name in incoming.get("ops", {}).keys()
        ]
    updated_ops = sorted({str(entry.get("key", {}).get("op_name", "unknown")) for entry in incoming_entries})
    print(f"Updated ops: {', '.join(updated_ops) if updated_ops else '(none)'}")


def _run_pack(args: argparse.Namespace) -> None:
    in_paths: list[Path] = []
    for raw in args.runs:
        p = Path(raw)
        if p.is_dir():
            best = p / "best_kernels.json"
            if best.exists():
                in_paths.append(best)
            continue
        if p.exists():
            in_paths.append(p)

    payloads: list[dict[str, Any]] = []
    for path in in_paths:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    merged = merge_best_payloads(payloads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Packed {len(payloads)} runs into {out}")


def _summarize_decisions(decisions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        reason = str(getattr(decision, "reason", "unknown"))
        ok = bool(getattr(decision, "ok", False))
        if ok:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _run_promote(args: argparse.Namespace) -> None:
    if args.patterns is not None and args.patch_profile is not None:
        raise ValueError("Use either --patterns or --patch-profile, not both.")

    run_path = Path(args.run)
    run_dir = run_path
    ndjson_path = run_path
    meta_path = run_path / "run_meta.json"
    if run_path.is_dir():
        ndjson_path = run_path / "run.ndjson"
    elif run_path.name.endswith(".ndjson"):
        run_dir = run_path.parent
        meta_path = run_dir / "run_meta.json"
    else:
        raise FileNotFoundError(f"Expected run directory or run.ndjson, got {run_path}")

    if not ndjson_path.exists():
        raise FileNotFoundError(f"Missing run log: {ndjson_path}")

    records = load_ndjson(ndjson_path)
    candidates = _load_candidates_from_records(records)
    runtime_env: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runtime_env = dict(meta.get("runtime", {}))
        except Exception:
            runtime_env = {}

    from .promotion import PromotionPolicy, build_promotion_capsule, run_promotion_validation, select_promoted_entries

    policy = PromotionPolicy(
        min_speedup_p10=float(args.min_speedup_p10),
        noise_guard=float(args.noise_guard),
        max_noise_pct=float(args.max_noise_pct),
    )
    selection = select_promoted_entries(candidates, runtime_env=runtime_env, policy=policy)

    promoted_path = Path(args.promoted_out) if args.promoted_out else (run_dir / "promoted_kernels.json")
    promoted_path.parent.mkdir(parents=True, exist_ok=True)
    promoted_path.write_text(json.dumps(selection.payload, indent=2), encoding="utf-8")

    print(f"Promoted entries: {selection.promoted_count}")
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "promoted_entries": selection.promoted_count,
            "policy": {
                "min_speedup_p10": policy.min_speedup_p10,
                "noise_guard": policy.noise_guard,
                "max_noise_pct": policy.max_noise_pct,
            },
            "decisions": [d.__dict__ for d in selection.decisions],
            "rejection_summary": _summarize_decisions(selection.decisions),
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Promotion report: {report_path}")

    if selection.promoted_count == 0:
        rejection_summary = _summarize_decisions(selection.decisions)
        if rejection_summary:
            print("No promotable kernels (summary):")
            for reason, count in rejection_summary.items():
                print(f"  - {reason}: {count}")
        raise SystemExit(2)

    if args.skip_validate:
        print("Skipping end-to-end validation (--skip-validate).")
        return

    from zmlx.kv_cache import kv_cache_kwargs

    gen_kwargs = kv_cache_kwargs(
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )

    validation = run_promotion_validation(
        model_id=args.model,
        patterns=args.patterns,
        patch_profile=args.patch_profile,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        runs=args.runs,
        gen_kwargs=gen_kwargs,
        discovered_path=promoted_path,
    )

    fidelity_line = f"{validation.match_count}/{validation.total} tokens identical"
    verdict = "PASS" if validation.fidelity_ok else "FAIL"
    print(f"FIDELITY: {fidelity_line} [{verdict}]")
    print(
        f"MEDIAN DECODE: control={validation.control['median_gen_tps']:.2f} "
        f"discovered={validation.candidate['median_gen_tps']:.2f} "
        f"ratio={validation.median_gen_ratio:.3f}"
    )

    if args.capsule_out:
        capsule_path = Path(args.capsule_out)
        capsule_path.parent.mkdir(parents=True, exist_ok=True)
        capsule = build_promotion_capsule(
            model_id=args.model,
            patterns=args.patterns,
            patch_profile=args.patch_profile,
            max_tokens=args.max_tokens,
            runs=args.runs,
            prompt=args.prompt,
            discovered_path=promoted_path,
            validation=validation,
            note="Kernel discovery promotion validation",
        )
        capsule_path.write_text(json.dumps(capsule, indent=2), encoding="utf-8")
        print(f"Repro capsule: {capsule_path}")

    if not validation.fidelity_ok:
        raise SystemExit(3)
    if validation.median_gen_ratio < float(args.min_gen_tps_ratio):
        raise SystemExit(4)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
    else:
        existing = {"schema_version": "2", "entries": []}

    merged = merge_best_payloads([existing, selection.payload])
    target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Installed promoted kernels to {target}")


def _run_pipeline(args: argparse.Namespace) -> None:
    _run_search(args)
    args.run = args.out
    _run_promote(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zmlx-kernel-discover",
        description=(
            "Hamiltonian-guided MLX Metal kernel discovery "
            "(phase-0 focus: rmsnorm_residual, rope, swiglu)"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--op",
            required=True,
            choices=_OP_CHOICES,
            help="Target op (phase-0: rmsnorm_residual, rope, swiglu; baseline: rmsnorm)",
        )
        p.add_argument("--budget", type=int, default=50)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
        p.add_argument("--out", default="runs/kernel_discovery")
        p.add_argument("--shapes", default=None)
        p.add_argument(
            "--shape-log",
            default=None,
            help="JSONL runtime shape log (ZMLX_KD_SHAPE_LOG) to derive shape suite.",
        )
        p.add_argument(
            "--log-max-shapes",
            type=int,
            default=8,
            help="Max shapes to keep from runtime shape log (per op).",
        )
        p.add_argument(
            "--log-min-count",
            type=int,
            default=1,
            help="Minimum occurrence count to keep a shape from runtime log.",
        )
        p.add_argument(
            "--model-id",
            default=None,
            help=(
                "Hugging Face model ID for model-aware shape derivation from config.json. "
                "Used when --shapes is omitted or set to 'auto'."
            ),
        )
        p.add_argument("--revision", default=None, help="Optional Hugging Face revision for config lookup")
        p.add_argument(
            "--local-files-only",
            action="store_true",
            help="Resolve Hugging Face config from local cache only",
        )
        p.add_argument(
            "--decode-rows",
            default="1,2,4",
            help="Comma-separated decode row counts for model-derived shape suites",
        )
        p.add_argument("--warmup", type=int, default=3)
        p.add_argument("--iters", type=int, default=10)
        p.add_argument(
            "--bench-repeats",
            type=int,
            default=3,
            help="Repeat interleaved A/B timing batches per shape.",
        )
        p.add_argument(
            "--bench-mode",
            default="interleaved",
            choices=["interleaved", "legacy"],
            help="Benchmarking mode (interleaved A/B or legacy sequential).",
        )
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--neighbors", type=int, default=4)
        p.add_argument("--knn", type=int, default=8)
        p.add_argument("--tours", type=int, default=3)
        p.add_argument("--exploit-fraction", type=float, default=0.25)
        p.add_argument("--novelty-fraction", type=float, default=0.25)
        p.add_argument("--min-tour-gap", type=int, default=3)

    def add_promote_args(p: argparse.ArgumentParser, *, include_run: bool = True) -> None:
        if include_run:
            p.add_argument("--run", required=True, help="Path to run dir or run.ndjson")
        p.add_argument("--model", required=True, help="Hugging Face model ID for validation")
        p.add_argument("--patterns", nargs="+", default=None, help="Patch patterns to apply")
        p.add_argument("--patch-profile", type=str, default=None, help="Patch profile to apply")
        p.add_argument("--max-tokens", type=int, default=200, help="Tokens to generate")
        p.add_argument("--runs", type=int, default=3, help="Timed runs per config")
        p.add_argument(
            "--prompt",
            type=str,
            default=(
                "Explain the key differences between TCP and UDP protocols, "
                "including their use cases, reliability guarantees, and "
                "performance characteristics. Be thorough and precise."
            ),
        )
        p.add_argument("--kv-bits", type=int, default=None, help="Quantize KV cache to N bits")
        p.add_argument("--kv-group-size", type=int, default=None, help="Group size for KV cache quantization")
        p.add_argument("--quantized-kv-start", type=int, default=None, help="Step to begin quantized KV cache")
        p.add_argument(
            "--min-gen-tps-ratio",
            type=float,
            default=1.0,
            help="Minimum median decode tok/s ratio vs control to promote.",
        )
        p.add_argument(
            "--min-speedup-p10",
            type=float,
            default=1.01,
            help="Minimum per-shape speedup p10 required for promotion.",
        )
        p.add_argument(
            "--noise-guard",
            type=float,
            default=0.5,
            help="Scale factor for noise-aware promotion gating.",
        )
        p.add_argument(
            "--max-noise-pct",
            type=float,
            default=0.2,
            help="Maximum tolerated speedup noise percentage.",
        )
        p.add_argument(
            "--promoted-out",
            default=None,
            help="Optional path to write promoted kernel payload.",
        )
        p.add_argument(
            "--output",
            default="configs/discovered_kernels.json",
            help="Pinned kernel config to update on successful promotion.",
        )
        p.add_argument("--report-out", default=None, help="Optional JSON report output for promotion decisions.")
        p.add_argument("--capsule-out", default=None, help="Optional JSON repro capsule output path.")
        p.add_argument("--skip-validate", action="store_true", help="Skip end-to-end validation.")

    p_run = sub.add_parser("run", help="Run kernel discovery")
    add_common(p_run)
    p_run.set_defaults(func=_run_search)

    p_sweep = sub.add_parser("sweep", help="Alias for run with larger budget")
    add_common(p_sweep)
    p_sweep.set_defaults(func=_run_search)

    p_report = sub.add_parser("report", help="Regenerate report from a run directory")
    p_report.add_argument("--run", required=True)
    p_report.set_defaults(func=_run_report)

    p_install = sub.add_parser("install", help="Install best kernels into pinned config")
    p_install.add_argument("--run", required=True, help="Path to run dir or best_kernels.json")
    p_install.add_argument("--output", default="configs/discovered_kernels.json")
    p_install.set_defaults(func=_run_install)

    p_pack = sub.add_parser("pack", help="Merge run outputs into a device kernel pack")
    p_pack.add_argument("--runs", nargs="+", required=True, help="Run dirs or best_kernels.json files")
    p_pack.add_argument("--out", required=True, help="Output merged kernel-pack path")
    p_pack.set_defaults(func=_run_pack)

    p_promote = sub.add_parser("promote", help="Promote discovered kernels with validation gating")
    add_promote_args(p_promote, include_run=True)
    p_promote.set_defaults(func=_run_promote)

    p_pipeline = sub.add_parser("pipeline", help="Run discovery + validate + promote")
    add_common(p_pipeline)
    add_promote_args(p_pipeline, include_run=False)
    p_pipeline.set_defaults(func=_run_pipeline)

    p_suggest = sub.add_parser(
        "suggest-shapes",
        help="Infer model-aware shape suites from Hugging Face config.json",
    )
    p_suggest.add_argument("--model-id", required=False, help="Hugging Face model ID")
    p_suggest.add_argument(
        "--shape-log",
        default=None,
        help="JSONL runtime shape log (ZMLX_KD_SHAPE_LOG) to derive shape suite.",
    )
    p_suggest.add_argument(
        "--log-max-shapes",
        type=int,
        default=8,
        help="Max shapes to keep from runtime shape log (per op).",
    )
    p_suggest.add_argument(
        "--log-min-count",
        type=int,
        default=1,
        help="Minimum occurrence count to keep a shape from runtime log.",
    )
    p_suggest.add_argument("--revision", default=None, help="Optional Hugging Face revision")
    p_suggest.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve Hugging Face config from local cache only",
    )
    p_suggest.add_argument(
        "--decode-rows",
        default="1,2,4",
        help="Comma-separated decode row counts for suggested suites",
    )
    p_suggest.add_argument("--op", choices=_PHASE0_OPS, default=None, help="Limit to one phase-0 op")
    p_suggest.add_argument("--output", default=None, help="Optional JSON output path")
    p_suggest.set_defaults(func=_run_suggest_shapes)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
