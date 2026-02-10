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
from .model_shapes import derive_shape_suite_from_config, load_hf_config, parse_decode_rows
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
    config = load_hf_config(
        args.model_id,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
    )
    decode_rows = parse_decode_rows(args.decode_rows)
    op_names = [args.op] if args.op else list(_PHASE0_OPS)
    shapes: dict[str, list[dict[str, int]]] = {}
    errors: dict[str, str] = {}
    for op_name in op_names:
        try:
            shapes[op_name] = derive_shape_suite_from_config(
                op_name,
                config,
                decode_rows=decode_rows,
            )
        except Exception as exc:
            errors[op_name] = str(exc)

    payload: dict[str, Any] = {
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
            status=str(rec.get("status", "new")),
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
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--neighbors", type=int, default=4)
        p.add_argument("--knn", type=int, default=8)
        p.add_argument("--tours", type=int, default=3)
        p.add_argument("--exploit-fraction", type=float, default=0.25)
        p.add_argument("--novelty-fraction", type=float, default=0.25)
        p.add_argument("--min-tour-gap", type=int, default=3)

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

    p_suggest = sub.add_parser(
        "suggest-shapes",
        help="Infer model-aware shape suites from Hugging Face config.json",
    )
    p_suggest.add_argument("--model-id", required=True, help="Hugging Face model ID")
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
