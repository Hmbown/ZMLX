"""CLI for ZMLX Discover: LLM-guided kernel search.

Usage:
    python -m zmlx.discover search <target> [options]
    python -m zmlx.discover autorun [options]
    python -m zmlx.discover benchmark <session_dir> [options]
    python -m zmlx.discover compare <session_dir>
    python -m zmlx.discover report <session_path>
    python -m zmlx.discover list
    python -m zmlx.discover export <session_path> --output <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def _run_search(args: argparse.Namespace) -> None:
    """Execute the main search loop."""
    from .candidates import KernelCandidate, KernelSpec
    from .evaluate import evaluate_candidate
    from .llm import get_backend
    from .prompts import SYSTEM_PROMPT, build_generation_prompt, build_history_summary
    from .report import print_report
    from .session import Session
    from .targets import TARGETS
    from .tree import SearchTree

    target_name = args.target
    if target_name not in TARGETS:
        print(f"Unknown target: {target_name!r}")
        print(f"Available targets: {', '.join(TARGETS.keys())}")
        sys.exit(1)

    # Build target search space
    target_kwargs: dict[str, Any] = {}
    if args.D is not None:
        target_kwargs["D"] = args.D
    if args.K is not None:
        target_kwargs["K"] = args.K
    search_space = TARGETS[target_name](**target_kwargs)

    # LLM backend
    backend = get_backend(args.llm, model=args.model)

    # Device info
    device_info_str = "Unknown device"
    device_dict: dict[str, Any] = {"chip": "unknown", "memory_gb": 0}
    try:
        from ..device import detect_device

        dev = detect_device()
        device_info_str = f"{dev.full_name} ({dev.gpu_cores} GPU cores, {dev.memory_gb}GB)"
        device_dict = {"chip": dev.full_name, "memory_gb": dev.memory_gb}
    except Exception:
        pass

    # Session
    session_dir = Path(args.session_dir)
    if args.resume:
        session_path = session_dir / f"{target_name}_session.json"
        if session_path.exists():
            session = Session.load(session_path)
            print(f"Resumed session {session.metadata.session_id}")
        else:
            print(f"No session to resume at {session_path}, starting new.")
            session = Session.new(target_name, args.llm, device_dict)
    else:
        session = Session.new(target_name, args.llm, device_dict)

    # Seed candidate
    if search_space.seed_candidates:
        root_candidate = search_space.seed_candidates[0]
    else:
        root_candidate = KernelCandidate(
            spec=KernelSpec(
                name=f"kk_{target_name}_seed",
                input_names=search_space.input_names,
                output_names=search_space.output_names,
                source=search_space.reference_source,
                header=search_space.header,
            ),
            generation=0,
            llm_reasoning="seed",
        )

    # Restore or create tree
    if session.tree_data:
        tree = SearchTree.from_dict(session.tree_data)
    else:
        tree = SearchTree(root_candidate, c_puct=args.c_puct)

    # Reference function and test inputs
    reference_fn = search_space.make_reference_fn()

    try:
        from .._compat import import_mx

        mx = import_mx()
    except ImportError:
        print("MLX not available. Cannot run search.")
        sys.exit(1)

    mx.random.seed(42)

    # Build test inputs from first concrete shape
    test_inputs = []
    for ispec in search_space.input_specs:
        shape = ispec.concrete_shapes[0] if ispec.concrete_shapes else (16,)
        if ispec.dtype == "uint32":
            inp = mx.zeros(shape, dtype=mx.uint32)
        else:
            inp = mx.random.normal(shape).astype(mx.float32)
        test_inputs.append(inp)

    # Compute output shapes/dtypes for kernel launch
    output_shapes = []
    output_dtypes: list[Any] = []
    for ospec in search_space.output_specs:
        shape = ospec.concrete_shapes[0] if ospec.concrete_shapes else (16,)
        output_shapes.append(shape)
        if ospec.dtype == "uint32":
            output_dtypes.append(mx.uint32)
        else:
            output_dtypes.append(mx.float32)

    # Baseline timing
    from .evaluate import _time_fn

    def _ref_run(*inputs: Any) -> Any:
        return reference_fn(*inputs)

    baseline_timings = _time_fn(_ref_run, tuple(test_inputs), mx, warmup=args.warmup, iters=args.iters)
    baseline_timings.sort()
    baseline_us = baseline_timings[len(baseline_timings) // 2]
    session.metadata.baseline_us = baseline_us
    print(f"Baseline: {baseline_us:.1f} us (median of {args.iters} iters)")

    # Compute grid from target
    grid: tuple[int, int, int] | None = None
    threadgroup: tuple[int, int, int] | None = None
    if search_space.compute_grid is not None:
        grid, threadgroup = search_space.compute_grid(test_inputs)

    # Also evaluate the seed
    seed_result = evaluate_candidate(
        root_candidate,
        reference_fn,
        test_inputs,
        baseline_us=baseline_us,
        warmup=args.warmup,
        iters=args.iters,
        timeout_s=args.timeout,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup,
        template=[("T", mx.float32)],
    )
    tree.root.eval_result = seed_result
    tree.root.max_reward = seed_result.reward
    tree.root.visit_count = 1

    if args.verbose:
        status = "OK" if seed_result.correct else "FAIL"
        print(f"  Seed: [{status}] median={seed_result.median_us:.1f}us "
              f"reward={seed_result.reward:.2f} speedup={seed_result.speedup:.2f}x")

    # Main search loop
    for step in range(args.steps):
        t0 = time.time()
        print(f"\n--- Step {step + 1}/{args.steps} ---")

        # Select parent via PUCT
        parent_node = tree.select()
        parent_source = parent_node.candidate.spec.source
        parent_timing = parent_node.eval_result.median_us if parent_node.eval_result else baseline_us
        best_node = tree.best_node()
        best_er = best_node.eval_result
        best_timing = best_er.median_us if best_er else baseline_us

        # Build prompt
        history = build_history_summary(tree.evaluated_nodes)
        user_prompt = build_generation_prompt(
            search_space,
            parent_source,
            parent_timing,
            best_timing,
            history,
            device_info_str,
            n_candidates=args.candidates_per_step,
        )

        # LLM generate
        print(f"  Generating {args.candidates_per_step} candidates via {args.llm}...")
        llm_response = backend.generate_candidates(
            SYSTEM_PROMPT, user_prompt,
            n_candidates=args.candidates_per_step,
            temperature=0.8,
        )
        print(f"  Got {len(llm_response.candidates)} candidates ({llm_response.tokens_used} tokens)")
        session.metadata.total_candidates += len(llm_response.candidates)

        # Create candidates and expand tree
        new_candidates = []
        for cd in llm_response.candidates:
            spec = KernelSpec(
                name=f"kk_{target_name}_gen{step + 1}",
                input_names=search_space.input_names,
                output_names=search_space.output_names,
                source=cd["source"],
                header=search_space.header,
                threadgroup=tuple(cd.get("threadgroup", (256, 1, 1))),
                template_params=search_space.template_params,
            )
            cand = KernelCandidate(
                spec=spec,
                parent_id=parent_node.node_id,
                generation=step + 1,
                llm_reasoning=cd.get("reasoning", ""),
            )
            new_candidates.append(cand)

        new_nodes = tree.expand(parent_node, new_candidates)

        # Evaluate each new node
        step_best_speedup = 0.0
        for node in new_nodes:
            result = evaluate_candidate(
                node.candidate,
                reference_fn,
                test_inputs,
                baseline_us=baseline_us,
                warmup=args.warmup,
                iters=args.iters,
                timeout_s=args.timeout,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                grid=grid,
                threadgroup=threadgroup,
                template=[("T", mx.float32)],
            )
            node.eval_result = result
            tree.backpropagate(node, result.reward)
            session.metadata.total_evaluated += 1

            if result.speedup > step_best_speedup:
                step_best_speedup = result.speedup

            if args.verbose:
                status = "OK" if result.correct else "FAIL"
                if result.compiled and result.correct:
                    print(f"    [{status}] median={result.median_us:.1f}us "
                          f"reward={result.reward:.2f} speedup={result.speedup:.2f}x "
                          f"— {node.candidate.llm_reasoning[:50]}")
                else:
                    err = result.compile_error or result.correctness_error or "unknown"
                    print(f"    [{status}] {err[:60]}")

            # Store source for session persistence
            session.candidate_sources[node.node_id] = node.candidate.spec.source

        elapsed = time.time() - t0
        print(f"  Step {step + 1}: {len(new_nodes)} evaluated, "
              f"best speedup={step_best_speedup:.2f}x, "
              f"tree best={tree.best_node().max_reward:.2f} reward, "
              f"({elapsed:.1f}s)")

        # Update session
        session.metadata.total_steps = step + 1
        best = tree.best_node()
        if best.eval_result:
            session.metadata.best_reward = best.eval_result.reward
            session.metadata.best_speedup = best.eval_result.speedup
            session.metadata.best_source = best.candidate.spec.source

        # Auto-save
        session.tree_data = tree.to_dict()
        session_path = session_dir / f"{target_name}_session.json"
        session.save(session_path)

    # Final report
    print_report(session)
    print(f"Session saved to {session_dir / f'{target_name}_session.json'}")


def _run_autorun(args: argparse.Namespace) -> None:
    """Run Discover searches across all targets and LLM backends."""
    import json

    from .targets import TARGETS

    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Determine which targets and backends to run
    if args.targets:
        targets = [t for t in args.targets if t in TARGETS]
    else:
        targets = list(TARGETS.keys())

    backends = [b.strip() for b in args.backends.split(",")]

    print("ZMLX Discover Autorun")
    print(f"  Targets:  {', '.join(targets)}")
    print(f"  Backends: {', '.join(backends)}")
    print(f"  Steps:    {args.steps}")
    print(f"  Cands:    {args.candidates_per_step}/step")
    print(f"  Output:   {session_dir}")
    print()

    results: list[dict[str, Any]] = []

    for target_name in targets:
        for backend_name in backends:
            run_id = f"{target_name}_{backend_name}"
            run_dir = session_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            session_path = run_dir / f"{target_name}_session.json"

            # Skip if already completed (unless --force)
            if session_path.exists() and not args.force:
                from .session import Session

                existing = Session.load(session_path)
                if existing.metadata.total_steps >= args.steps:
                    print(f"[SKIP] {run_id} — already has {existing.metadata.total_steps} steps")
                    results.append({
                        "target": target_name,
                        "backend": backend_name,
                        "best_speedup": existing.metadata.best_speedup,
                        "best_reward": existing.metadata.best_reward,
                        "steps": existing.metadata.total_steps,
                        "evaluated": existing.metadata.total_evaluated,
                        "session_path": str(session_path),
                    })
                    continue

            print(f"\n{'=' * 60}")
            print(f"  {run_id} — {args.steps} steps × {args.candidates_per_step} candidates")
            print(f"{'=' * 60}")

            # Build args namespace for _run_search
            search_args = argparse.Namespace(
                target=target_name,
                steps=args.steps,
                candidates_per_step=args.candidates_per_step,
                llm=backend_name,
                model=args.model,
                resume=False,
                session_dir=str(run_dir),
                D=None,
                K=None,
                c_puct=args.c_puct,
                warmup=args.warmup,
                iters=args.iters,
                timeout=args.timeout,
                verbose=args.verbose,
            )

            try:
                _run_search(search_args)
            except Exception as e:
                print(f"  [ERROR] {run_id}: {e}")
                results.append({
                    "target": target_name,
                    "backend": backend_name,
                    "best_speedup": 0.0,
                    "best_reward": 0.0,
                    "error": str(e),
                })
                continue

            # Load results
            if session_path.exists():
                from .session import Session

                sess = Session.load(session_path)
                results.append({
                    "target": target_name,
                    "backend": backend_name,
                    "best_speedup": sess.metadata.best_speedup,
                    "best_reward": sess.metadata.best_reward,
                    "steps": sess.metadata.total_steps,
                    "evaluated": sess.metadata.total_evaluated,
                    "session_path": str(session_path),
                })

    # Summary
    print(f"\n{'=' * 70}")
    print("  AUTORUN SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Target':<25s}  {'Backend':<14s}  {'Speedup':>8s}  {'Reward':>7s}  {'Eval':>5s}")
    print(f"  {'-' * 25}  {'-' * 14}  {'-' * 8}  {'-' * 7}  {'-' * 5}")
    for r in results:
        if "error" in r:
            print(f"  {r['target']:<25s}  {r['backend']:<14s}  {'ERROR':>8s}")
        else:
            print(
                f"  {r['target']:<25s}  {r['backend']:<14s}  "
                f"{r['best_speedup']:7.2f}x  {r['best_reward']:7.2f}  "
                f"{r.get('evaluated', 0):5d}"
            )
    print(f"{'=' * 70}\n")

    # Save summary
    summary_path = session_dir / "autorun_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Summary saved to {summary_path}")


def _run_compare(args: argparse.Namespace) -> None:
    """Compare results across backends for each target."""
    import json

    session_dir = Path(args.session_dir)
    summary_path = session_dir / "autorun_summary.json"

    if summary_path.exists():
        results = json.loads(summary_path.read_text())
    else:
        # Scan session dirs
        results = []
        for run_dir in sorted(session_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for session_file in run_dir.glob("*_session.json"):
                from .session import Session

                sess = Session.load(session_file)
                results.append({
                    "target": sess.metadata.target_name,
                    "backend": sess.metadata.llm_backend,
                    "best_speedup": sess.metadata.best_speedup,
                    "best_reward": sess.metadata.best_reward,
                    "steps": sess.metadata.total_steps,
                    "evaluated": sess.metadata.total_evaluated,
                    "baseline_us": sess.metadata.baseline_us,
                    "session_path": str(session_file),
                })

    if not results:
        print("No sessions found.")
        return

    # Group by target
    by_target: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_target.setdefault(r["target"], []).append(r)

    print(f"\n{'=' * 70}")
    print("  BACKEND COMPARISON — Best Kernel Speedup per Target")
    print(f"{'=' * 70}")

    for target, runs in sorted(by_target.items()):
        print(f"\n  {target}:")
        runs_sorted = sorted(runs, key=lambda r: r.get("best_speedup", 0), reverse=True)
        for r in runs_sorted:
            sp = r.get("best_speedup", 0)
            marker = " <-- BEST" if r is runs_sorted[0] and sp > 1.0 else ""
            print(f"    {r['backend']:<14s}  {sp:7.2f}x{marker}")

    # Overall recommendation
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 70}")
    for target, runs in sorted(by_target.items()):
        best = max(runs, key=lambda r: r.get("best_speedup", 0))
        sp = best.get("best_speedup", 0)
        if sp > 1.05:
            print(f"  {target}: USE discovered kernel ({sp:.2f}x via {best['backend']})")
        elif sp > 1.0:
            print(f"  {target}: MARGINAL gain ({sp:.2f}x via {best['backend']}) — needs more search")
        else:
            print(f"  {target}: KEEP current ({sp:.2f}x best) — no improvement found")
    print()


def _run_report(args: argparse.Namespace) -> None:
    """Print a report from a saved session."""
    from .report import print_report
    from .session import Session

    session = Session.load(args.session_path)
    print_report(session, top_n=args.top)


def _run_list(_args: argparse.Namespace) -> None:
    """List available targets."""
    from .targets import TARGETS

    print("Available targets:")
    for name, fn in TARGETS.items():
        space = fn()
        print(f"  {name:20s}  {space.description[:60]}")


def _run_export(args: argparse.Namespace) -> None:
    """Export the best kernel from a session."""
    from .report import export_kernel
    from .session import Session

    session = Session.load(args.session_path)
    export_kernel(session, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zmlx.discover",
        description="LLM-guided Metal kernel search for ZMLX",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search
    sp = subparsers.add_parser("search", help="Run a kernel search")
    sp.add_argument("target", help="Target kernel name")
    sp.add_argument("--steps", type=int, default=10, help="Search steps")
    sp.add_argument("--candidates-per-step", type=int, default=8)
    sp.add_argument("--llm", choices=["claude", "claude-code", "openai", "mock"], default="mock")
    sp.add_argument("--model", help="Override LLM model name")
    sp.add_argument("--resume", action="store_true", help="Resume from session")
    sp.add_argument("--session-dir", default="discover_sessions")
    sp.add_argument("--D", type=int, default=None, help="Hidden dimension")
    sp.add_argument("--K", type=int, default=None, help="Top-K for MoE targets")
    sp.add_argument("--c-puct", type=float, default=1.0)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--iters", type=int, default=20)
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(func=_run_search)

    # autorun
    ap = subparsers.add_parser("autorun", help="Run all targets across all backends")
    ap.add_argument("--targets", nargs="*", help="Specific targets (default: all)")
    ap.add_argument("--backends", default="claude-code",
                    help="Comma-separated backends (default: claude-code)")
    ap.add_argument("--steps", type=int, default=10, help="Search steps per target")
    ap.add_argument("--candidates-per-step", type=int, default=4)
    ap.add_argument("--model", help="Override LLM model name")
    ap.add_argument("--session-dir", default="discover_sessions/autorun")
    ap.add_argument("--c-puct", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--force", action="store_true", help="Re-run even if session exists")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.set_defaults(func=_run_autorun)

    # compare
    cp = subparsers.add_parser("compare", help="Compare results across backends")
    cp.add_argument("session_dir", help="Autorun session directory")
    cp.set_defaults(func=_run_compare)

    # report
    rp = subparsers.add_parser("report", help="Print session report")
    rp.add_argument("session_path", help="Path to session JSON")
    rp.add_argument("--top", type=int, default=5)
    rp.set_defaults(func=_run_report)

    # list
    lp = subparsers.add_parser("list", help="List available targets")
    lp.set_defaults(func=_run_list)

    # export
    ep = subparsers.add_parser("export", help="Export best kernel")
    ep.add_argument("session_path", help="Path to session JSON")
    ep.add_argument("--output", "-o", required=True, help="Output file path")
    ep.set_defaults(func=_run_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
