"""Command-line interface for ZMLX training."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zmlx",
        description="ZMLX: Fine-tune MLX models with fused Metal kernels.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    train_p = sub.add_parser("train", help="Fine-tune a model")

    # Model
    train_p.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    train_p.add_argument("--tokenizer", type=str, default=None)

    # Data
    train_p.add_argument("--dataset", type=str, default="")
    train_p.add_argument("--data-format", type=str, default="auto")
    train_p.add_argument("--max-seq-length", type=int, default=2048)

    # LoRA
    train_p.add_argument("--lora", action="store_true")
    train_p.add_argument("--lora-rank", type=int, default=8)
    train_p.add_argument("--lora-alpha", type=float, default=16.0)
    train_p.add_argument("--lora-dropout", type=float, default=0.0)
    train_p.add_argument("--dora", action="store_true")

    # Quantization
    train_p.add_argument("--quantize", action="store_true")
    train_p.add_argument("--q-bits", type=int, default=4)

    # Optimization
    train_p.add_argument("--lr", "--learning-rate", type=float, default=1e-4, dest="learning_rate")
    train_p.add_argument("--optimizer", type=str, default="adam")
    train_p.add_argument("--warmup-steps", type=int, default=100)
    train_p.add_argument("--weight-decay", type=float, default=0.01)

    # Training
    train_p.add_argument("--iters", type=int, default=1000)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--grad-accum-steps", type=int, default=1)
    train_p.add_argument("--eval-interval", type=int, default=100)
    train_p.add_argument("--save-interval", type=int, default=500)
    train_p.add_argument("--output-dir", type=str, default="adapters")
    train_p.add_argument("--seed", type=int, default=42)

    # ZMLX
    train_p.add_argument("--no-patch", action="store_true", help="Disable ZMLX patching")
    train_p.add_argument("--no-fused-loss", action="store_true", help="Disable fused cross-entropy")
    train_p.add_argument("--patch-verbose", action="store_true")

    # Config file
    train_p.add_argument("--config", type=str, default=None, help="YAML config file")

    # Resume
    train_p.add_argument("--resume-from", type=str, default=None)

    # Misc
    train_p.add_argument("--verbose", action="store_true")

    # --- export ---
    export_p = sub.add_parser("export", help="Merge adapters and export model")
    export_p.add_argument("--model", type=str, required=True)
    export_p.add_argument("--adapter-path", type=str, required=True)
    export_p.add_argument("--output-path", type=str, default=None)
    export_p.add_argument("--verbose", action="store_true")

    # --- profile ---
    profile_p = sub.add_parser("profile", help="Profile a model or kernel")
    profile_p.add_argument("--model", type=str, help="Model path to profile")
    profile_p.add_argument("--analyze", action="store_true", help="Run bottleneck analysis")
    profile_p.add_argument("--iters", type=int, default=10, help="Number of iterations")
    profile_p.add_argument("--patch", action="store_true", help="Apply ZMLX patching before profiling")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``zmlx`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _run_train(args)
    elif args.command == "export":
        _run_export(args)
    elif args.command == "profile":
        _run_profile(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_profile(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from zmlx import load
    from zmlx.profile import analyze_bottlenecks, time_kernel

    if not args.model:
        print("Error: --model is required for profiling.")
        sys.exit(1)

    print(f"[zmlx] Loading model for profiling: {args.model}")
    model, tokenizer = load(args.model, patch=args.patch)

    # Dummy input
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    
    if args.analyze:
        print("[zmlx] Running bottleneck analysis...")
        analyze_bottlenecks(model, input_ids)
    else:
        print(f"[zmlx] Timing {args.iters} iterations...")
        stats = time_kernel(model, input_ids, iters=args.iters)
        print(f"Median time: {stats['median_us']/1000:.2f} ms")
        print(f"Mean time:   {stats['mean_us']/1000:.2f} ms")


def _run_train(args: argparse.Namespace) -> None:
    from .config import TrainConfig
    from .runner import train

    # Start from config file or defaults
    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    # Merge CLI args
    cli_dict = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "data_format": args.data_format,
        "max_seq_length": args.max_seq_length,
        "lora": args.lora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dora": args.dora,
        "quantize": args.quantize,
        "q_bits": args.q_bits,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "patch": not args.no_patch,
        "use_fused_loss": not args.no_fused_loss,
        "patch_verbose": args.patch_verbose,
        "resume_from": args.resume_from,
        "verbose": args.verbose,
    }
    config.merge_cli(cli_dict)

    print(f"[zmlx] Training {config.model}")
    print(f"[zmlx] LoRA: {config.lora}, Dataset: {config.dataset}")
    print(f"[zmlx] ZMLX patching: {config.patch}")

    result = train(config)
    print(f"[zmlx] Training complete. Output: {result['output_dir']}")


def _run_export(args: argparse.Namespace) -> None:
    from .export import merge_and_export

    output = merge_and_export(
        model_path=args.model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        verbose=args.verbose,
    )
    print(f"[zmlx] Exported to: {output}")


if __name__ == "__main__":
    main()
