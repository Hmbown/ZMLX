"""Adapter export and model merging utilities."""

from __future__ import annotations

from pathlib import Path


def merge_and_export(
    model_path: str,
    adapter_path: str,
    output_path: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    """Merge LoRA adapters into the base model and export.

    Args:
        model_path: Path or HuggingFace ID of the base model.
        adapter_path: Path to the adapter weights directory.
        output_path: Where to save the merged model. Defaults to
            ``{adapter_path}/merged``.
        verbose: Print progress.

    Returns:
        Path to the exported merged model.
    """
    try:
        from mlx_lm import fuse as mlx_fuse
    except ImportError as e:
        raise ImportError(
            "mlx_lm is required for export. "
            "Install with: pip install 'zmlx[train]'"
        ) from e

    if output_path is None:
        output_path = str(Path(adapter_path) / "merged")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[zmlx.export] Merging {adapter_path} into {model_path}")
        print(f"[zmlx.export] Output: {output_path}")

    mlx_fuse.fuse(
        model=model_path,
        adapter_file=str(Path(adapter_path) / "adapters.safetensors"),
        save_path=output_path,
    )

    if verbose:
        print("[zmlx.export] Done!")

    return output_path
