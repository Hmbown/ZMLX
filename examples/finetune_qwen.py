#!/usr/bin/env python3
"""End-to-end Qwen3 fine-tuning demo with ZMLX kernel acceleration.

Demonstrates the full workflow:
1. Load Qwen3-1.7B-4bit with ZMLX kernel patching
2. Apply LoRA adapters
3. Generate text BEFORE training (baseline)
4. Train 100 iterations with fused cross-entropy loss on WikiSQL
5. Generate text AFTER training (show improvement)

Usage:
    python examples/finetune_qwen.py
"""

from __future__ import annotations

import zmlx


def main() -> None:
    model_name = "mlx-community/Qwen3-1.7B-4bit"
    dataset_name = "mlx-community/WikiSQL"
    prompt = "Translate the following to SQL: How many people live in New York?"

    # 1. Load model with ZMLX patching
    print("=" * 60)
    print("Step 1: Loading model with ZMLX patches")
    print("=" * 60)
    model, tokenizer = zmlx.load(
        model_name,
        patch=True,
        dtype="float16",
        verbose=True,
    )

    # 2. Apply LoRA adapters
    print("\n" + "=" * 60)
    print("Step 2: Applying LoRA adapters")
    print("=" * 60)
    model = zmlx.lora(
        model,
        r=8,
        alpha=16.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    from mlx.utils import tree_flatten

    total_params = sum(p.size for _, p in tree_flatten(model.parameters())) / 1e6
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters())) / 1e6
    print(f"Total parameters:     {total_params:.1f}M")
    print(f"Trainable parameters: {trainable_params:.1f}M "
          f"({trainable_params / total_params * 100:.2f}%)")

    # 3. Generate BEFORE training
    print("\n" + "=" * 60)
    print("Step 3: Generate BEFORE training")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    before = zmlx.generate(model, tokenizer, prompt, max_tokens=100, temp=0.3)
    print(f"Output: {before}")

    # 4. Train
    print("\n" + "=" * 60)
    print("Step 4: Training (100 iters, fused loss)")
    print("=" * 60)
    stats = zmlx.train(
        model,
        tokenizer,
        dataset_name,
        iters=100,
        batch_size=2,
        learning_rate=2e-4,
        max_seq_length=512,
        output_dir="adapters/qwen3_demo",
        use_fused_loss=True,
        eval_interval=50,
        log_interval=10,
        save_interval=100,
    )
    print(f"\nTraining complete: {stats}")

    # 5. Generate AFTER training
    print("\n" + "=" * 60)
    print("Step 5: Generate AFTER training")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    after = zmlx.generate(model, tokenizer, prompt, max_tokens=100, temp=0.3)
    print(f"Output: {after}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
