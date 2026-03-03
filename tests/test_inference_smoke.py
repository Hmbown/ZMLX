from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "inference_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("inference_smoke", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PatchResult:
    def summary(self) -> str:
        return "Patched 1 modules:\n  swiglu_mlp: 1"


def test_inference_smoke_calls_patch_and_generate_with_cli_args():
    module = _load_module()

    model = SimpleNamespace()
    tokenizer = SimpleNamespace()
    calls: dict[str, object] = {}

    def fake_load(model_id: str):
        calls["load"] = model_id
        return model, tokenizer

    def fake_patch(patched_model):
        calls["patch_model"] = patched_model
        patched_model._zmlx_patch_result = _PatchResult()

    def fake_generate(patched_model, used_tokenizer, *, prompt: str, max_tokens: int):
        calls["generate"] = {
            "model": patched_model,
            "tokenizer": used_tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        return "generated text"

    args = module.parse_args(
        [
            "--model-id",
            "mlx-community/test-model",
            "--prompt",
            "hello world",
            "--max-tokens",
            "7",
        ]
    )

    stdout = StringIO()
    stderr = StringIO()
    rc = module.run(
        args,
        mlx_lm_module=SimpleNamespace(load=fake_load, generate=fake_generate),
        patch_fn=fake_patch,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 0
    assert calls["load"] == "mlx-community/test-model"
    assert calls["patch_model"] is model
    assert calls["generate"] == {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": "hello world",
        "max_tokens": 7,
    }

    out = stdout.getvalue()
    assert "[patch] Patched 1 modules:" in out
    assert "[patch]   swiglu_mlp: 1" in out
    assert "[output]" in out
    assert "generated text" in out
    assert stderr.getvalue() == ""


def test_inference_smoke_returns_error_when_load_fails():
    module = _load_module()
    patch_called = False

    def fake_patch(_model):
        nonlocal patch_called
        patch_called = True

    args = module.parse_args(["--model-id", "broken/model"])
    stdout = StringIO()
    stderr = StringIO()
    rc = module.run(
        args,
        mlx_lm_module=SimpleNamespace(
            load=lambda _model_id: (_ for _ in ()).throw(RuntimeError("cannot load")),
            generate=lambda *_a, **_kw: "unused",
        ),
        patch_fn=fake_patch,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 1
    assert patch_called is False
    assert "[error] Failed to load model 'broken/model'." in stderr.getvalue()


def test_inference_smoke_returns_error_when_generate_fails():
    module = _load_module()

    model = SimpleNamespace()
    tokenizer = SimpleNamespace()
    patch_called = False

    def fake_patch(_model):
        nonlocal patch_called
        patch_called = True

    args = module.parse_args(
        [
            "--model-id",
            "ok/model",
            "--prompt",
            "abc",
            "--max-tokens",
            "3",
        ]
    )
    stdout = StringIO()
    stderr = StringIO()
    rc = module.run(
        args,
        mlx_lm_module=SimpleNamespace(
            load=lambda _model_id: (model, tokenizer),
            generate=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("decode fail")),
        ),
        patch_fn=fake_patch,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 1
    assert patch_called is True
    assert "[error] Generation failed." in stderr.getvalue()


def test_inference_smoke_rejects_non_positive_max_tokens():
    module = _load_module()
    args = module.parse_args(["--model-id", "model", "--max-tokens", "0"])
    rc = module.run(
        args,
        mlx_lm_module=SimpleNamespace(load=lambda _: (object(), object()), generate=lambda *_a, **_kw: "x"),
        patch_fn=lambda _model: None,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert rc == 2
