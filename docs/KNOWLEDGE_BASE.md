# Knowledge Base

`zmlx_knowledge_base.json` is a deterministic, source-derived artifact for retrieval and tooling.

It complements kernel discovery:
- `zmlx.kd`: discovers and benchmarks candidate kernels.
- `zmlx.kb`: normalizes model/pattern/kernel metadata into a validated JSON knowledge artifact.

## Rebuild

```bash
source .venv/bin/activate

python -m zmlx.kb build --output zmlx_knowledge_base.json --check-deterministic
python -m zmlx.kb validate --input zmlx_knowledge_base.json
python -m zmlx.kb report --input zmlx_knowledge_base.json
```

## Schema Guarantees

- Canonical schema version: `2.0`
- Deterministic serialization: sorted keys + stable entry ordering
- Unified definition shape for both pattern and discovered-kernel entries:
  - `name`, `description`, `math_formula`, `implementation_notes`,
    `source_path`, `source_symbol`, `constraints`, `env_flags`,
    `compatibility`, `evidence`
- Missing metadata is explicit:
  - Nullable fields use `null` with a required `missing_reason[field]`
- Strict validation checks:
  - required fields
  - duplicate IDs
  - broken model-to-definition references
  - metadata count drift (`model_count`, `definition_count`)

## Outputs

- Canonical KB JSON: `zmlx_knowledge_base.json`
- Schema dump (optional):

```bash
source .venv/bin/activate
python -m zmlx.kb schema --output docs/knowledge_base.schema.json
```
