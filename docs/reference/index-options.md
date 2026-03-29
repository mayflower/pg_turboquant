# Index options

TurboQuant index behavior is configured through reloptions and session-level tuning knobs.

## Core reloptions

### `bits`

Quantization bit width for the stored code.

### `lists`

- `0`: flat mode
- `> 0`: IVF mode

### `transform`

Current persistent contract is structured transform metadata, typically Hadamard-style.

### `normalized`

Declares whether vectors are pre-normalized for the intended metric.

## Query knobs

### `turboquant.probes`

Controls how many IVF lists are scanned in routed mode.

### `turboquant.oversample_factor`

Controls candidate retention pressure ahead of SQL-side reranking.

## Important semantics

- lane count is derived from real page budget, not a fixed large batch assumption
- exact reranking stays outside the access method
- flat mode uses `lists = 0`
- IVF mode uses immutable routing after build

## Example

```sql
CREATE INDEX docs_embedding_tq_idx
ON docs
USING turboquant (embedding tq_cosine_ops)
WITH (
  bits = 4,
  lists = 256,
  transform = 'hadamard',
  normalized = true
);
```
