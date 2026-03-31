# Index options

TurboQuant index behavior is configured through reloptions and session-level GUCs.

## Core reloptions

### `bits`

Quantization bit width. The v2 Qprod codec uses `b - 1` bits for stage-1 scalar codes plus 1 bit for the residual QJL sketch, so `bits = 4` stores 3-bit scalar codes + 1-bit QJL per subvector.

### `lists`

- `0`: flat mode (scan all batch pages)
- `> 0`: IVF mode (K-means routed)

### `transform`

Structured transform metadata. Currently only `'hadamard'` is supported. The transform is persisted as compact metadata (kind + seed), not a dense matrix.

### `normalized`

Declares whether vectors are pre-normalized. When `true`, cosine and inner-product queries use the faithful code-domain fast path. When `false`, those metrics fall back to decode-score mode.

## Query GUCs

### `turboquant.probes`

Number of IVF lists to probe per query. Higher values improve recall at the cost of more page reads. Ignored in flat mode.

### `turboquant.oversample_factor`

Candidate retention multiplier relative to the requested LIMIT. Controls how many approximate candidates the index scan returns before SQL-side reranking.

### `turboquant.max_visited_codes`

IVF code-visit budget. The scan stops adding lower-ranked lists once the cumulative live-count across selected lists reaches this threshold. Set to `0` to disable.

### `turboquant.max_visited_pages`

IVF page-visit budget. Parallel to `max_visited_codes` but counted in pages rather than codes. Set to `0` to disable.

### `turboquant.enable_summary_bounds`

Use persisted per-page summary structures for safe page pruning in IVF mode. Default: `true`.

### `turboquant.decode_rescore_factor`

Internal decode-rescore expansion factor for boundary-band candidates. Must be `>= 1`.

### `turboquant.decode_rescore_extra_candidates`

SQL rerank boundary-band expansion. `-1` for auto, `0` to disable, `> 0` for explicit extra candidate count.

### `turboquant.shadow_decode_diagnostics`

When enabled, keeps a parallel decode-scored candidate heap for diagnostic comparison against the code-domain heap. Development/debugging only.

### `turboquant.force_decode_score_diagnostics`

When enabled, uses decode-scored ranking instead of code-domain even on code-domain-capable scans. Development/debugging only.

## Important semantics

- lane count is derived from the real page budget, not a fixed batch assumption
- exact reranking stays outside the access method
- flat mode uses `lists = 0`; IVF mode uses immutable routing after build
- normalized cosine/IP use the code-domain fast path; L2 and non-normalized use decode-score fallback
- older indexes without per-vector `gamma` must be rebuilt with `REINDEX`

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
