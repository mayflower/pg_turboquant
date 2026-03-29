This directory contains the PostgreSQL extension implementation.

Key modules:

- `tq_am.c` and `tq_am_routine.*`: access-method entry points and routine wiring
- `tq_page.*`: page layout and batch-page helpers
- `tq_codec_mse.*` and `tq_codec_prod.*`: scalar TurboQuant-style codecs
- `tq_router.*`: IVF routing and training support
- `tq_scan.*`: scan-time candidate scoring and retrieval
- `tq_wal.*`: generic-WAL-localized write helpers
- `tq_pgvector_compat.*`: narrow compatibility layer for `vector` and `halfvec`

The implementation is structured so page layout, codec logic, pgvector compatibility, and WAL behavior stay separated instead of collapsing into one translation unit.
