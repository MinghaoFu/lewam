# datasets

Per-backend data adapters behind a common batch format, lazy-imported by `build_dataset`. Add one file per backend; a missing backend gives an "install the extra" message, not an import error.

Batch fields are keyed by the modality spec's `from` values (entry name if omitted), carry the sorted union of all windows reading them, and are float32. A new backend only needs `from` + `raw_index` (which defaults to `index`), never `via`/`encoder`/`dim`.
