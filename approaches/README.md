# Approaches

`benchmark_approaches_src` holds the individual tabular embedding approaches evaluated by
TEmBed. It is a separate installable package from `benchmark_src` (the benchmark framework)
so each approach can declare its own dependencies and conda environment without touching the
core framework — see the root [README.md](../README.md) for installation and how to run the
benchmark, and [configs/README.md](../configs/README.md) for the full run/approach config
reference.

---

## Plugin architecture

Approaches are **not** imported as normal Python modules by the framework. Every approach
class extends `BaseTabularEmbeddingApproach`
(`benchmark_src/approach_interfaces/base_interface.py`). At runtime, task code requests a
specific capability by calling `self._load_component(module_file, class_name, interface)`,
which:

1. Resolves `<approach's module_path>/<module_file>.py`,
2. Loads it directly off disk via `importlib.util.spec_from_file_location` — not a package
   import,
3. Instantiates the class and checks `isinstance(component, interface)`, raising
   `NotImplementedError` / `TypeError` if the file is missing or doesn't satisfy the interface.

This means an approach only needs the component files for the capabilities it actually
implements — task code probes for a component and treats the capability as unsupported if the
file is absent. Each component type has a matching ABC in `benchmark_src/approach_interfaces/`:

| Component file | Interface | Used for |
|---|---|---|
| `cell_embedding_component.py` | `CellEmbeddingInterface` | cell-level tasks (cell similarity search, value linking) |
| `row_embedding_component.py` | `RowEmbeddingInterface` | row-level tasks (row similarity search, triplet evaluation, predictive ML) |
| `column_embedding_component.py` | `ColumnEmbeddingInterface` | column-level tasks (column similarity search, column type annotation, schema linking) |
| `table_embedding_component.py` | `TableEmbeddingInterface` | table-level tasks (table retrieval, table similarity search, table shuffling, table type detection) |
| `predictive_ml_component.py` | `PredictiveMLInterface` | approaches that train directly on the input table + labels instead of producing row embeddings |
| `row_similarity_search_component.py` | `RowSimilaritySearchInterface` | approaches with custom row-similarity logic instead of cosine similarity over row embeddings (optional — none of the implemented approaches currently use this) |

---

## Implemented approaches

| Approach | conda env | Components | Model / package |
|---|---|---|---|
| `baseline` | `benchmark_env` | Pred-ML | XGBoost + skrub `TableVectorizer` on raw table features — non-embedding sanity floor for predictive ML |
| `GritLM` | `benchmark_env_gritlm` | Cell, Row, Column, Table | [`gritlm`](https://pypi.org/project/gritlm/) |
| `hashing` | `benchmark_env` | Table | non-learned hashing baseline, no external model |
| `hytrel` | `benchmark_env_hytrel` | Cell, Row, Column, Table | [HyTrel](https://github.com/awslabs/hypergraph-tabular-lm) |
| `mitra` | `benchmark_env` | Pred-ML | [Mitra](https://huggingface.co/autogluon/mitra-classifier) (AutoGluon) |
| `sap_rpt_oss` | `benchmark_env` | Cell, Row, Column, Pred-ML | [SAP RPT-1-OSS](https://github.com/SAP-samples/sap-rpt-1-oss) |
| `sentence_transformer` | `benchmark_env` | Cell, Row, Column, Table | [`sentence-transformers`](https://www.sbert.net/) (e.g. MiniLM, Granite) |
| `tabbie` | `benchmark_env` | Cell, Row, Column, Table | [TABBIE](https://github.com/SFIG611/tabbie) |
| `tabdpt` | `benchmark_env` | Row, Table, Pred-ML | [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference) |
| `tabert` | `benchmark_env_tabert` | Cell, Row, Column, Table | [TaBERT](https://github.com/facebookresearch/TaBERT) |
| `tabicl` | `benchmark_env_tabicl` | Cell, Row, Column, Pred-ML | [`tabicl`](https://pypi.org/project/tabicl/) |
| `tabpfn` | `benchmark_env` | Row, Pred-ML | [`tabpfn`](https://pypi.org/project/tabpfn/) |
| `tabula_8b` | `benchmark_env` | Row, Pred-ML | Llama-based model, prompted via [`tableshift`](https://github.com/mlfoundations/tableshift) |
| `tarte` | `benchmark_env` | Cell, Row, Column, Table | [TARTE](https://github.com/soda-inria/tarte-ai) (TMLR 2025, [arXiv:2505.14415](https://arxiv.org/abs/2505.14415)) |
| `tfidf` | `benchmark_env` | Table | non-learned TF-IDF baseline, no external model |
| `tuta` | `benchmark_env` | Cell, Row, Column, Table | [TUTA](https://github.com/microsoft/TUTA_table_understanding) |

Which tasks each approach is actually evaluated on (as opposed to which components it
implements) is declared per-approach in `configs/approaches/<name>.yaml` under
`supported_tasks` — see [configs/README.md](../configs/README.md).

---

## Directory layout

```
approaches/
├── pyproject.toml                        # package: benchmark_approaches_src, installed editable
└── benchmark_approaches_src/
    ├── <approach_name>/                  # template — copy this to add a new approach
    │   ├── approach.py                   # main approach class, extends BaseTabularEmbeddingApproach
    │   ├── approach_utils.py             # optional shared helpers (not a component)
    │   ├── cell_embedding_component.py
    │   ├── row_embedding_component.py
    │   ├── column_embedding_component.py
    │   ├── table_embedding_component.py
    │   ├── predictive_ml_component.py
    │   └── row_similarity_search_component.py
    ├── sentence_transformer/
    ├── GritLM/
    ├── ...                                # one folder per approach, same layout as the template
```

Some approaches also carry extra helper modules that are not components (e.g.
`sap_rpt_oss/torch_compatibility.py`, `tabbie/tabbie_model.py`) — these are imported by
`approach.py` or the components, not loaded by the framework directly.

---

## Adding a new approach

1. Copy `benchmark_approaches_src/<approach_name>/` (the template) to a new folder and rename
   the class in `approach.py`.
2. Delete the component files you don't implement; implement the rest to satisfy their
   interface (see the table above).
3. Create `configs/approaches/<name>.yaml` with `approach_name`, `module_path`, `class_name`,
   `conda_env`, hyperparameters, and a `supported_tasks` block — see
   [configs/README.md](../configs/README.md) for the full reference.
4. Add the approach to a run config under `configs/runs/` — it can be mixed with approaches
   from other conda envs in the same file; the orchestrator dispatches per-env subprocesses
   automatically.
5. Use `logging`, not `print` — the orchestrator captures it to `run.log` in the job's output
   directory.

If your approach needs extra Python dependencies, add them to
`benchmark_approaches_src/<name>/setup.sh` and/or `requirements.txt`, not the root
`reqs_benchmark.txt`, and wire the new setup step into `setup_benchmark.sh.template`.
