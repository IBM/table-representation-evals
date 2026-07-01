# TEmBed - Tabular Embedding Test Bed

A comprehensive benchmark suite for evaluating tabular embeddings across four representation levels: **cell**, **row**, **column**, and **table**, using a diverse collection of tasks and datasets.

## Section 1: Benchmark Tasks

| Task | Level | Description | Interface | Metric |
|---|---|---|---|---|
| Row Similarity Search | Row | Find the most similar row in a table to a given query row | `row_embedding` or `row_similarity_search` | Top-1 [%] |
| Row Triplet Test | Row | Determine whether an anchor row is more similar to a positive than to a negative example | `row_embedding` | Accuracy |
| Tabular Prediction | Row | Use row embeddings as features for classification or regression | `row_embedding` or `predictive_ml` | ROC-AUC / RMSE |
| Column Similarity Search | Column | Retrieve semantically similar columns from a data lake | `column_embedding` | MRR |
| Column Type Annotation | Column | Predict the semantic type of a column from a fixed label vocabulary | `column_embedding` | Macro-F1 |
| NL→Column Mapping | Column | Match natural language query concepts to database columns | `column_embedding` | Recall@k |
| Cell Semantic Retrieval | Cell | Retrieve the most semantically similar cells across a table collection | `cell_embedding` | — |
| NL→Cell→Column Mapping | Cell | Map NL-extracted values to database cells and identify relevant columns (exact match) | `cell_embedding` | Recall@k |
| NL→Cell→Column Fuzzy Mapping | Cell | Map NL-extracted values to database cells and identify relevant columns (fuzzy/semantic match) | `cell_embedding` | Recall@k |
| Table Retrieval | Table | Retrieve semantically similar tables from a collection | `table_embedding` | Recall@1 |
| Table Shuffling Triplet Test | Table | Determine whether an anchor table is more similar to a positive than to a structurally varied negative | `table_embedding` | Triplet Accuracy |

---

## Section 2: Installation

1. Check out this repository (including submodules):
   ```bash
   git clone --recurse-submodules <repo-url>
   ```

2. Copy the setup template and edit the `SETUP_*` flags at the top to select which approaches to install:
   ```bash
   cp setup_benchmark.sh.template setup_benchmark.sh
   ```

3. Run the setup script:
   ```bash
   bash setup_benchmark.sh
   ```
   This creates the `benchmark_env` conda environment (Python 3.13) and installs `benchmark_src` and `approaches` as editable packages. If you enabled additional approaches, it also creates per-approach conda environments (`benchmark_env_hytrel`, `benchmark_env_gritlm`, `benchmark_env_tabicl`).

4. (Optional) Create a `.env` file at the repo root for approaches that require Hugging Face model access:
   ```bash
   HF_TOKEN=hf_...
   HF_HOME=/path/to/hf/cache
   ```
   Required for `sap_rpt_oss` and `tabpfn` after accepting their model license terms on Hugging Face.

---

## Section 3: How to Add Your Approach

1. Copy `approaches/benchmark_approaches_src/<approach_name>/` (the template folder) to a new folder and rename the class in `approach.py`.

2. Create `configs/approaches/<your_approach>.yaml` declaring `approach_name`, `module_path`, `class_name`, `conda_env`, any hyperparameters, and a `supported_tasks` block listing which tasks your approach supports and their task-specific defaults:
   ```yaml
   approach_name: my_approach
   module_path: "approaches/benchmark_approaches_src/my_approach"
   class_name: MyApproach
   conda_env: benchmark_env      # conda environment that has your approach's dependencies
   my_param: some_value

   supported_tasks:
     row_similarity_search:
       run_similarity_search_based_on: row_embeddings
     predictive_ml:
       run_task_based_on: row_embeddings
   ```

3. Implement only the component files for the capabilities you support (delete the rest). Each component must satisfy its interface in `benchmark_src/approach_interfaces/`.

4. Add your approach to a run config under `configs/runs/` (see Section 4).

5. Use `logging` instead of `print` — the orchestrator captures log output to `run.log` in each result directory.

---

## Section 4: How to Run the Benchmark

For a full reference on all configuration options, see [configs/README.md](configs/README.md).

### Run configs

Create or edit a file under `configs/runs/<name>.yaml`. You can freely mix approaches from
different conda environments in one file — the orchestrator handles env switching automatically.

```yaml
benchmark_output_dir: my_results

# Optional: restrict all approaches below to these tasks only
tasks: [row_similarity_search, predictive_ml]

approaches:
  - name: my_approach
    params:
      my_param: some_value
    tasks: [row_similarity_search]  # overrides the run-level tasks for this entry only
    task_datasets:                  # optional: override the global dataset list
      row_similarity_search: [Amazon-Google]
```

The same approach can appear multiple times with different `params` — output directories are
automatically differentiated by a slug derived from the params (e.g.
`my_approach/embedding_model=all-MiniLM-L6-v2/row_similarity_search/Amazon-Google/`).

### Running

```bash
# Run all jobs defined in the run config (works from any conda env)
bash run.sh <run_config_name>

# Stop on the first failure instead of continuing
bash run.sh <run_config_name> --stop-on-error

# Write results to a custom directory
bash run.sh <run_config_name> --results-dir results_testing
```

`run.sh` activates `benchmark_env` and sets `PYTHONPATH` automatically. When the run config
includes approaches from multiple conda envs (set via `conda_env` in each approach config),
the orchestrator dispatches them as subprocesses in sequence — no manual env switching needed.

Results are written to `results/<benchmark_output_dir>/<approach>/<params>/<task>/<dataset>/results.json`. A run is skipped if `results.json` already exists — delete it to force a re-run.

### Smoke test (before committing)

```bash
bash run_test_before_commit.sh <conda_env_name>
```

Runs one small dataset per task type and fails on any error.

### Aggregating results

Results are automatically aggregated at the end of every `run.sh` run. To re-run manually:
```bash
python benchmark_src/results_processing/gather_results.py results/<benchmark_output_dir>
```

---

## Section 5: Repository Structure

```
├── configs/
│   ├── global_datasets.yaml              # canonical dataset list per task
│   ├── approaches/
│   │   └── <approach>.yaml               # approach params, conda_env, supported_tasks
│   ├── runs/
│   │   └── <run>.yaml                    # what to run: approaches + param overrides
│   ├── task/                             # task-level defaults (top_k, metrics, etc.)
│   └── dataset/                          # dataset-specific settings for creation scripts
├── approaches/
│   └── benchmark_approaches_src/
│       └── <approach>/
│           ├── approach.py               # main approach class
│           └── <task>_component.py       # one file per supported capability
├── benchmark_src/
│   ├── approach_interfaces/              # ABCs for all component types
│   ├── tasks/                            # one run_*_benchmark.py per task
│   ├── utils/                            # metrics, result aggregation, etc.
│   └── results_processing/              # gather_results.py, ranking, plots
├── run.sh                                # entry point — activates benchmark_env, dispatches multi-env runs
├── run_experiments.py                    # orchestrator (called by run.sh)
├── run_test_before_commit.sh
└── run_paper_experiments.sh
```
