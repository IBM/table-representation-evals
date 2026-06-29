# TEmBed - Tabular Embedding Test Bed

A comprehensive benchmark suite for evaluating tabular embeddings across four representation levels: **cell**, **row**, **column**, and **table**, using a diverse collection of tasks and datasets.

## Section 1: Benchmark Tasks

### 1) Tasks on Row Level

#### Row Similarity Search
**Description:** Given an input table and a row, find the most similar row from the input table to the given row.

**Approaches:** Implement the `row_embedding_component` to provide row embeddings, or implement `row_similarity_search_component` to return a ranked list directly. Set `run_similarity_search_based_on` in the approach's `supported_tasks` config accordingly.

#### Triplet-Based Evaluation (More Similar Than)
**Description:** Given a triplet of rows (anchor, positive, negative), evaluate whether the anchor is more similar to the positive than to the negative row.

**Approaches:** Implement the `row_embedding_component`.

#### Tabular Prediction
**Description:** Use row embeddings as features for downstream supervised tasks (classification or regression).

**Approaches:** Implement the `row_embedding_component` for embedding-based prediction, or implement `predictive_ml_component` to run the approach's own ML model.

### 2) Tasks on Column Level

#### Column Similarity Search
**Description:** Given a query column, retrieve and rank the most semantically similar columns from a data lake.

**Approaches:** Implement the `column_embedding_component`.

#### Column Type Annotation
**Description:** Given a table column, predict its semantic type from a fixed label vocabulary.

**Approaches:** Implement the `column_embedding_component`.

### 3) Tasks on Cell Level

#### Cell Level Semantic Retrieval
**Description:** Given a query cell, retrieve the top-k most semantically similar cells across a collection of tables.

**Approaches:** Implement the `cell_embedding_component`.

### 4) Tasks on Table Level

#### Table Retrieval
**Description:** Given a query table, retrieve and rank the most semantically similar tables from a collection.

**Approaches:** Implement the `table_embedding_component`.

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

2. Create `configs/approaches/<your_approach>.yaml` declaring `approach_name`, `module_path`, `class_name`, any hyperparameters, and a `supported_tasks` block listing which tasks your approach supports and their task-specific defaults:
   ```yaml
   approach_name: my_approach
   module_path: "approaches/benchmark_approaches_src/my_approach"
   class_name: MyApproach
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

Always set `PYTHONPATH` first (the shell scripts below do this automatically):
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

For a full reference on all configuration options, see [configs/README.md](configs/README.md).

### Run configs

Create or edit a file under `configs/runs/<name>.yaml`:
```yaml
benchmark_output_dir: my_results
approaches:
  - name: my_approach
    params:
      my_param: some_value
    tasks: [row_similarity_search, predictive_ml]   # optional whitelist; omit to run all supported
    task_datasets:                                   # optional: override the global dataset list
      row_similarity_search: [Amazon-Google]
```

The same approach can appear multiple times with different `params` — output directories are
automatically differentiated by a slug derived from the params (e.g.
`my_approach/embedding_model=all-MiniLM-L6-v2/row_similarity_search/Amazon-Google/`).

### Running

```bash
# Run all jobs defined in the run config
conda activate benchmark_env
python run_experiments.py <run_config_name>

# Stop on the first failure instead of continuing
python run_experiments.py <run_config_name> --stop-on-error

# Write results to a custom directory
python run_experiments.py <run_config_name> --results-dir results_testing
```

Results are written to `results/<benchmark_output_dir>/<approach>/<params>/<task>/<dataset>/results.json`. A run is skipped if `results.json` already exists — delete it to force a re-run.

### Smoke test (before committing)

```bash
bash run_test_before_commit.sh <conda_env_name>
```

Runs one small dataset per task type and fails on any error.

### Aggregating results

Results are automatically aggregated at the end of every `run_experiments.py` run. To re-run manually:
```bash
python benchmark_src/results_processing/gather_results.py results/<benchmark_output_dir>
```

---

## Section 5: Repository Structure

```
├── configs/
│   ├── global_datasets.yaml              # dataset registry per task
│   ├── approaches/
│   │   └── <approach>.yaml               # approach params + supported_tasks
│   └── runs/
│       └── <run>.yaml                    # what to run: approaches + param overrides
├── approaches/
│   └── benchmark_approaches_src/
│       └── <approach>/
│           ├── approach.py               # main approach class
│           └── <task>_component.py       # one file per supported capability
├── configs/
│   ├── approaches/                       # per-approach params + supported_tasks
│   ├── runs/                             # run configs (what to run + overrides)
│   ├── task/                             # task-level defaults (top_k, metrics, etc.)
│   ├── dataset/                          # dataset-specific settings for creation scripts
│   └── global_datasets.yaml             # canonical dataset list per task
├── benchmark_src/
│   ├── approach_interfaces/              # ABCs for all component types
│   ├── tasks/                            # one run_*_benchmark.py per task
│   ├── utils/                            # metrics, result aggregation, etc.
│   └── results_processing/              # gather_results.py, ranking, plots
├── run_experiments.py                    # main orchestrator entry point
├── run_test_before_commit.sh
└── run_paper_experiments.sh
```
