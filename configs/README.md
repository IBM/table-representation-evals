# Configuration Reference

All benchmark configuration resides in this directory. The framework composes three
configuration files at runtime to fully specify each benchmark job:

```
configs/global_datasets.yaml         — registry of datasets per task
configs/approaches/<name>.yaml       — approach parameters and task capabilities
configs/runs/<name>.yaml             — experiment specification and parameter overrides
```

Two supporting directories contain configuration that is rarely modified:

```
configs/task/<task_name>.yaml        — task-level evaluation defaults (e.g. top_k, metric names)
configs/dataset/<dataset_name>.yaml  — dataset-specific settings used by dataset creation scripts
```

Task defaults are loaded automatically by the orchestrator and only require modification
when introducing a new task or adjusting its evaluation protocol.

---

## File 1 — `global_datasets.yaml`

Defines the canonical dataset list for each task. Each task has a corresponding
`<task_name>_datasets` key. Commenting out an entry disables that dataset across all
approaches and run configurations.

```yaml
row_similarity_search_datasets:
  - Amazon-Google
  # - Beer        ← disabled globally for all runs
  - DBLP-ACM
```

---

## File 2 — `configs/approaches/<name>.yaml`

Specifies an approach's implementation entry point, hyperparameters, and the set of tasks
it supports. The `supported_tasks` block declares task capability and provides
approach-specific parameter defaults per task.

```yaml
approach_name: sentence_transformer
module_path: "approaches/benchmark_approaches_src/sentence_transformer"
class_name: SentenceTransformerEmbedder
conda_env: benchmark_env  # conda environment that has this approach's dependencies

embedding_model: ~        # required — must be provided in the run configuration
table_row_limit: 100

supported_tasks:
  row_similarity_search:
    run_similarity_search_based_on: row_embeddings
  predictive_ml:
    run_task_based_on: row_embeddings
    exclude_datasets:       # excluded for this approach regardless of run configuration
      - some_large_dataset  # e.g. exceeds available GPU memory
  column_similarity_search:
    run_similarity_search_based_on: column_embeddings
  row_triplet_evaluation: {}     # supported with no additional parameters
  clustering: {}
```

An approach is evaluated only on tasks listed under `supported_tasks`; unlisted tasks are
skipped without error.

`conda_env` tells the orchestrator which environment to activate when running this approach.
Approaches with different `conda_env` values can be freely mixed in a single run config —
the orchestrator dispatches them as separate subprocesses automatically (see File 3).

---

## File 3 — `configs/runs/<name>.yaml`

Defines which approaches to evaluate and how to override their defaults. The run
configuration name is passed to the orchestrator via `run.sh` (recommended) or directly:

```bash
bash run.sh <run_config_name>          # works from any conda env
python run_experiments.py <run_config_name>  # must be in benchmark_env
```

### Minimal run configuration

```yaml
benchmark_output_dir: my_results
approaches:
  - name: sentence_transformer
    params:
      embedding_model: all-MiniLM-L6-v2
```

Fields under `params` are merged into the approach configuration. Parameters marked `~`
(required) in the approach YAML must be supplied here. The `params` block also determines
the output path slug, ensuring that two entries for the same approach with different
parameters produce distinct output directories automatically.

For the full field reference, override precedence, output directory layout, and common
usage patterns (dataset subsets, mixing conda envs, hyperparameter comparisons, etc.), see
[configs/runs/README.md](runs/README.md).

---

## Registering a new approach

1. Create `configs/approaches/<name>.yaml` with `approach_name`, `module_path`,
   `class_name`, `conda_env`, hyperparameters, and a `supported_tasks` block.
2. Add the approach entry to any run configuration under `approaches:` — it can be mixed
   freely with approaches from other conda environments.

An approach is only required to implement component files for the tasks declared in its
`supported_tasks` block. Refer to `benchmark_src/approach_interfaces/` for the
corresponding abstract interfaces.
