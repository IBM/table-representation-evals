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
  more_similar_than: {}     # supported with no additional parameters
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

### Complete run configuration reference

```yaml
benchmark_output_dir: my_results
log_level: DEBUG          # optional; overrides the default INFO level for this run

# Run-level task whitelist: restrict all approaches to these tasks.
# Per-approach 'tasks' overrides this for a specific entry.
tasks:
  - row_similarity_search
  - predictive_ml

# Run-level task_params: applied to every task across all approaches.
# Lower priority than per-approach task_params.
task_params:
  max_queries: 100        # e.g. limit queries for a quick smoke-test run

approaches:
  - name: sentence_transformer
    params:                         # merged into approach config; drives output path slug
      embedding_model: all-MiniLM-L6-v2
      table_row_limit: 50

    tasks:                          # overrides the run-level task whitelist for this entry
      - row_similarity_search
      - predictive_ml

    task_datasets:                  # overrides the global dataset list for specific tasks
      row_similarity_search:
        - Amazon-Google
        - Beer

    task_params:                    # per-task parameter overrides (not reflected in output path)
      column_similarity_search:
        test_case_limit: 5

    task_exclude_datasets:          # additional dataset exclusions beyond approach-level ones
      predictive_ml:
        - kddcup09_appetency
```

### Override precedence

For task configuration fields (highest priority first):
per-approach `task_params` > `params` (when the key matches a task field) > run-level `task_params` > `supported_tasks` defaults

For task whitelist (highest priority first):
per-approach `tasks` > run-level `tasks` > all supported tasks

For dataset lists:
`task_datasets` (run-level override) > `global_datasets.yaml`

Dataset exclusions are cumulative: approach-level `exclude_datasets` and run-level
`task_exclude_datasets` are merged.

---

## Output directory structure

```
results/<benchmark_output_dir>/<approach_name>/[<param_slug>/]<task_name>/<dataset_name>/
```

`<param_slug>` is a sorted, path-sanitised `key=value` string derived from the run-level
`params` block (e.g. `embedding_model=all-MiniLM-L6-v2,table_row_limit=50`). It is
omitted when no run-level parameters are specified, preserving a flat directory structure
for approaches without overrides. Two entries for the same approach with differing `params`
therefore produce distinct output paths without requiring manual labels.

---

## Common usage patterns

### Running all approaches on a specific set of tasks

Use the run-level `tasks` key to restrict every approach in the file to a task subset,
without repeating the whitelist on each entry:

```yaml
benchmark_output_dir: schema_linking_experiments
tasks:
  - nl2column_mapping
  - nl2cell2column_mapping

approaches:
  - name: sentence_transformer
    params:
      embedding_model: all-MiniLM-L6-v2
  - name: GritLM
    params:
      embedding_model: GritLM/GritLM-7B
  - name: hytrel
```

### Mixing approaches from different conda environments

Run configs do not need to be split by conda env. List all approaches freely regardless of
their `conda_env`; the orchestrator groups them by env and dispatches subprocesses
automatically:

```yaml
benchmark_output_dir: my_results
approaches:
  - name: sentence_transformer   # conda_env: benchmark_env
    params:
      embedding_model: all-MiniLM-L6-v2
  - name: GritLM                 # conda_env: benchmark_env_gritlm
    params:
      embedding_model: GritLM/GritLM-7B
  - name: hytrel                 # conda_env: benchmark_env_hytrel
```

```bash
bash run.sh my_results           # dispatches benchmark_env, benchmark_env_gritlm, benchmark_env_hytrel automatically
```

### Restricting a single entry while others run all tasks

Per-approach `tasks` overrides the run-level whitelist for that entry only:

```yaml
tasks: [row_similarity_search, predictive_ml]  # default for all approaches

approaches:
  - name: sentence_transformer
    params:
      embedding_model: all-MiniLM-L6-v2
    # inherits run-level tasks

  - name: tabula_8b
    tasks: [predictive_ml]       # this entry only runs predictive_ml
```

### Restricting evaluation to a subset of tasks (per-approach)

Individual tasks can also be commented out temporarily using standard YAML comments:

```yaml
approaches:
  - name: sentence_transformer
    params:
      embedding_model: all-MiniLM-L6-v2
    tasks:
      - row_similarity_search
      # - predictive_ml    ← temporarily disabled
```

### Evaluating on a dataset subset

Override the global dataset list for a specific task via `task_datasets`:

```yaml
task_datasets:
  row_similarity_search:
    - Amazon-Google
```

### Comparing hyperparameter configurations

Include multiple entries for the same approach with differing `params`. Each entry
produces a distinct output path via the automatically derived parameter slug:

```yaml
approaches:
  - name: sentence_transformer
    params:
      embedding_model: all-MiniLM-L6-v2

  - name: sentence_transformer
    params:
      embedding_model: ibm-granite/granite-embedding-english-r2
```

### Evaluating an approach in multiple inference modes

Use `params` to differentiate modes (the value appears in the output slug) and `tasks`
to restrict which tasks each entry covers:

```yaml
approaches:
  - name: tabula_8b
    params:
      run_task_based_on: row_embeddings      # default from approach config; can be omitted

  - name: tabula_8b
    params:
      run_task_based_on: custom_predictiveML_model
    tasks: [predictive_ml]
    task_datasets:
      predictive_ml: [healthcare_insurance_expenses, wine_quality]
```

### Controlling logging verbosity

Set `log_level` in the run configuration to apply a default for all users of that file:

```yaml
log_level: DEBUG
```

Override on the command line for a one-off change (takes precedence over the run config):

```bash
bash run.sh my_run --log-level DEBUG
```

### Registering a new approach

1. Create `configs/approaches/<name>.yaml` with `approach_name`, `module_path`,
   `class_name`, `conda_env`, hyperparameters, and a `supported_tasks` block.
2. Add the approach entry to any run configuration under `approaches:` — it can be mixed
   freely with approaches from other conda environments.

An approach is only required to implement component files for the tasks declared in its
`supported_tasks` block. Refer to `benchmark_src/approach_interfaces/` for the
corresponding abstract interfaces.
