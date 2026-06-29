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

---

## File 3 — `configs/runs/<name>.yaml`

Defines which approaches to evaluate and how to override their defaults. The run
configuration name is passed directly to the orchestrator:

```bash
python run_experiments.py <run_config_name>
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

approaches:
  - name: sentence_transformer
    params:                         # merged into approach config; drives output path slug
      embedding_model: all-MiniLM-L6-v2
      table_row_limit: 50

    tasks:                          # task whitelist — omit to run all supported tasks
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
`task_params` > `params` (when the key matches a task field) > `supported_tasks` defaults

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

### Restricting evaluation to a subset of tasks

Provide an explicit task whitelist under `tasks`. Individual tasks can also be commented
out temporarily using standard YAML comments:

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
python run_experiments.py my_run --log-level DEBUG
```

### Registering a new approach

1. Create `configs/approaches/<name>.yaml` with `approach_name`, `module_path`,
   `class_name`, hyperparameters, and a `supported_tasks` block.
2. Add the approach entry to the relevant run configuration under `approaches:`.

An approach is only required to implement component files for the tasks declared in its
`supported_tasks` block. Refer to `benchmark_src/approach_interfaces/` for the
corresponding abstract interfaces.
