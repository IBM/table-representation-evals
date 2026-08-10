# Run Configuration Reference

Full reference for `configs/runs/<name>.yaml` — the file that defines which approaches to
evaluate and how to override their defaults. See [configs/README.md](../README.md) for the
minimal example and how this file fits together with `global_datasets.yaml` and
`configs/approaches/<name>.yaml`.

## Complete run configuration reference

```yaml
benchmark_output_dir: my_results
log_level: DEBUG          # optional; overrides the default INFO level for this run

# Run-level task whitelist: restrict all approaches to these tasks.
# Per-approach 'tasks' overrides this for a specific entry.
tasks:
  - row_similarity_search
  - predictive_ml

# Run-level task_params: a flat key applies to every task across all approaches; a key that
# matches a task name (any configs/task/<name>.yaml stem) scopes its dict value to just that
# task, across all approaches that run it. Lower priority than per-approach task_params.
task_params:
  max_queries: 100        # e.g. limit queries for a quick smoke-test run
  value_linking: # scoped to just this task, across all approaches
    force_embed_corpus: true

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
benchmark_output_dir: schema_and_value_linking_experiments
tasks:
  - schema_linking
  - value_linking

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
