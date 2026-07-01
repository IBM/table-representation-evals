"""
Benchmark orchestrator. Expands a run config into (approach, task, dataset) jobs and
runs them sequentially, then aggregates results.

Config layers:
  configs/global_datasets.yaml          — dataset registry per task
  configs/approaches/<name>.yaml        — approach params + supported_tasks block
  configs/runs/<name>.yaml              — what to run: approach entries with optional
                                          params / tasks / task_datasets / task_params /
                                          task_exclude_datasets overrides

Output layout:
  results/<benchmark_output_dir>/<approach>/[<param_slug>/]<task>/<dataset>/
  where <param_slug> is derived from run-level params overrides (omitted when empty).

Multi-env runs:
  When approaches in a run config declare different conda_env values (set in each
  configs/approaches/<name>.yaml), the orchestrator automatically dispatches one
  subprocess per env via `conda run -n <env>`. No manual env switching needed:

    python run_experiments.py schema_linking   # handles all envs automatically

Usage (recommended — works from any env):
    bash run.sh <run_config_name>
    bash run.sh <run_config_name> --results-dir results_testing
    bash run.sh <run_config_name> --stop-on-error

Direct usage (must be in benchmark_env):
    python run_experiments.py <run_config_name>
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from omegaconf import OmegaConf

from benchmark_src.utils.framework import StreamToLogger

logger = logging.getLogger(__name__)


def _load_yaml(path: Path):
    return OmegaConf.load(path)


def _collect_env_groups(run_cfg, configs_dir: Path) -> dict:
    """
    Read each approach config and group approach names by their conda_env.
    Returns {env_name: [approach_name, ...], None: [approach_name, ...]}
    where None means no conda_env declared (run in current env).
    """
    groups: dict = {}
    for entry in run_cfg.approaches:
        acfg = OmegaConf.to_container(
            _load_yaml(configs_dir / "approaches" / f"{entry.name}.yaml"), resolve=True
        )
        env = acfg.get("conda_env")
        groups.setdefault(env, [])
        if entry.name not in groups[env]:
            groups[env].append(entry.name)
    return groups


def _build_jobs(run_cfg, project_root: Path, results_dir: str, conda_env_filter: Optional[str] = None):
    """
    Expand a run config into a flat list of job cfgs.
    Each job corresponds to one (approach, task, dataset) triple.

    conda_env_filter: when set, only include approaches whose conda_env matches.
    """
    configs_dir = project_root / "configs"
    global_datasets = OmegaConf.to_container(
        _load_yaml(configs_dir / "global_datasets.yaml"), resolve=True
    )

    # Global task whitelist — applies to all approaches unless overridden per-approach.
    global_task_whitelist = list(run_cfg.tasks) if "tasks" in run_cfg else None

    # Global task_params: flat key-value overrides applied to every task in this run.
    # Lower priority than per-approach task_params; do not affect the task-param slug.
    global_task_params = (
        OmegaConf.to_container(run_cfg.task_params) if "task_params" in run_cfg else {}
    )

    jobs = []
    for approach_entry in run_cfg.approaches:
        approach_name = approach_entry.name

        approach_cfg = OmegaConf.to_container(
            _load_yaml(configs_dir / "approaches" / f"{approach_name}.yaml"),
            resolve=True,
        )

        # Filter by conda_env when running as a dispatched subprocess.
        if conda_env_filter is not None and approach_cfg.get("conda_env") != conda_env_filter:
            continue

        # Run-level param overrides — these also drive the output path so that two entries
        # for the same approach with different params produce distinct directories without
        # any manual label: e.g. embedding_model=all-MiniLM-L6-v2 vs embedding_model=granite-r2,
        # or run_task_based_on=row_embeddings vs run_task_based_on=custom_predictiveML_model.
        run_params = (
            OmegaConf.to_container(approach_entry.params) if "params" in approach_entry else {}
        )
        approach_cfg.update(run_params)

        # Param slug inserted between approach_name and task_name when non-empty.
        # Derived from run-level overrides only (not approach yaml defaults), path-sanitized.
        param_slug = ",".join(
            f"{k}={str(v).replace('/', '_')}" for k, v in sorted(run_params.items())
        )

        # Task whitelist: per-approach entry overrides the run-level global whitelist.
        task_whitelist = (
            list(approach_entry.tasks)
            if "tasks" in approach_entry and approach_entry.tasks
            else global_task_whitelist
        )

        supported_tasks = approach_cfg.get("supported_tasks", {})

        for task_name, task_defaults in supported_tasks.items():
            if task_whitelist is not None and task_name not in task_whitelist:
                continue

            task_defaults = dict(task_defaults) if task_defaults else {}

            # Run-level params that match task_defaults fields override those defaults.
            # This lets run_task_based_on (and similar) be set at the approach level in
            # run configs so they flow into both the path slug and the task config.
            for k, v in run_params.items():
                if k in task_defaults:
                    task_defaults[k] = v

            # Global task_params from run config (applies to all tasks uniformly).
            task_defaults.update(global_task_params)

            # Per-task run overrides (highest priority, override even the above).
            # Captured separately so they can drive the task-param slug in the output path.
            run_task_params = {}
            if "task_params" in approach_entry and task_name in approach_entry.task_params:
                run_task_params = OmegaConf.to_container(approach_entry.task_params[task_name])
                task_defaults.update(run_task_params)

            # Slug derived from run-level task_params overrides — inserted between task_name
            # and dataset_name so task-level variations produce distinct output directories
            # without polluting the approach-level param slug.
            task_param_slug = ",".join(
                f"{k}={str(v).replace('/', '_')}" for k, v in sorted(run_task_params.items())
            )

            # Load task-level defaults (task_name, top_k, elo_metric, etc.)
            task_config_path = (
                project_root / "configs" / "task" / f"{task_name}.yaml"
            )
            if task_config_path.exists():
                task_cfg = OmegaConf.to_container(_load_yaml(task_config_path), resolve=True)
            else:
                task_cfg = {"task_name": task_name}

            # Merge task_defaults (from supported_tasks + run overrides) into task_cfg
            for k, v in task_defaults.items():
                if k != "exclude_datasets":
                    task_cfg[k] = v

            # Determine dataset list
            if "task_datasets" in approach_entry and task_name in approach_entry.task_datasets:
                datasets = list(approach_entry.task_datasets[task_name])
            else:
                global_key = f"{task_name}_datasets"
                datasets = list(global_datasets.get(global_key, []))

            # Apply approach-level exclusions
            approach_exclusions = set(task_defaults.get("exclude_datasets", []))

            # Apply run-level additional exclusions
            if (
                "task_exclude_datasets" in approach_entry
                and task_name in approach_entry.task_exclude_datasets
            ):
                approach_exclusions |= set(approach_entry.task_exclude_datasets[task_name])

            datasets = [d for d in datasets if d not in approach_exclusions]

            for dataset_name in datasets:
                # Build output path:
                #   approach/[param_slug/]task_name/[task_param_slug/]dataset_name
                base = project_root / results_dir / run_cfg.benchmark_output_dir / approach_name
                if param_slug:
                    base = base / param_slug
                base = base / task_name
                if task_param_slug:
                    base = base / task_param_slug
                output_dir = base / dataset_name

                run_identifier = (
                    f"{run_cfg.benchmark_output_dir},{approach_name}"
                    + (f",{param_slug}" if param_slug else "")
                    + f",task={task_name}"
                    + (f",{task_param_slug}" if task_param_slug else "")
                    + f",dataset_name={dataset_name}"
                ).replace("/", "_")

                cfg = OmegaConf.create(
                    {
                        "task": task_cfg,
                        "approach": approach_cfg,
                        "dataset_name": dataset_name,
                        "output_dir": str(output_dir),
                        "cache_dir": str(project_root / "cache"),
                        "project_root": str(project_root),
                        "run_identifier": run_identifier,
                        "test_case_limit": None,
                    }
                )

                jobs.append(cfg)

    return jobs


def _run_job(cfg, project_root: Path):
    """Run a single (approach, task, dataset) job with per-run file logging."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    if (output_dir / "results.json").is_file():
        logger.info(
            f"Skipping {cfg.approach.approach_name}/{cfg.task.task_name}/{cfg.dataset_name}"
            f" — results.json already exists"
        )
        return

    # Per-run log file
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        logger.info(
            f"Starting {cfg.approach.approach_name}/{cfg.task.task_name}/{cfg.dataset_name}"
        )
        from benchmark_src.run_benchmark import run_single
        run_single(cfg)
        logger.info(
            f"Finished {cfg.approach.approach_name}/{cfg.task.task_name}/{cfg.dataset_name}"
        )
    except Exception:
        logger.exception(
            f"Failed {cfg.approach.approach_name}/{cfg.task.task_name}/{cfg.dataset_name}"
        )
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def _gather(results_dir: str, run_cfg):
    results_folder = Path(results_dir) / run_cfg.benchmark_output_dir
    logger.info(f"Gathering results from {results_folder}")
    try:
        from benchmark_src.results_processing import gather_results
        gather_results.run(str(results_folder))
    except Exception:
        logger.exception("Results gathering failed")


def main(
    run_config: Annotated[str, typer.Argument(help="Name of run config in configs/runs/ (without .yaml)")],
    results_dir: Annotated[str, typer.Option(help="Top-level results directory")] = "results",
    stop_on_error: Annotated[bool, typer.Option("--stop-on-error", help="Stop immediately on the first job failure instead of continuing")] = False,
    log_level: Annotated[Optional[str], typer.Option(help="Logging verbosity override (DEBUG, INFO, WARNING, ERROR); run config log_level is used when omitted")] = None,
    conda_env: Annotated[Optional[str], typer.Option("--conda-env", help="Only run approaches with this conda_env (set automatically by multi-env dispatch; rarely needed manually)", hidden=True)] = None,
):
    project_root = Path(__file__).parent.resolve()

    run_config_path = project_root / "configs" / "runs" / f"{run_config}.yaml"
    if not run_config_path.exists():
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Run config not found: {run_config_path}")
        raise typer.Exit(1)

    run_cfg = _load_yaml(run_config_path)

    # CLI --log-level > run config log_level > INFO
    effective_level = log_level or OmegaConf.select(run_cfg, "log_level") or "INFO"
    logging.basicConfig(
        level=getattr(logging, effective_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Redirect stderr to root logger so approach code that prints to stderr is captured
    root_logger = logging.getLogger()
    sys.stderr = StreamToLogger(root_logger, logging.ERROR)

    from dotenv import load_dotenv
    load_dotenv()

    # ── Multi-env dispatch ────────────────────────────────────────────────────
    # When this is the top-level invocation (not a subprocess), check if the run
    # config spans multiple conda envs. If so, dispatch one subprocess per env
    # via `conda run` so the user never has to switch envs manually.
    if conda_env is None:
        env_groups = _collect_env_groups(run_cfg, project_root / "configs")
        named_envs = [e for e in env_groups if e is not None]

        if len(named_envs) > 1 or (len(named_envs) == 1 and None not in env_groups):
            logger.info(
                f"Multi-env run: dispatching subprocesses for envs: {named_envs}"
                + (f" + current env (no conda_env)" if None in env_groups else "")
            )
            failed_envs = []

            for env_name in named_envs:
                logger.info(
                    f"--- Activating env '{env_name}' "
                    f"({env_groups[env_name]}) ---"
                )
                cmd = [
                    "conda", "run", "-n", env_name, "--no-capture-output",
                    "python", str(project_root / "run_experiments.py"),
                    run_config,
                    "--conda-env", env_name,
                    "--results-dir", results_dir,
                ]
                if stop_on_error:
                    cmd.append("--stop-on-error")
                if log_level:
                    cmd.extend(["--log-level", log_level])

                result = subprocess.run(cmd, cwd=str(project_root))
                if result.returncode != 0:
                    failed_envs.append(env_name)
                    logger.error(f"Env '{env_name}' exited with code {result.returncode}")
                    if stop_on_error:
                        raise typer.Exit(1)

            # Run any approaches with no conda_env declared inline in the current env
            if None in env_groups:
                logger.info(f"Running {env_groups[None]} inline (no conda_env declared)")
                _run_inline(run_cfg, project_root, results_dir, stop_on_error, conda_env_filter="__none__")

            _gather(results_dir, run_cfg)

            if failed_envs:
                logger.error(f"Failed envs: {failed_envs}")
                raise typer.Exit(1)
            return

    # ── Single-env execution (inline or dispatched subprocess) ────────────────
    # Map the "__none__" sentinel back to None so the filter works correctly.
    env_filter = None if conda_env == "__none__" else conda_env
    _run_inline(run_cfg, project_root, results_dir, stop_on_error, conda_env_filter=env_filter)
    _gather(results_dir, run_cfg)


def _run_inline(run_cfg, project_root: Path, results_dir: str, stop_on_error: bool, conda_env_filter: Optional[str] = None):
    """Build and execute all jobs for the given conda_env_filter inline."""
    jobs = _build_jobs(run_cfg, project_root, results_dir, conda_env_filter=conda_env_filter)

    logger.info(f"Jobs: {len(jobs)}" + (f" (env filter: {conda_env_filter})" if conda_env_filter else ""))
    for i, cfg in enumerate(jobs):
        logger.info(
            f"  [{i+1}/{len(jobs)}] {cfg.approach.approach_name}/"
            f"{cfg.task.task_name}/{cfg.dataset_name}"
        )

    failed = 0
    for cfg in jobs:
        try:
            _run_job(cfg, project_root)
        except Exception:
            failed += 1
            if stop_on_error:
                logger.error("Stopping after first failure (--stop-on-error)")
                break

    if failed:
        logger.error(f"{failed}/{len(jobs)} jobs failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
