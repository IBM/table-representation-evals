"""T8 + T8-BCS: Table Shuffling main tables — accuracy and BCS, v0 only."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def _prepare_v0_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to v0 variation (hi-pos/hi-neg, perturbation=both, default window)."""
    filtered = df.copy()
    filtered = filtered[filtered['dataset'].str.endswith('@@v0')]
    # Strip @@v0 suffix for cleaner display
    filtered['dataset'] = filtered['dataset'].str.replace(r'@@v0$', '', regex=True)
    return filtered


def create_accuracy_table(df: pd.DataFrame, plots_folder: Path):
    """T8: Triplet accuracy per dataset at v0."""
    filtered = _prepare_v0_data(df)

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    # Pivot: rows=approach, columns=dataset
    pivoted = filtered.pivot_table(
        index='chart_name',
        columns='dataset',
        values=metric_col,
        aggfunc='mean',
    )

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_accuracy_table.tex',
        caption='Table Shuffling accuracy per dataset (v0: hi-pos/hi-neg, both row+col perturbation, '
                'default window). Best per column in bold, second-best underlined.',
        label='tab:shuffling_accuracy',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
    )


def create_bcs_table(df: pd.DataFrame, plots_folder: Path):
    """T8-BCS: Bounded Contrastive Score per dataset at v0."""
    filtered = _prepare_v0_data(df)

    metric_col = 'Bounded Contrastive Score_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    # Pivot: rows=approach, columns=dataset
    pivoted = filtered.pivot_table(
        index='chart_name',
        columns='dataset',
        values=metric_col,
        aggfunc='mean',
    )

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_bcs_table.tex',
        caption='Table Shuffling Bounded Contrastive Score per dataset '
                '(v0: hi-pos/hi-neg, both row+col perturbation, default window). '
                'Lower is better. Best per column in bold, second-best underlined.',
        label='tab:shuffling_bcs',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
        higher_better=False,
    )
