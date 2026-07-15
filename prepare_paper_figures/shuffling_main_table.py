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

    # Build CI map for Accuracy
    ci_map = None
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        ci_acc = ci_df[(ci_df['Metric'] == 'Accuracy') & (ci_df['Dataset'].str.endswith('@@v0'))]
        if len(ci_acc) > 0:
            # Build mapping: (chart_name, dataset_stripped) -> (ci_lower, ci_upper)
            filtered_for_map = filtered[['chart_name', 'Approach', 'Configuration', 'dataset']].drop_duplicates()
            name_to_keys = {}
            for _, row in filtered_for_map.iterrows():
                ds_stripped = row['dataset'].replace('@@v0', '')
                name_to_keys[(row['Approach'], row['Configuration'], ds_stripped)] = (row['chart_name'], ds_stripped)

            ci_map = {}
            for _, row in ci_acc.iterrows():
                ds_stripped = row['Dataset'].replace('@@v0', '')
                mapped = name_to_keys.get((row['Approach'], row['Configuration'], ds_stripped))
                if mapped:
                    ci_map[mapped] = (row['CI_lower'], row['CI_upper'])

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_accuracy_table.tex',
        caption='Table Shuffling accuracy per dataset (v0: hi-pos/hi-neg, both row+col perturbation, '
                'default window). Best per column in bold, second-best underlined. Bracketed values show bootstrapped 95\\% CIs.',
        label='tab:shuffling_accuracy',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
        ci_map=ci_map,
        tabcolsep=3.84,
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
