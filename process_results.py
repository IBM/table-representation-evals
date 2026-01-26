import pandas as pd
import sys

task = sys.argv[1]

df = pd.read_csv('all_results_aggregated.csv') 
df = df[df['task'] == task]

keep = ['Approach', 'task', 'dataset']

def pivot_by_metric(df, metric_name, agg_func='mean'):
    return df.groupby(['dataset', 'Approach'])[metric_name].agg(agg_func).unstack('Approach')


df['dataset'] = df['dataset'].str.replace('_', r'\_')
if task == 'predictive_ml':
    metric = sys.argv[2]
    
    for c in df.columns:
        if ('roc_auc' in c and 'std' not in c) or ('rmse' in c and 'std' not in c) or ('log_loss' in c and 'std' not in c):
            print('metric columns', c)
            keep.append(c)
    keep.append('Configuration')

    orig_df = df[keep]
    df = orig_df[orig_df['Approach'] != 'GritLM']
    df = df[~df['Configuration'].str.contains('based_on=row_embeddings')]
    
    df = pivot_by_metric(df, metric)
    df = df[df.notna().sum(axis=1) > 1]
elif task == 'row_similarity_search' or task == 'column_similarity_search':
    metric = sys.argv[2]
    for c in df.columns:
        if ('top' in c and 'std' not in c) or ('MRR' in c and 'std' not in c):
            print('metric columns', c)
            keep.append(c)
    orig_df = df[keep]
    df = orig_df[orig_df['Approach'] != 'GritLM']
    print(orig_df)
    df = pivot_by_metric(df, metric)
df = df.round(2)
df = df.fillna('-')
print(df)
print(df.to_latex(float_format="%.2f"))
    
