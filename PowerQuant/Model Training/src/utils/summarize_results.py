import json
import os
import sys

import dotenv
import numpy as np
import pandas as pd

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)


def prepare_df_with_multiindex(df, test_arch: str = None):
    """
    Use MultiIndex for better hierarchical structure
    """
    rows = []
    index = []

    classifiers = df['classifier'].unique()
    if test_arch is not None:
        df = df[df['test_arch'] == test_arch]

    for clf in classifiers:
        clf_data = df[df['classifier'] == clf]

        baseline = clf_data[clf_data['quantile/normal'] == 'Normal'].iloc[0]
        quantile = clf_data[clf_data['quantile/normal'] == 'Quantile'].iloc[0]

        # Baseline row
        index.append((clf, 'Baseline'))
        rows.append(
            {
                '$R^2$': f'${baseline["r2_mean"]:.3f} \\pm {baseline["r2_std"]:.3f}$',
                'RMSE': f'${baseline["rmse_mean"]:.3f} \\pm {baseline["rmse_std"]:.3f}$',
                'MAE': f'${baseline["mae_mean"]:.3f} \\pm {baseline["mae_std"]:.3f}$',
            }
        )

        # Quantile row
        index.append((clf, 'Quantile'))
        rows.append(
            {
                '$R^2$': f'${quantile["r2_mean"]:.3f} \\pm {quantile["r2_std"]:.3f}$',
                'RMSE': f'${quantile["rmse_mean"]:.3f} \\pm {quantile["rmse_std"]:.3f}$',
                'MAE': f'${quantile["mae_mean"]:.3f} \\pm {quantile["mae_std"]:.3f}$',
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.index = pd.MultiIndex.from_tuples(index, names=['Classifier', ''])

    return result_df


"""
Experiment 01: Train/Test on all architectures
"""
results_dict_exp01 = {
    'experiment_id': [],
    'classifier': [],
    'quantile/normal': [],
    'r2': [],
    'rmse': [],
    'mae': [],
    'file': [],
}

list_exp01_dirs = {
    'exp01_quant_catboost': ('CatBoost', 'Quantile'),
    'exp01_catboost': ('CatBoost', 'Normal'),
    'exp01_quant_neuralnet': ('NeuralNet', 'Quantile'),
    'exp01_neuralnet': ('NeuralNet', 'Normal'),
    'exp01_quant_randomforest': ('RandomForest', 'Quantile'),
    'exp01_randomforest': ('RandomForest', 'Normal'),
}


for exp01_dir, (classifier, quantile_normal) in list_exp01_dirs.items():
    results_dir = f'{project_root}/results/{exp01_dir}/'
    best_r2 = -np.inf
    best_rmse = np.inf
    best_mae = np.inf
    best_file = ''
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                results_dict_exp01['experiment_id'].append(data['metadata']['experiment_id'])
                results_dict_exp01['classifier'].append(classifier)
                results_dict_exp01['quantile/normal'].append(quantile_normal)
                results_dict_exp01['r2'].append(data['results']['r2'])
                results_dict_exp01['rmse'].append(data['results']['rmse'])
                results_dict_exp01['mae'].append(data['results']['mae'])
                results_dict_exp01['file'].append(file)

df_exp01 = pd.DataFrame(results_dict_exp01)
print(df_exp01)

# Calculate mean and std of top 5 results for each experiment_id
summary_dict = {
    'experiment_id': [],
    'classifier': [],
    'quantile/normal': [],
    'r2_mean': [],
    'r2_std': [],
    'rmse_mean': [],
    'rmse_std': [],
    'mae_mean': [],
    'mae_std': [],
}

for exp_id in df_exp01['experiment_id'].unique():
    exp_df = df_exp01[df_exp01['experiment_id'] == exp_id]
    top_k = 5

    # Get top 5 values for each metric
    top_k_r2 = exp_df.nlargest(top_k, 'r2')['r2']
    top_k_rmse = exp_df.nsmallest(top_k, 'rmse')['rmse']
    top_k_mae = exp_df.nsmallest(top_k, 'mae')['mae']

    summary_dict['experiment_id'].append(exp_id)
    summary_dict['classifier'].append(exp_df['classifier'].iloc[0])
    summary_dict['quantile/normal'].append(exp_df['quantile/normal'].iloc[0])
    summary_dict['r2_mean'].append(top_k_r2.mean())
    summary_dict['r2_std'].append(top_k_r2.std())
    summary_dict['rmse_mean'].append(top_k_rmse.mean())
    summary_dict['rmse_std'].append(top_k_rmse.std())
    summary_dict['mae_mean'].append(top_k_mae.mean())
    summary_dict['mae_std'].append(top_k_mae.std())

df_summary = pd.DataFrame(summary_dict)
print('\nSummary statistics of top 5 results for each experiment:')
print(df_summary)

# Usage
prepared_df = prepare_df_with_multiindex(df_summary)

latex_code = prepared_df.to_latex(
    escape=False,
    column_format='llccc',
    multirow=True,
    caption='Performance comparison across classifiers. Results reported as mean $\\pm$ standard deviation.',
    label='tab:results',
    position='htbp',
)

print(latex_code)


"""
Experiment 02: Train/Test on adalovelace architecture
"""
results_dict_exp02 = {
    'experiment_id': [],
    'classifier': [],
    'quantile/normal': [],
    'test_arch': [],
    'r2': [],
    'rmse': [],
    'mae': [],
    'file': [],
}


list_exp02_dirs = {
    'exp02_quant_catboost': ('CatBoost', 'Quantile'),
    'exp02_catboost': ('CatBoost', 'Normal'),
    'exp02_quant_neuralnet': ('NeuralNet', 'Quantile'),
    'exp02_neuralnet': ('NeuralNet', 'Normal'),
    'exp02_quant_randomforest': ('RandomForest', 'Quantile'),
    'exp02_randomforest': ('RandomForest', 'Normal'),
}

for exp02_dir, (classifier, quantile_normal) in list_exp02_dirs.items():
    results_dir = f'{project_root}/results/{exp02_dir}/'
    for file in os.listdir(results_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
        results_dict_exp02['experiment_id'].append(data['metadata']['experiment_id'])
        results_dict_exp02['classifier'].append(classifier)
        results_dict_exp02['quantile/normal'].append(quantile_normal)
        results_dict_exp02['test_arch'].append(data['metadata']['test_arch'])
        results_dict_exp02['r2'].append(data['results']['r2'])
        results_dict_exp02['rmse'].append(data['results']['rmse'])
        results_dict_exp02['mae'].append(data['results']['mae'])
        results_dict_exp02['file'].append(file)


df_exp02 = pd.DataFrame(results_dict_exp02)


# Calculate mean and std of to  p 5 results for each experiment_id
summary_dict = {
    'key': [],
    'experiment_id': [],
    'classifier': [],
    'quantile/normal': [],
    'test_arch': [],
    'r2_mean': [],
    'r2_std': [],
    'rmse_mean': [],
    'rmse_std': [],
    'mae_mean': [],
    'mae_std': [],
}

df_exp02['Key'] = df_exp02['experiment_id'] + '_' + df_exp02['test_arch']
for key in df_exp02['Key'].unique():
    exp_df = df_exp02[df_exp02['Key'] == key]
    top_k = 5

    # Get top 5 values for each metric
    top_k_r2 = exp_df.nlargest(top_k, 'r2')['r2']
    top_k_rmse = exp_df.nsmallest(top_k, 'rmse')['rmse']
    top_k_mae = exp_df.nsmallest(top_k, 'mae')['mae']

    summary_dict['key'].append(key)
    summary_dict['experiment_id'].append(exp_df['experiment_id'].iloc[0])
    summary_dict['classifier'].append(exp_df['classifier'].iloc[0])
    summary_dict['quantile/normal'].append(exp_df['quantile/normal'].iloc[0])
    summary_dict['test_arch'].append(exp_df['test_arch'].iloc[0])
    summary_dict['r2_mean'].append(top_k_r2.mean())
    summary_dict['r2_std'].append(top_k_r2.std())
    summary_dict['rmse_mean'].append(top_k_rmse.mean())
    summary_dict['rmse_std'].append(top_k_rmse.std())
    summary_dict['mae_mean'].append(top_k_mae.mean())
    summary_dict['mae_std'].append(top_k_mae.std())

df_summary = pd.DataFrame(summary_dict)
print('\nSummary statistics of top 5 results for each experiment:')
print(df_summary)

for test_arch in df_summary['test_arch'].unique():
    prepared_df = prepare_df_with_multiindex(df_summary, test_arch=test_arch)

    latex_code = prepared_df.to_latex(
        escape=False,
        column_format='llccc',
        multirow=True,
        caption=f'Performance comparison across classifiers. Results reported as mean $\\pm$ standard deviation. Test architecture: {test_arch}.',
        label=f'tab:results_{test_arch}',
        position='htbp',
    )
    print('%' * 60)
    print(f'% Test architecture: {test_arch}')
    print('%' * 60)
    print(latex_code)
    print('%' * 60)
