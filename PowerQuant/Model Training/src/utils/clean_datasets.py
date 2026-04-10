import numpy as np
import pandas as pd

dset_names = [
    'all_merge_df_AdaLovelace.csv',
    'all_merge_df_Ampere.csv',
    'all_merge_df_K80.csv',
    'all_merge_df_Tesla.csv',
]
df_ada = pd.read_csv(
    f'./data/{dset_names[0]}'
)
df_ada['Key'] = (
    df_ada['KernelName'].astype(str)
    + df_ada['GridSize'].astype(str)
    + df_ada['BlockSize'].astype(str)
)
df_ada = df_ada.drop_duplicates(subset=['Key'])
df_ada['Architecture'] = 'adaloveace'

df_ampere = pd.read_csv(
    f'./data/{dset_names[1]}'
)
df_ampere['Key'] = (
    df_ampere['KernelName'].astype(str)
    + df_ampere['GridSize'].astype(str)
    + df_ampere['BlockSize'].astype(str)
)
df_ampere = df_ampere.drop_duplicates(subset=['Key'])
df_ampere['Architecture'] = 'ampere'

df_k80 = pd.read_csv(
    f'./data/{dset_names[2]}'
)
df_k80['Key'] = (
    df_k80['KernelName'].astype(str)
    + df_k80['GridSize'].astype(str)
    + df_k80['BlockSize'].astype(str)
)
df_k80 = df_k80.drop_duplicates(subset=['Key'])
df_k80['Architecture'] = 'k80'

df_tesla = pd.read_csv(
    f'./data/{dset_names[3]}'
)
df_tesla['Key'] = (
    df_tesla['KernelName'].astype(str)
    + df_tesla['GridSize'].astype(str)
    + df_tesla['BlockSize'].astype(str)
)
df_tesla = df_tesla.drop_duplicates(subset=['Key'])
df_tesla['Architecture'] = 'tesla'

lstfeat = ['Key', 'Avg']
df1 = df_ada[lstfeat].copy().rename(columns={'Avg': 'Avg_ada'})
df1 = pd.merge(df1, df_ampere[lstfeat], on='Key', how='outer').rename(columns={'Avg': 'Avg_ampere'})
df1 = pd.merge(df1, df_k80[lstfeat], on='Key', how='outer').rename(columns={'Avg': 'Avg_k80'})
df1 = pd.merge(df1, df_tesla[lstfeat], on='Key', how='outer').rename(columns={'Avg': 'Avg_tesla'})

df1 = df1.dropna()

# Get the common keys that exist in all datasets (after removing NaN)
common_keys = set(df1['Key'])

# Filter each dataset to only keep rows with common keys
df_ada_filtered = df_ada[df_ada['Key'].isin(common_keys)].copy()
df_ampere_filtered = df_ampere[df_ampere['Key'].isin(common_keys)].copy()
df_k80_filtered = df_k80[df_k80['Key'].isin(common_keys)].copy()
df_tesla_filtered = df_tesla[df_tesla['Key'].isin(common_keys)].copy()

print('Original dataset sizes:')
print(f'Ada: {len(df_ada)}, Ampere: {len(df_ampere)}, K80: {len(df_k80)}, Tesla: {len(df_tesla)}')
print('Filtered dataset sizes:')
print(
    f'Ada: {len(df_ada_filtered)}, Ampere: {len(df_ampere_filtered)}, K80: {len(df_k80_filtered)}, Tesla: {len(df_tesla_filtered)}'
)
print(f'Common keys: {len(common_keys)}')

df_ada_filtered.to_csv(
    './cache/cleaned_datasets/ada.csv',
    index=False,
)
df_ampere_filtered.to_csv(
    './cache/cleaned_datasets/ampere.csv',
    index=False,
)
df_k80_filtered.to_csv(
    './cache/cleaned_datasets/k80.csv',
    index=False,
)
df_tesla_filtered.to_csv(
    './cache/cleaned_datasets/tesla.csv',
    index=False,
)

df_combined = pd.concat(
    [df_ada_filtered, df_ampere_filtered, df_k80_filtered, df_tesla_filtered], ignore_index=True
)
# df_combined = pd.concat([df_ada_filtered], ignore_index=True)
df_combined.to_csv(
    './cache/cleaned_datasets/combined.csv',
    index=False,
)


lst_selected_features = [
    'avg_comp_lat',
    'avg_glob_lat',
    # 'avg_shar_lat',
    'glob_inst_kernel',
    'glob_load_sm',
    'glob_store_sm',
    'misc_inst_kernel',
    'inst_issue_cycles',
    'cache_penalty',
]


def quant_transform(arr):
    tmp = arr + 0.1 * np.random.randn(len(arr))
    return np.mean(tmp[:, np.newaxis] >= tmp[np.newaxis, :], axis=1)


for col in lst_selected_features:
    df_ada_filtered[col] = quant_transform(df_ada_filtered[col].values)
    df_ampere_filtered[col] = quant_transform(df_ampere_filtered[col].values)
    df_k80_filtered[col] = quant_transform(df_k80_filtered[col].values)
    df_tesla_filtered[col] = quant_transform(df_tesla_filtered[col].values)

df_ada_filtered.to_csv(
    './cache/cleaned_datasets/ada_quant.csv',
    index=False,
)
df_ampere_filtered.to_csv(
    './cache/cleaned_datasets/ampere_quant.csv',
    index=False,
)
df_k80_filtered.to_csv(
    './cache/cleaned_datasets/k80_quant.csv',
    index=False,
)
df_tesla_filtered.to_csv(
    './cache/cleaned_datasets/tesla_quant.csv',
    index=False,
)
df_combined.to_csv(
    './cache/cleaned_datasets/combined_quant.csv',
    index=False,
)
