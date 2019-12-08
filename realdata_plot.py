import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt


def relative_mae(mae_list, mae_best):
    return [mae / mae_best for mae in mae_list]

dir = '_realdata_results'
mae_str = 'Mean Absolute Error'
spearman_str = 'Spearman Correlation'

# PERFORMANCE
sample_size = [100, 400, 1600, 6400]
df_mae = pd.DataFrame(columns=['Value', 'Method', 'Sample Size', 'Metric'])
for n in sample_size:
    p = 'nonlin3_r50_s2_n{}_conf1.pickle'.format(n)
    mae_full, mae_reg, mae_reg_obs, mae_best, _, _, _, SCM = pickle.load(open(os.path.join(dir, p), 'rb'))
    mae_best = mae_best[np.newaxis, :]
    n_rep = len(mae_full)
    df_n = pd.DataFrame({mae_str: np.concatenate([mae_full, mae_reg, mae_reg_obs, mae_best])[:, 0],
                         spearman_str: np.concatenate([mae_full, mae_reg, mae_reg_obs, mae_best])[:, 1],
                         'Method': np.concatenate([np.repeat('SCM', n_rep), np.repeat('REG', n_rep),
                                                   np.repeat('REG_OBS', n_rep), np.repeat('ORACLE', 1)])})
    df_n = pd.melt(df_n, id_vars=['Method'], var_name='Metric', value_name='Value')
    df_n['Sample Size'] = n
    df_mae = pd.concat([df_mae, df_n], ignore_index=True)

# ax = sns.boxplot(x='Method', y=spearman_str, data=df_mae[df_mae.Method != 'REG_OBS'])
# ax.set(yscale='log')
df_mae = df_mae[df_mae.Method != 'REG_OBS']
# ax = sns.catplot(x='Sample Size', y='Value', hue='Method', col='Quantity',
#             data=df_mae, kind="box",height=4, aspect=.7, sharey=False)
g = sns.FacetGrid(df_mae, hue='Method', height=3.5, aspect=1.2, col='Metric', sharey=False,
                  hue_kws={"marker": ["o", "s", ""]})
(g.map(sns.lineplot, 'Sample Size', 'Value', markers=True, dashes=False, ci=None))
methods = ['SCM', 'REG']
metrics = [mae_str, spearman_str]
for i in range(2):
    for j in range(2):
        ax1 = g.axes[0][i]
        cur_df = df_mae.query("Method == '{}' and Metric == '{}'".format(methods[j], metrics[i]))
        lb = cur_df.groupby("Sample Size")['Value'].agg(lambda x: np.quantile(x, 0.1))
        ub = cur_df.groupby("Sample Size")['Value'].agg(lambda x: np.quantile(x, 0.9))
        ax1.fill_between(x=np.array(sample_size),
                         y1=ub,
                         y2=lb,
                         alpha=0.2)

g.set(xscale='log')
g.set_ylabels("")
plt.legend(loc='lower right',
           fancybox=True, title="Method")
plt.tight_layout()
plt.savefig("_plots/realdata_nonlin.pdf")

# VARY CONFOUNDING LEVEL
max_corre_list = [0.1, 0.35, 0.6, 0.85]
df_mae = pd.DataFrame(columns=['Value', 'Method', 'Sample Size', 'Metric'])
for c in max_corre_list:
    p = 'nonlin3_r50_s2_n1600_conf{}.pickle'.format(c)
    mae_full, mae_reg, mae_reg_obs, mae_best, _, _, _, SCM = pickle.load(open(os.path.join(dir, p), 'rb'))
    mae_best = mae_best[np.newaxis, :]
    n_rep = len(mae_full)
    df_n = pd.DataFrame({mae_str: np.concatenate([mae_full, mae_reg, mae_reg_obs, mae_best])[:, 0],
                         spearman_str: np.concatenate([mae_full, mae_reg, mae_reg_obs, mae_best])[:, 1],
                         'Method': np.concatenate([np.repeat('SCM', n_rep), np.repeat('REG', n_rep),
                                                   np.repeat('REG_OBS', n_rep), np.repeat('ORACLE', 1)])})
    df_n = pd.melt(df_n, id_vars=['Method'], var_name='Metric', value_name='Value')
    df_n['Max Corr'] = c
    df_mae = pd.concat([df_mae, df_n], ignore_index=True)

# ax = sns.boxplot(x='Method', y=spearman_str, data=df_mae[df_mae.Method != 'REG_OBS'])
# ax.set(yscale='log')
df_mae = df_mae[df_mae.Method != 'REG_OBS']
# ax = sns.catplot(x='Sample Size', y='Value', hue='Method', col='Quantity',
#             data=df_mae, kind="box",height=4, aspect=.7, sharey=False)
g = sns.FacetGrid(df_mae, hue='Method', height=3.5, aspect=1.2, col='Metric', sharey=False,
                  hue_kws={"marker": ["o", "s", ""]})
(g.map(sns.lineplot, 'Max Corr', 'Value', markers=True, dashes=False, ci=None))
g.set_ylabels("")
g.set_xlabels("Max Corr (c)")
methods = ['SCM', 'REG']
metrics = [mae_str, spearman_str]
for i in range(2):
    for j in range(2):
        ax1 = g.axes[0][i]
        cur_df = df_mae.query("Method == '{}' and Metric == '{}'".format(methods[j], metrics[i]))
        lb = cur_df.groupby("Max Corr")['Value'].agg(lambda x: np.quantile(x, 0.05))
        ub = cur_df.groupby("Max Corr")['Value'].agg(lambda x: np.quantile(x, 0.95))
        ax1.fill_between(x=np.array(max_corre_list),
                         y1=ub,
                         y2=lb,
                         alpha=0.3)

plt.legend(loc='lower left',
           fancybox=True, title="Method")
plt.tight_layout()
plt.savefig("_plots/realdata_conf.pdf")