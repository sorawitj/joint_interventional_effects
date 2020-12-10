import numpy as np
import pandas as pd
import pickle
import os

import seaborn as sns
import matplotlib.pyplot as plt

from utils import add_interaction

main_dir = '_synthetic_results'

# CONSISTENCY
sample_size = [100, 400, 1600, 6400, 25600]
file = {'Linear': 'cons2_interFalse_s2_n{}.pickle', 'Nonlinear': 'cons2_interTrue_s2_n{}.pickle'}
df_mae = pd.DataFrame(columns=['Mean Absolute Error', 'Method', 'Sample Size', 'Quantity'])
for fn in ['Linear', 'Nonlinear']:
    dir = file[fn]
    for n in sample_size:
        p = dir.format(n)
        _, _, _, f_res, coef_reg, actual_coef = pickle.load(open(os.path.join(main_dir, p), 'rb'))
        _, _, _, _, coef_reg_ind, _ = pickle.load(
            open(os.path.join(main_dir, "reg_regime_inter{}_s2_n{}.pickle".format(fn == 'Nonlinear', n)), 'rb'))
        n_rep = len(coef_reg)
        test_point = np.random.uniform(-3, 3, size=(100000, 3))
        if fn == 'Nonlinear':
            test_point = add_interaction(test_point)
        ex_y = actual_coef.dot(test_point.T)

        pred_y = f_res['y'].dot(test_point.T).squeeze()
        pred_reg = coef_reg.dot(test_point.T).squeeze()
        pred_reg_ind = coef_reg_ind.dot(test_point.T).squeeze()

        mse_pred = np.abs(pred_y - ex_y).mean(1)
        mse_reg = np.abs(pred_reg - ex_y).mean(1)
        mse_reg_ind = np.abs(pred_reg_ind - ex_y).mean(1)

        mse_param_pred = np.abs((actual_coef - f_res['y'])).mean(1)
        mse_param_reg = np.abs((actual_coef - coef_reg)).mean(1)
        mse_param_reg_ind = np.abs((actual_coef - coef_reg_ind)).mean(1)

        df_n = pd.DataFrame({'Mean Absolute Error': np.concatenate([mse_pred, mse_reg, mse_reg_ind,
                                                                    mse_param_pred, mse_param_reg, mse_param_reg_ind]),
                             'Method': np.concatenate(
                                 [np.repeat('ANM', n_rep), np.repeat('REG', n_rep), np.repeat('REG_IND', n_rep),
                                  np.repeat('ANM', n_rep), np.repeat('REG', n_rep), np.repeat('REG_IND', n_rep)]),
                             'Quantity': np.concatenate([np.repeat('Joint Effect', 3 * n_rep),
                                                         np.repeat('Parameters', 3 * n_rep)])})
        df_n['Sample Size'] = n
        df_n['Function'] = fn
        df_mae = pd.concat([df_mae, df_n], ignore_index=True)

for fn in ['Linear', 'Nonlinear']:
    df_fn = df_mae.query("Function == '{}'".format(fn))
    g = sns.FacetGrid(df_fn, hue='Method', col='Quantity', height=3, aspect=0.9,
                      sharey=False, hue_kws={"marker": ["s", "D", "o"]})
    (g.map(sns.lineplot, 'Sample Size', 'Mean Absolute Error', markers=True, dashes=False).add_legend())
    g.set(xscale='log')
    g.set(yscale='log')
    g.add_legend()
    g.axes[0][0].set_title(r'E$(Y \:\vert\: do(X_1, X_2, X_3))$', size=11)
    g.axes[0][1].set_title(r'Parameters ($\theta$)', size=11)
    g.fig.suptitle("{}".format(fn), size=12)
    g.fig.subplots_adjust(top=.8)
    plt.savefig("_plots/consistencyNew_{}.pdf".format(fn))

# UNBIASEDNESS
for fn in ['Linear', 'Nonlinear']:
    n = 1600
    is_lin = 'False' if fn == 'Linear' else 'True'
    p = 'cons2_inter{}_s2_n{}.pickle'.format(is_lin, n)
    ex_y, pred_y, pred_reg, f_res, coef_reg, act_coefs = pickle.load(open(os.path.join(main_dir, p), 'rb'))
    n_rep = len(pred_y)
    df_pred = pd.DataFrame({'Value': np.concatenate([pred_y, pred_reg]),
                            'Method': np.concatenate([np.repeat('ANM', n_rep), np.repeat('REG', n_rep)])})

    # PARAMETERS
    columns = ['b_{}'.format(i) for i in range(f_res['y'].shape[1])]
    df_params_pred = pd.DataFrame(f_res['y'], columns=columns)
    df_params_pred = pd.melt(df_params_pred, value_vars=columns, var_name='Params')
    df_params_pred['Model'] = 'ANM'

    df_params_reg = pd.DataFrame(coef_reg, columns=columns)
    df_params_reg = pd.melt(df_params_reg, value_vars=columns, var_name='Params')
    df_params_reg['Model'] = 'REG'

    df_prams = pd.concat([df_params_pred, df_params_reg], ignore_index=True)

    g = sns.FacetGrid(df_prams, hue='Params', row='Model',
                      palette="Set1", height=1.7, aspect=4.5, sharey=False)
    g.map(sns.distplot, "value", norm_hist='True')
    cols = []
    for line in g.axes[0][0].legend().parent.get_lines():
        cols += [line.get_color()]
    actual_coef = [1.5, 1, -0.5, 0.5, -1, -1.5]
    for col, coef in zip(cols, actual_coef):
        g.map(plt.axvline, x=coef, ls='--', c=col, alpha=0.8)
    g.fig.suptitle("Linear Function", size=12)
    g.fig.subplots_adjust(top=.85)
    plt.savefig("_plots/unbiasness_{}_params.pdf".format(fn))

# MULTIPLE EXPERIMENTS
sample_size = [400, 1600, 6400]
file = {'Linear': 'multiexp_lin_s2_n{}.pickle', 'Nonlinear': 'multiexp_nonlin_s2_n{}.pickle'}
win_st = 'MAE-ANM < MAE-REG'
reg_st = 'MAE-REG'
pred_st = 'MAE-ANM'
df_acc = pd.DataFrame(columns=[pred_st, reg_st, win_st])
for n in sample_size:
    for fn in ['Linear']:
        p = file[fn].format(n)
        ex_y, pred_y, pred_reg, f_res, coef_reg, int_cov_res, obs_cov_res = pickle.load(
            open(os.path.join(main_dir, p), 'rb'))
        n_rep = len(pred_y)
        mae_pred, mae_reg = np.abs((pred_y - ex_y)), np.abs((pred_reg - ex_y))
        df_n = pd.DataFrame({reg_st: mae_reg,
                             pred_st: mae_pred,
                             win_st: mae_pred < mae_reg})
        df_n['Function'] = fn
        df_n['Sample Size'] = str(n)
        df_acc = pd.concat([df_acc, df_n], ignore_index=True)

# EXPECTED VALUE
g = sns.FacetGrid(df_acc, hue=win_st, hue_order=[True, False], hue_kws=dict(marker=["o", "x"]),
                  col='Sample Size', col_order=[str(s) for s in sample_size],
                  height=3, aspect=1., palette=['forestgreen', 'orangered'], legend_out=True)
(g.map(sns.scatterplot, pred_st, reg_st, s=40))
g.set(xlim=(0, 1.35))
g.set(ylim=(0, 3.8))

plt.legend(loc='upper right',
           fancybox=True, ncol=2, title=win_st)
g.fig.suptitle("Linear", size=12)
g.fig.subplots_adjust(top=.8)

plt.savefig("_plots/multiexp_lin.pdf")

# CONSISTENCY
sample_size = [100, 400, 1600, 6400, 25600]
file = {'Linear': 'reg_regime_interFalse_s2_n{}.pickle', 'Nonlinear': 'reg_regime_interTrue_s2_n{}.pickle'}
df_mae = pd.DataFrame(columns=['Mean Absolute Error', 'Method', 'Sample Size', 'Quantity'])
for fn in ['Linear', 'Nonlinear']:
    dir = file[fn]
    for n in sample_size:
        p = dir.format(n)
        ex_y, pred_reg_ind, pred_reg, f_res, coef_reg, actual_coef = pickle.load(open(os.path.join(main_dir, p), 'rb'))
        n_rep = len(coef_reg)

        mse_reg = np.abs(pred_reg - ex_y).mean(1)
        mse_reg_ind = np.abs(pred_reg_ind - ex_y).mean(1)

        df_n = pd.DataFrame({'Mean Absolute Error': np.concatenate([mse_reg, mse_reg_ind]),
                             'Method': np.concatenate(
                                 [np.repeat('REG', n_rep), np.repeat('REG_IND', n_rep)])})
        df_n['Sample Size'] = n
        df_n['Function'] = fn
        df_mae = pd.concat([df_mae, df_n], ignore_index=True)

for fn in ['Linear', 'Nonlinear']:
    df_fn = df_mae.query("Function == '{}'".format(fn))
    g = sns.FacetGrid(df_fn, hue='Method', height=3, aspect=0.9,
                      sharey=False, hue_kws={"marker": ["s", "D"]})
    (g.map(sns.lineplot, 'Sample Size', 'Mean Absolute Error', markers=True, dashes=False).add_legend())
    g.set(xscale='log')
    g.add_legend()
    g.fig.suptitle("{}".format(fn), size=12)
    g.fig.subplots_adjust(top=.8)
    plt.savefig("_plots/sanity_NEW_{}.pdf".format(fn))
