import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity

from utils import add_interaction

sample_size = [400, 1600]
df_pred = pd.DataFrame(columns=['Prediction', 'Distance', 'Sample Size'])
dir = '_realdata_results/nonlin_r50_s2_n{}_conf1.pickle'
for n in sample_size:
    np.random.seed(1)
    torch.random.manual_seed(1)
    p = dir.format(n)
    _, _, _, _, f_res, _, _, SCM = pickle.load(open(p, 'rb'))
    f_res = f_res.squeeze()
    test_size = 1000
    y_parents = SCM.parents[SCM.nodes[-1]]
    samples = SCM.sample(20000)[:, y_parents].detach().numpy()
    kde = KernelDensity(1.)
    kde.fit(samples)

    test_points = np.random.uniform(-2, 2, size=(test_size, len(y_parents)))
    distance = -kde.score_samples(test_points)
    test_points = add_interaction(test_points)
    preds = f_res[:, 0] + test_points.dot(f_res[:, 1:].T)
    selected_idx = (np.argsort(np.argsort(distance)) % 100 == 0)
    selected_dist = distance[selected_idx]
    rank_dist = pd.DataFrame(distance[selected_idx]).rank()
    selected_rank = np.concatenate([np.repeat(d, preds.shape[1]) for d in rank_dist[0]])

    df_n = pd.DataFrame({'Prediction': preds[selected_idx, :].flatten(),
                         'Density Rank': selected_rank})
    df_n['Sample Size'] = n
    df_pred = pd.concat([df_pred, df_n], ignore_index=True)

g = sns.FacetGrid(df_pred, col='Sample Size', height=3.5, aspect=1.2)
g.map(sns.pointplot, 'Density Rank', 'Prediction', scale=0.7, data=df_pred, join=False, ci=False, order=np.arange(1, len(rank_dist) + 1))
g.set_xticklabels(np.arange(1, len(rank_dist) + 1))
# ax = sns.pointplot(x='Rank Distance', y='Prediction', hue='Sample Size', dodge=True, scale=0.8, data=df_pred, join=False)
for i in range(len(sample_size)):
    y_preds = df_pred[df_pred['Sample Size'] == sample_size[i]].groupby("Density Rank")['Prediction'].mean()
    se = df_pred[df_pred['Sample Size'] == sample_size[i]].groupby("Density Rank")['Prediction'].agg(
        lambda x: np.std(x))
    g.axes[0][i].errorbar(x=np.arange(0, len(rank_dist)),
                          y=y_preds,
                          yerr=2 * se,
                          capsize=4,
                          fmt='none')

plt.tight_layout()
plt.savefig("_plots/uncertainty_real.pdf")
