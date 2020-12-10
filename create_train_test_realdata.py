import pickle

import pandas as pd

from models import *
from utils import sample_corr_matrix
import igraph as ig

data_dir = '_realdata/DREAM4_InSilico_Size10/insilico_size10_3'
data = pd.read_csv(os.path.join(data_dir, 'insilico_size10_3_multifactorial.tsv'), sep='\t')
# first we standardize data
data = (data - data.mean()) / data.std()
max_sample = 6400
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

g_network = pd.read_csv(
    '_realdata/DREAM4_Challenge2_GoldStandards/Size 10/DREAM4_GoldStandard_InSilico_Size10_3.tsv',
    sep='\t', names=['L', 'R', 'V'])

g_network = g_network[g_network.V == 1]

edges = [tuple(x) for x in g_network[g_network.columns[:2]].values]

G = ig.Graph(directed=True)
G.add_vertices(data.columns.values)
G.add_edges(edges)
rm_idx = G.feedback_arc_set(method='ip')
G.delete_edges(rm_idx)
G.is_dag()
node_map = dict(zip(G.vs['name'], G.vs.indices))

intercept = False
interaction = True

corr_matrix = to_torch(sample_corr_matrix(G.vcount(), 4))
linearSCM = LinearSCM('gen_dat', G, corr_matrix=corr_matrix,
                      interaction=interaction, intercept=intercept)
train_data = to_torch(data.values)
linearSCM.train(obs_data=train_data, lr=0.001)


def set_conf_level(corr_matrix, conf_level=1.):
    cp_cor = corr_matrix.clone()
    cp_cor[np.arange(G.vcount()), np.arange(G.vcount())] = 0.0
    max_corr = cp_cor.abs().max()
    cp_cor = (corr_matrix / max_corr) * conf_level
    cp_cor[np.arange(G.vcount()), np.arange(G.vcount())] = 1.
    return cp_cor


for conf_level in [0.1, 0.35, 0.6, 0.85, 1]:
    if conf_level != 1:
        mod_corr_matrix = set_conf_level(corr_matrix, conf_level)
        linearSCM.corr_matrix = mod_corr_matrix

    include_node = G.vs['name']
    include_idx = [node_map[var] for var in include_node]

    obs_data_full = linearSCM.sample(max_sample)[:, include_idx]
    joint_intervention = {}
    n_test = 100000

    for var in G.topological_sorting()[:-1]:
        dat = obs_data_full[:, var]
        mean, std = dat.mean(), dat.std()
        joint_intervention[var] = torch.rand(n_test) * 4 * std + (mean - 2 * std)

    test_data = linearSCM.sample(n_test, joint_intervention)[:, include_idx]

    with open("_realdata/train_test_s{}_conf{}.pickle".format(seed, conf_level), 'wb') as handle:
        pickle.dump([obs_data_full, test_data, linearSCM, G], handle, protocol=pickle.HIGHEST_PROTOCOL)
