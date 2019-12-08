import sys
import numpy as np
from model import *
import pickle

from utils import get_gen_data

if __name__ == "__main__":

    np.random.seed(2)

    try:
        # get an index as an argument
        idx = int(sys.argv[1])
    except:
        idx = 0

    n = 1000
    f_res = {'m1': [], 'm2': [], 'm3': [], 'y': []}
    obs_cov_res = {'obs': [], 'int0': [], 'int1': [], 'int2': []}
    int_cov_res = {'m1': [], 'm2': [], 'm3': []}
    for i in range(n):
        cov = np.array([[1, .7, .3, .4], [.7, 1, .5, .3], [.3, .5, 1, .1], [.4, .3, .1, 1]])
        f = {'m2': lambda m1: 2 * m1,
             'm3': lambda m1, m2: 1 * m1 - 1 * m2,
             'y': lambda m1, m2, m3: 1.5 * m1 - 2 * m2 - 1.5 * m3}
        gen_data = get_gen_data(cov, f)

        m1, m2, m3, y = gen_data(n)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n) * 2
        x3 = np.random.randn(n) * 2
        do_m1, m2_dom1, m3_dom1, y_dom1 = gen_data(n, x1=x1)
        m1_dom2, do_m2, m3_dom2, y_dom2 = gen_data(n, x2=x2)
        m1_dom3, m2_dom3, do_m3, y_dom3 = gen_data(n, x3=x3)

        g = ig.Graph(directed=True)
        g.add_vertices([0, 1, 2, 3])
        g.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        model = JointInterventionModel('test', g)
        obs_data = torch.cat([m1, m2, m3, y], dim=1)
        int_data = {1: torch.cat([m1_dom2, do_m2, m3_dom2, y_dom2], dim=1),
                    0: torch.cat([do_m1, m2_dom1, m3_dom1, y_dom1], dim=1),
                    2: torch.cat([m1_dom3, m2_dom3, do_m3, y_dom3], dim=1)}
        model.train(obs_data, int_data, outer_n_iter=10, inner_n_iter=1000, lr=0.01)

        for f, k in zip(model.F, f_res.keys()):
            f_res[k] += [f.squeeze().detach().numpy()]
        for cov, k in zip(model.int_covs.values(), int_cov_res.keys()):
            int_cov_res[k] += [cov.squeeze().detach().numpy()]

        obs_cov_res['obs'] += [model.obs_cov.squeeze().detach().numpy()]

    for k in f_res.keys():
        f_res[k] = np.stack(f_res[k])
    for k in int_cov_res.keys():
        int_cov_res[k] = np.stack(int_cov_res[k])
    obs_cov_res['obs'] = np.stack(obs_cov_res['obs'])

    # save the results
    with open("../_synthetic_results/params_res{}.pickle".format(idx), 'wb') as handle:
        pickle.dump(f_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(int_cov_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(obs_cov_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
