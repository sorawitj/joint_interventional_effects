import itertools
import sys
from tqdm import tqdm

from model import *
import pickle

from utils import get_gen_data, sample_corr_matrix, create_F, sample_coefs

if __name__ == "__main__":

    try:
        # get an index as an argument
        round = int(sys.argv[1])
    except:
        round = -1

    seed = 2
    np.random.seed(seed)

    sample_size = [1600, 6400]
    if round == -1:
        pass
    else:
        sample_size = [sample_size[round]]
    n_rounds = 100
    n_dim = 4
    interaction = False
    intercept = False
    g = ig.Graph(directed=True)
    g.add_vertices(list(range(n_dim)))
    edges = list(itertools.combinations(range(n_dim), 2))
    g.add_edges(edges)

    model = JointInterventionModel('exp', g, interaction=interaction, intercept=intercept)
    reg_model = RegressionModel(g, interaction=interaction, intercept=intercept)
    for n_sample in sample_size:
        f_res = {'m1': [], 'm2': [], 'm3': [], 'y': []}
        obs_cov_res = {'obs': [], 'int0': [], 'int1': [], 'int2': []}
        int_cov_res = {'m1': [], 'm2': [], 'm3': []}
        pred_y, pred_reg, reg_coefs, ex_y = [], [], [], []

        for idx in tqdm(range(n_rounds)):

            corr = sample_corr_matrix(n_dim, 2)
            diag_std = np.eye(n_dim) * np.repeat(1.0, n_dim)
            cov = diag_std.dot(corr).dot(diag_std)
            coefs = sample_coefs(n_dim, 1.5, interaction=interaction, intercept=intercept)
            F = create_F(coefs, interaction=interaction)
            gen_data = get_gen_data(cov, F)

            obs_data = gen_data(n_sample, n_dim)

            test_point = torch.from_numpy(np.random.uniform(-4, 4, size=n_dim - 1).astype(np.float32))

            obs_mean = obs_data.mean(0)[:3]
            obs_var = obs_data.std(0)[:3]
            ex_y += [F(n_dim - 1, test_point.numpy().reshape(1, -1)).item()]

            xs = np.random.multivariate_normal(obs_mean.numpy(), obs_var.numpy() * np.eye(3), size=n_sample)

            do_m1_data = gen_data(n_sample, n_dim, {0: xs[:, 0]})
            do_m2_data = gen_data(n_sample, n_dim, {1: xs[:, 1]})
            do_m3_data = gen_data(n_sample, n_dim, {2: xs[:, 2]})

            int_data = {0: do_m1_data,
                        1: do_m2_data,
                        2: do_m3_data}

            model.train(obs_data, int_data, outer_n_iter=30, inner_n_iter=3000, lr=0.01)

            for f, k in zip(model.F, f_res.keys()):
                f_res[k] += [f.squeeze().detach().numpy().copy()]
            for cov, k in zip(model.int_covs.values(), int_cov_res.keys()):
                int_cov_res[k] += [cov.squeeze().detach().numpy().copy()]

            obs_cov_res['obs'] += [model.obs_cov.squeeze().detach().numpy().copy()]

            pred_y += [model.pred_c(model.nodes[-1], test_point.reshape(1, -1)).squeeze().detach().numpy()]

            pool_data = torch.cat(list(int_data.values()) + [obs_data])
            reg_model.train(pool_data)

            pred_reg += [reg_model.pred(test_point.reshape(1, -1)).squeeze().detach().numpy()]
            reg_coefs += [np.concatenate([np.array([reg_model.model.intercept_]), reg_model.model.coef_])]

            # clear param stores
            model.reset_params()

        for k in f_res.keys():
            f_res[k] = np.stack(f_res[k])
        for k in int_cov_res.keys():
            int_cov_res[k] = np.stack(int_cov_res[k])
        obs_cov_res['obs'] = np.stack(obs_cov_res['obs'])
        pred_y = np.stack(pred_y).squeeze()
        ex_y = np.stack(ex_y).squeeze()
        pred_reg = np.stack(pred_reg).squeeze()
        reg_coefs = np.stack(reg_coefs)

        # save the results
        with open("_synthetic_results/multiexp_lin_s{}_n{}.pickle".format(seed, n_sample), 'wb') as handle:
            pickle.dump([ex_y, pred_y, pred_reg, f_res, reg_coefs, int_cov_res, obs_cov_res], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
