import itertools
import sys

from tqdm import tqdm

from models import *
import pickle

from utils import get_gen_data, create_F

if __name__ == "__main__":

    try:
        # get an index as an argument
        process = int(sys.argv[1])
    except:
        process = -1

    seed = 2
    np.random.seed(seed)
    interaction = True
    intercept = False

    sample_size = [100, 400, 1600, 6400, 25600]
    # sample_size = [1600]
    if process == -1:
        pass
    else:
        idx = process % 5
        interaction = process > 4
        sample_size = [sample_size[idx]]
    n_rounds = 50
    n_dim = 4
    corr = np.array([[1, .3, .8, -.6], [.3, 1, .3, -.5], [.8, .3, 1, -.5], [-.6, -.5, -.5, 1]])
    diag_std = np.eye(4) * np.array([1., 1., 1., 1.])
    act_cov = diag_std.dot(corr).dot(diag_std)

    if interaction:
        coefs = [
            np.array([0.]),
            np.array([0, 1.]),
            np.array([0, 0.5, -1, 1]),
            np.array([0, 1.5, 1, -0.5, 0.5, -1, -1.5])
        ]
    else:
        coefs = [
            np.array([0.]),
            np.array([0, 1.]),
            np.array([0, 0.5, -1]),
            np.array([0, 1.5, 1, -0.5])
        ]

    F = create_F(coefs, interaction=interaction)
    gen_data = get_gen_data(act_cov, F)

    g = ig.Graph(directed=True)
    g.add_vertices(list(range(n_dim)))
    edges = list(itertools.combinations(range(n_dim), 2))
    g.add_edges(edges)

    model = JointInterventionModel('exp', g, interaction=interaction, intercept=intercept)
    reg_model = RegressionModel(g, interaction=interaction, intercept=intercept)
    reg_model_ind = RegressionModel(g, interaction=interaction, intercept=intercept)
    for n_sample in sample_size:
        f_res = {'m1': [], 'm2': [], 'm3': [], 'y': []}
        obs_cov_res = {'obs': [], 'int0': [], 'int1': [], 'int2': []}
        int_cov_res = {'m1': [], 'm2': [], 'm3': []}
        pred_y, pred_reg, reg_coefs, ex_y, reg_coefs_ind = [], [], [], [], []

        for idx in tqdm(range(n_rounds)):

            obs_data = gen_data(n_sample, n_dim)

            obs_mean = obs_data.mean(0)[:3]
            obs_var = obs_data.var(0)[:3]  # check if we should std or var

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

            # clear param stores
            model.reset_params()

            pool_data = torch.cat(list(int_data.values()) + [obs_data])
            reg_model.train(pool_data)

            reg_coefs += [np.concatenate([np.array([reg_model.model.intercept_]), reg_model.model.coef_])]

        for k in f_res.keys():
            f_res[k] = np.stack(f_res[k])

        pred_y = np.stack(pred_y).squeeze()
        ex_y = np.stack(ex_y).squeeze()
        pred_reg = np.stack(pred_reg).squeeze()
        reg_coefs = np.stack(reg_coefs)

        if intercept:
            save_ceof = coefs[-1].copy()
        else:
            save_ceof = coefs[-1][1:].copy()
            reg_coefs = reg_coefs[:, 1:]

        # save the results
        with open("_synthetic_results/reg_regime_inter{}_s{}_n{}.pickle".format(interaction, seed, n_sample),
                  'wb') as handle:
            pickle.dump([ex_y, pred_y, pred_reg, f_res, reg_coefs, save_ceof], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
