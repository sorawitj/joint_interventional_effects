import pickle
import sys

from tqdm import tqdm

from model import *

if __name__ == "__main__":
    try:
        # get an index as an argument
        round = int(sys.argv[1])
    except:
        round = -1

    seed = 2
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # sample_size = [100, 400, 1600, 6400]
    sample_size = [1600]
    conf_level = [0.1, 0.35, 0.6, 0.85]
    n_round = 50
    # conf_level = 1
    intercept = False
    interaction = True
    boostrap = True

    if round == -1:
        pass
    else:
        # sample_size = [sample_size[round]]
        conf_level = conf_level[round]

    data_path = '_realdata/train_test_s{}_conf{}.pickle'.format(seed, conf_level)
    obs_data_full, test_data, linearSCM, G = pickle.load(open(data_path, 'rb'))

    mae_best = linearSCM.evaluate(test_data)

    mae_scm, mae_pool, mae_obs = [], [], []
    f_scm, f_pool, f_obs = [], [], []

    for n_sample in sample_size:
        obs_data_n = obs_data_full[:n_sample, :]
        int_data_n = {}

        for node in G.topological_sorting()[:-1]:
            dat = obs_data_n[:, node]
            mean, std = dat.mean(), dat.std()
            intervention = {node: torch.randn(n_sample) * std + mean}
            int_data_n[node] = linearSCM.sample(n_sample, intervention)

        for _ in tqdm(range(n_round)):
            if boostrap:
                # bootstrap index
                s_idx = torch.randint(high=n_sample, size=(n_sample,), dtype=torch.int64).tolist()
                obs_data = obs_data_n[s_idx]
                int_data = {k: v[s_idx] for k, v in int_data_n.items()}
            else:
                obs_data = obs_data_n
                int_data = int_data_n

            model_scm = JointInterventionModel('realexp', G,
                                               interaction=interaction, intercept=intercept)
            model_scm.train(obs_data, int_data, outer_n_iter=30, inner_n_iter=3000, lr=0.01)

            reg_model = RegressionModel(G, intercept=intercept, interaction=interaction)
            reg_model_obs = RegressionModel(G, intercept=intercept, interaction=interaction)
            pool_data = torch.cat(list(int_data.values()) + [obs_data])
            reg_model.train(pool_data)
            reg_model_obs.train(obs_data)

            mae_scm += [model_scm.evaluate(test_data)]
            mae_pool += [reg_model.evaluate(test_data)]
            mae_obs += [reg_model_obs.evaluate(test_data)]

            f_scm += [model_scm.F[model_scm.nodes[-1]].detach().numpy()]
            f_pool += [reg_model.model.coef_]
            f_obs += [reg_model_obs.model.coef_]

            model_scm.reset_params()

        mae_scm = np.stack(mae_scm)
        mae_pool = np.stack(mae_pool)
        mae_obs = np.stack(mae_obs)
        f_scm = np.stack(f_scm)
        f_pool = np.stack(f_pool)
        f_obs = np.stack(f_obs)

        # save the results
        with open("_realdata_results/nonlin3_r50_s{}_n{}_conf{}.pickle".format(seed, n_sample, conf_level),
                  'wb') as handle:
            pickle.dump([mae_scm, mae_pool, mae_obs, mae_best,
                         f_scm, f_pool, f_obs, linearSCM], handle, protocol=pickle.HIGHEST_PROTOCOL)
