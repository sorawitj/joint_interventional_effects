import numpy as np
import warnings

import pyro
import torch
import pyro.distributions as dist
from sklearn.preprocessing import PolynomialFeatures
from torch.optim import Adam
from pyro.poutine import trace
from utils import EarlyStopping, to_torch, add_interaction
import igraph as ig
import os
from sklearn.linear_model import LinearRegression
from math import factorial
from scipy.stats import spearmanr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.simplefilter(action='ignore', category=FutureWarning)


class SCM(object):

    def __init__(self, name, ig: ig.Graph, interaction=False, intercept=True):
        self.G = ig
        self.nodes = self.G.topological_sorting()
        self.node_size = len(self.nodes)
        self.F = [torch.empty(0)] * self.node_size
        self.parents = [None] * self.node_size
        self.name = name
        self.interaction = interaction
        self.intercept = intercept

    def initialize_param(self, name):
        for node in self.nodes:
            parent = self.G.predecessors(node)
            self.parents[node] = parent
            n_parents = len(parent)
            n_int_terms = 0
            intercept_term = 0
            if self.interaction:
                if n_parents > 1:
                    n_int_terms = int(factorial(n_parents) / (factorial(2) * factorial(n_parents - 2)))
            if self.intercept or (len(parent) == 0):
                intercept_term = 1

            self.F[node] = pyro.param("{}_f{}".format(name, node),
                                      torch.randn(n_parents + n_int_terms + intercept_term, 1))

    def pred_c(self, node, data):
        parent = self.parents[node]
        params = self.F[node]
        if len(parent) == 0:
            res = torch.ones_like(data[:, [0]]) * params
        else:
            if self.interaction:
                x = add_interaction(data[:, parent])
            else:
                x = data[:, parent]
            if self.intercept:
                res = params[0] + x.mm(params[1:])
            else:
                res = x.mm(params)
        return res

    def llh_model(self, data, cov=None):
        pass

    def get_neg_llh(self, data, cov=None):
        return -trace(self.llh_model).get_trace(data, cov).log_prob_sum() / data.shape[0]

    def evaluate(self, test_data):
        target_node = self.nodes[-1]
        pred = self.pred_c(target_node, test_data).squeeze().detach().numpy()
        act = test_data[:, target_node].detach().numpy()
        error = pred - act
        mae = np.mean(np.abs(error))
        spearman = spearmanr(pred, act).correlation
        return np.array([mae, spearman])

    def reset_params(self):
        pyro.clear_param_store()
        self.initialize_param(self.name)


class LinearSCM(SCM):

    def __init__(self, name, ig: ig.Graph, interaction=False, intercept=True, corr_matrix=None):

        super().__init__(name, ig, interaction, intercept)
        self.log_cov = pyro.param("cov_" + str(name), torch.randn(self.node_size, 1) * .5)
        if corr_matrix is None:
            self.corr_matrix = torch.eye(self.node_size)
        else:
            self.corr_matrix = corr_matrix
        self.initialize_param(name)

    def cov(self):
        diag_std = torch.eye(self.node_size) * self.diag_std
        cov = diag_std.mm(self.corr_matrix).mm(diag_std)
        # cov = self.corr_matrix
        return cov

    def llh_model(self, data, diag_std=None):
        preds = [torch.empty(0)] * self.node_size

        for node in self.nodes:
            preds[node] = self.pred_c(node, data)
        mu = torch.cat(preds, dim=1)

        pyro.sample('obs', dist.MultivariateNormal(loc=torch.zeros(self.node_size),
                                                   covariance_matrix=torch.eye(self.node_size) * self.cov()),
                    obs=data - mu)

        return mu

    def train(self, obs_data, lr=0.005, n_iter=5000):
        params = self.F
        # params += [self.log_cov]
        optimizer = Adam(params, lr=lr, betas=(0.95, 0.999))
        early_stop = EarlyStopping(delta=0.000001, patience=5)
        self.diag_std = torch.ones(self.node_size)
        for i in range(20):
            for _ in range(n_iter):
                optimizer.zero_grad()
                obs_probs = self.get_neg_llh(obs_data, self.diag_std)
                obs_probs.backward()
                optimizer.step()

                curr_loss = obs_probs.item()
                early_stop(curr_loss)
                if early_stop.early_stop:
                    break

            mu_obs = self.llh_model(obs_data, self.diag_std)
            u_obs = mu_obs - obs_data
            self.diag_std = torch.from_numpy(np.std(u_obs.detach().numpy(), axis=1).astype(np.float32))

        print(self.F)

    def sample(self, sample_size, interventions: dict = None):
        with torch.no_grad():
            if interventions is None:
                interventions = {}
            samples = torch.empty((sample_size, self.node_size))
            noise_dist = dist.MultivariateNormal(torch.zeros(self.node_size), self.cov())
            U = noise_dist.sample((sample_size,))

            for node in self.nodes:
                if node in interventions.keys():
                    samples[:, node] = interventions[node]
                    continue
                samples[:, node] = self.pred_c(node, samples).squeeze() + U[:, node]
                # parent = self.parents[node]
                # if len(parent) == 0:
                noise_std = torch.sqrt(self.cov()[node, node])
                # samples[:, node] += U[:, node]
                # adjust for variance
                # samples[:, node] *= noise_std / torch.std(samples[:, node])

        return samples


class JointInterventionModel(SCM):

    def __init__(self, name, ig: ig.Graph, interaction=False, intercept=True):
        super().__init__(name, ig, interaction, intercept)
        self.initialize_param(name)

    def llh_model(self, data, cov=None):
        preds = [torch.empty(0)] * self.node_size

        for node in self.nodes:
            preds[node] = self.pred_c(node, data)

        mu = torch.cat(preds, dim=1)
        pyro.sample('obs', dist.MultivariateNormal(loc=mu, covariance_matrix=cov),
                    obs=data)

        return mu

    def train(self, obs_data: torch.tensor, int_data: dict, lr=0.005, lr_damp=0.99, outer_n_iter=20, inner_n_iter=2000):
        obs_cov = torch.eye(self.node_size)
        int_covs = {node: torch.eye(self.node_size) for node in self.nodes[:-1]}

        for _ in range(outer_n_iter):
            lr *= lr_damp
            optimizers = [Adam([f], lr=lr, betas=(0.95, 0.999)) for f in self.F]
            early_stops = [EarlyStopping(delta=0.000001, patience=5) for _ in range(len(int_data) + 1)]
            for _ in range(inner_n_iter):
                obs_probs = self.get_neg_llh(obs_data, obs_cov)
                int_probs = {node: self.get_neg_llh(int_data[node], int_covs[node]) for node in int_data.keys()}

                for node in self.nodes:
                    retain_graph = False if node == self.nodes[-1] else True
                    optimizer = optimizers[node]
                    optimizer.zero_grad()
                    obs_probs.backward(retain_graph=retain_graph)
                    for int_node in int_data.keys():
                        if len(self.parents[node]) != 0:
                            if node == int_node:
                                continue
                        int_probs[int_node].backward(retain_graph=retain_graph)
                    optimizer.step()

                curr_losses = [prob.item() for prob in int_probs.values()]
                curr_losses += [obs_probs.item()]

                [early_stop(curr_loss) for early_stop, curr_loss in zip(early_stops, curr_losses)]
                should_stop = [early_stop.early_stop for early_stop in early_stops]
                if all(should_stop):
                    break

            mu_obs = self.llh_model(obs_data, obs_cov)
            u_obs = mu_obs - obs_data
            obs_cov = torch.from_numpy(np.cov(u_obs.detach().numpy().T).astype(np.float32))

            for node in int_data.keys():
                mu_int = self.llh_model(int_data[node], int_covs[node])
                u_int = mu_int - int_data[node]
                int_cov = torch.from_numpy(np.cov(u_int.detach().numpy().T).astype(np.float32))
                for j in range(self.node_size):
                    if j != node:
                        int_cov[[node, j], [j, node]] = 0

                int_covs[node] = int_cov

            self.int_covs = int_covs
            self.obs_cov = obs_cov

        print(self.F)


class RegressionModel(object):

    def __init__(self, ig: ig.Graph, interaction=False, intercept=True):
        self.ig = ig
        self.nodes = ig.topological_sorting()
        self.colnames = ig.vs['name']
        self.target = ig.topological_sorting()[-1]
        self.predictors = [node for node in ig.vs.indices if node != self.target]
        self.poly = PolynomialFeatures(interaction_only=True, degree=2, include_bias=False)
        self.interaction = interaction
        self.intercept = intercept

    def train(self, tensor):
        X = tensor[:, self.predictors]
        y = tensor[:, self.target]
        if self.interaction:
            X = self.poly.fit_transform(X)
        self.model = LinearRegression(fit_intercept=self.intercept).fit(X, y)

    def pred(self, tensor):
        X = tensor[:, self.predictors]
        if self.interaction:
            X = self.poly.fit_transform(X)
        y_pred = to_torch(self.model.predict(X))

        return y_pred

    def evaluate(self, test_data):
        target_node = self.nodes[-1]
        pred = self.pred(test_data).squeeze().detach().numpy()
        act = test_data[:, target_node].detach().numpy()
        error = pred - act
        mae = np.mean(np.abs(error))
        spearman = spearmanr(pred, act).correlation
        return np.array([mae, spearman])


if __name__ == "__main__":
    pass
    # np.random.seed(2)
    # n = 30000
    # m1, m2, m3, y = gen_data3(n)
    # x1 = np.random.randn(n)
    # x2 = np.random.randn(n) * 2
    # x3 = np.random.randn(n) * 2
    # do_m1, m2_dom1, m3_dom1, y_dom1 = gen_data3(n, x1=x1)
    # m1_dom2, do_m2, m3_dom2, y_dom2 = gen_data3(n, x2=x2)
    # m1_dom3, m2_dom3, do_m3, y_dom3 = gen_data3(n, x3=x3)
    #
    # g = ig.Graph(directed=True)
    # g.add_vertices([0, 1, 2, 3])
    # g.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    # model = JointInterventionModel('test', g)
    # obs_data = torch.cat([m1, m2, m3, y], dim=1)
    # int_data = {1: torch.cat([m1_dom2, do_m2, m3_dom2, y_dom2], dim=1),
    #             0: torch.cat([do_m1, m2_dom1, m3_dom1, y_dom1], dim=1),
    #             2: torch.cat([m1_dom3, m2_dom3, do_m3, y_dom3], dim=1)}
    # model.train(obs_data, int_data)
