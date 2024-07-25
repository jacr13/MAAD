import copy

import numpy as np
import ot
import torch
from sklearn import preprocessing

# From https://github.com/siddhanthaldar/ROT


def optimal_transport_plan(
    X, Y, cost_matrix, method="sinkhorn_gpu", niter=500, epsilon=0.01
):
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan


def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = 1 - C / norms
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c


class PWILRewarder(object):
    """Rewarder class to compute PWIL rewards.
    Taken from: https://github.com/google-research/google-research/blob/master/pwil/rewarder.py
    With minimal changes to fit our code.
    """

    def __init__(
        self,
        demonstrations,
        time_horizon=1000.0,
        alpha=5.0,
        beta=5.0,
        observation_only=True,
    ):
        """Initialize the rewarder.

        Args:
          demonstrations: a dataset of expert demonstrations.
          time_horizon: int time length of the task.
          alpha: float scaling the reward function.
          beta: float controling the kernel size of the reward function.
          observation_only: boolean whether or not to use action to compute reward.
        """
        self.time_horizon = time_horizon

        obs_dim = demonstrations.data["obs0"].shape[-1]
        acs_dim = demonstrations.data["acs"].shape[-1]

        # Observations and actions are flat.
        self.reward_sigma = beta * time_horizon / np.sqrt(acs_dim + obs_dim)
        self.reward_scale = alpha

        self.observation_only = observation_only
        self.demonstrations = self.filter_demonstrations(demonstrations)
        self.vectorized_demonstrations = self.vectorize(self.demonstrations)
        self.scaler = self.get_scaler()

    def filter_demonstrations(self, demonstrations):
        """Select a subset of expert demonstrations.

        Args:
          demonstrations: list of expert demonstrations

        Returns:
          filtered_demonstrations: list of filtered expert demonstrations
        """
        filtered_demonstrations = []
        for i in range(len(demonstrations.data["obs0"])):
            obs = demonstrations.data["obs0"][i].cpu().numpy()
            acs = demonstrations.data["acs"][i].cpu().numpy()

            episode = [
                {"observation": obs, "action": acs} for obs, acs in zip(obs, acs)
            ]
            filtered_demonstrations += episode
        return filtered_demonstrations

    def vectorize(self, demonstrations):
        """Convert filtered expert demonstrations to numpy array.

        Args:
          demonstrations: list of expert demonstrations

        Returns:
          numpy array with dimension:
          [num_expert_transitions, dim_observation] if observation_only
          [num_expert_transitions, (dim_observation + dim_action)] otherwise
        """

        if self.observation_only:
            demonstrations = [t["observation"] for t in demonstrations]
        else:
            demonstrations = [
                np.concatenate([t["observation"], t["action"]]) for t in demonstrations
            ]
        return np.array(demonstrations)

    def get_scaler(self):
        """Defines a scaler to derive the standardized Euclidean distance."""
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.vectorized_demonstrations)
        return scaler

    def reset(self):
        """Makes all expert transitions available and initialize weights."""
        self.expert_atoms = copy.deepcopy(
            self.scaler.transform(self.vectorized_demonstrations)
        )
        num_expert_atoms = len(self.expert_atoms)
        self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)

    def compute_reward(self, obs_act):
        """Computes reward as presented in Algorithm 1."""
        # Scale observation and action.
        if self.observation_only:
            agent_atom = obs_act["observation"]
        else:
            agent_atom = np.concatenate([obs_act["observation"], obs_act["action"]])
        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
        agent_atom = self.scaler.transform(agent_atom)[0]

        cost = 0.0
        # As we match the expert's weights with the agent's weights, we might
        # raise an error due to float precision, we substract a small epsilon from
        # the agent's weights to prevent that.
        weight = 1.0 / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)
        while weight > 0:
            # Get closest expert state action to agent's state action.
            argmin = norms.argmin()
            expert_weight = self.expert_weights[argmin]

            # Update cost and weights.
            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights = np.delete(self.expert_weights, argmin, 0)
                self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[argmin] -= weight
                weight = 0

        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        return reward.astype("float32")

    def compute_w2_dist_to_expert(self, trajectory):
        """Computes Wasserstein 2 distance to expert demonstrations."""
        self.reset()
        if self.observation_only:
            trajectory = [t["observation"] for t in trajectory]
        else:
            trajectory = [
                np.concatenate([t["observation"], t["action"]]) for t in trajectory
            ]

        trajectory = self.scaler.transform(trajectory)
        trajectory_weights = 1.0 / len(trajectory) * np.ones(len(trajectory))
        cost_matrix = ot.dist(trajectory, self.expert_atoms, metric="euclidean")
        w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
        return w2_dist
