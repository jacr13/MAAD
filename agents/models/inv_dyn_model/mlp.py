import math

import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim

from agents.models.inv_dyn_model.base import InvDynModelBase
from helpers.config_helper import Config
from helpers.distributed_util import RunMoms

STANDARDIZED_OB_CLAMPS = [-5.0, 5.0]


class InvDynModelMLP(InvDynModelBase):
    def __init__(
        self,
        obs_size: int,
        actions_size: int,
        hps: Config,
        rms_obs: RunMoms,
        obs_shape: tuple = None,
    ):
        super(InvDynModelMLP, self).__init__(
            obs_size, actions_size, hps, rms_obs, obs_shape
        )
        self.obs_size = obs_size
        self.actions_size = actions_size
        self.hps = hps

        self.layers = nn.Sequential(
            nn.Linear(obs_size * 2, self.hps.idm_hidsize),
            nn.ReLU(),
            nn.Linear(self.hps.idm_hidsize, self.hps.idm_hidsize),
            nn.ReLU(),
            nn.Linear(self.hps.idm_hidsize, actions_size),
        )

        self.ac_logstd_head = nn.Parameter(torch.full((actions_size,), math.log(0.01)))

        self.optimizer = optim.Adam(self.layers.parameters(), lr=self.hps.idm_lr)
        # self.loss_fn = nn.MSELoss()

        if self.hps.idm_pretrained:
            self.load()

    def loss_fn(self, ac_dist, actions=None):
        if self.hps.idm_loss == "mse":
            assert actions is not None, "actions should not be None"
            acions_pred = ac_dist.rsample()
            loss = nn.functional.mse_loss(acions_pred, actions)
            return loss
        else:
            raise ValueError(f"Unknown loss type for {self.hps.idm_loss}")

    def rsample(self, obs_before, obs_after):
        return self.forward(obs_before, obs_after).rsample()

    def sample(self, obs_before, obs_after, return_distribution=False):
        ac_dist = self.forward(obs_before, obs_after)
        actions = ac_dist.sample()
        if return_distribution:
            return actions, ac_dist
        return actions

    def forward(self, obs_before, obs_after):
        if self.hps.p_batch_norm:
            obs_before = self.rms_obs.standardize(obs_before).clamp(
                *STANDARDIZED_OB_CLAMPS
            )
            obs_after = self.rms_obs.standardize(obs_after).clamp(
                *STANDARDIZED_OB_CLAMPS
            )
        ac_mu = self.layers(torch.cat([obs_before, obs_after], dim=1))
        ac_std = self.ac_logstd_head.expand_as(ac_mu).exp()
        ac_dist = td.Independent(td.Normal(ac_mu, ac_std), 1)
        return ac_dist
