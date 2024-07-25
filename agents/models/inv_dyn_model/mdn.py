import math
from typing import Optional, Tuple, Union

import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim

from agents.models.inv_dyn_model.base import InvDynModelBase
from helpers.config_helper import Config
from helpers.distributed_util import RunMoms

STANDARDIZED_OB_CLAMPS = [-5.0, 5.0]


class InvDynModelMDN(InvDynModelBase):
    def __init__(
        self,
        obs_size: int,
        actions_size: int,
        hps: Config,
        rms_obs: RunMoms,
        obs_shape: tuple = None,
    ):
        super(InvDynModelMDN, self).__init__(
            obs_size, actions_size, hps, rms_obs, obs_shape
        )
        self.hps = hps
        self.device = self.hps.device
        self.obs_size = obs_size
        self.actions_size = actions_size

        self.in_dim = 2 * obs_size
        self.out_dim = self.hps.idm_mdn_n_gaussians * actions_size

        self.pi = nn.Sequential(
            nn.Linear(self.in_dim, self.hps.idm_hidsize),
            nn.ELU(),
            nn.Linear(self.hps.idm_hidsize, self.hps.idm_mdn_n_gaussians),
        )
        self.mu = nn.Sequential(
            nn.Linear(self.in_dim, self.hps.idm_hidsize),
            nn.ELU(),
            nn.Linear(self.hps.idm_hidsize, self.out_dim),
        )
        self.sigma = self.ac_logstd_head = nn.Parameter(
            torch.full((self.out_dim,), math.log(0.01))
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.hps.idm_lr)

        if self.hps.idm_pretrained:
            self.load()

    def sample(
        self,
        obs_before: torch.Tensor,
        obs_after: torch.Tensor,
        return_distribution: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, td.MixtureSameFamily]]:
        dist = self.forward(obs_before, obs_after)
        samples = dist.sample()
        if return_distribution:
            return samples, dist
        return samples

    def loss_fn(
        self,
        dist: td.MixtureSameFamily,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert actions is not None, "actions should not be None"
        ac_dist = dist.component_distribution.base_dist
        pi = dist.mixture_distribution

        actions = actions.unsqueeze(1).expand_as(ac_dist.loc)

        log_probs = ac_dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=2)

        # use pi.logits directly instead of torch.log(pi.probs) to
        # avoid numerical problem
        nll = -torch.logsumexp(pi.logits + log_probs, dim=1)
        return nll.mean()

    def forward(
        self, obs_before: torch.Tensor, obs_after: torch.Tensor
    ) -> td.MixtureSameFamily:
        if self.hps.p_batch_norm:
            obs_before = self.rms_obs.standardize(obs_before).clamp(
                *STANDARDIZED_OB_CLAMPS
            )
            obs_after = self.rms_obs.standardize(obs_after).clamp(
                *STANDARDIZED_OB_CLAMPS
            )

        x = torch.cat([obs_before, obs_after], dim=1)

        # pi
        pi = td.Categorical(logits=self.pi(x))

        # normal distribution for actions
        mu = self.mu(x).view(-1, self.hps.idm_mdn_n_gaussians, self.actions_size)
        sigma = (
            self.ac_logstd_head.view(
                -1, self.hps.idm_mdn_n_gaussians, self.actions_size
            )
            .expand_as(mu)
            .exp()
        )
        ac_dist = td.Normal(mu, sigma)
        return td.MixtureSameFamily(pi, td.Independent(ac_dist, 1))
