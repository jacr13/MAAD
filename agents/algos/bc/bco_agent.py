import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from gym import spaces
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from agents import get_kwarg
from agents.algos.base_agent import BaseAgent, get_local_rank
from agents.models.inv_dyn_model import get_idm
from agents.nets import CatPolicy, GaussPolicy
from agents.postprocessing import concatenate_rollouts, postprocess_rollout
from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.dataloader import DataLoader
from helpers.dataset import DatasetTorch
from helpers.distributed_util import (
    RunMoms,
    average_gradients,
    mpi_mean_reduce,
    sync_data,
    sync_weights,
)
from helpers.math_util import LRDecay, gradients_norm


class BCOAgent(BaseAgent):
    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        obs_type = "state"  # TODO: hardcoded
        self.use_encoder = True if obs_type == "pixels" else False
        self.timesteps_so_far = 0

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]

        self.device = "cpu"
        self._device = device

        self.hps = hps
        self.expert_dataset = expert_dataset

        self.simulator_steps = 0

        if self.hps.clip_norm <= 0:
            logger.info(
                "[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm),
            )

        # Create observation normalizer that maintains running statistics
        self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps, self.rms_obs).to(self.device)

        self.inv_dyn_model = get_idm(self.hps.idm_model)(
            self.ob_dim, self.ac_dim, self.hps, self.rms_obs, self.ob_shape
        )
        self._sync_weights(all_=True)
        self.trunk_target = deepcopy(self.policy.perception_stack)

        # Set up demonstrations dataset
        self.setup_expert_dataloader(
            self.expert_dataset, self.hps.batch_size, self.device
        )

        self.expert_true_actions = deepcopy(self.expert_dataset.data["acs"])

        # Set up the optimizers
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)

        self.lr_decay = LRDecay(self.hps, self.hps.p_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.p_optimizer,
            self.lr_decay,
        )

        log_module_info(logger, "policy", self.policy)

    def _sync_weights(
        self,
        all_=False,
        policy=False,
        inv_model=False,
    ):
        if all_ or policy:
            sync_weights(self.policy, msg="[policy]")

        if all_ or inv_model:
            sync_weights(self.inv_dyn_model, msg="[inv_dyn_model]")

    def to(self, device):
        if device != self.device:
            self.device = device
            self.policy = self.policy.to(device)
            self.inv_dyn_model = self.inv_dyn_model.to(device)

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, None, logp

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(
                torch.eq(
                    row,
                    torch.cat(
                        [
                            torch.zeros_like(row[0:-1]),
                            torch.Tensor([1.0]).to(self.device),
                        ],
                        dim=-1,
                    ),
                ),
            ):
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update_policy(self, p_rollout, iters_so_far):
        """Update the policy network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # INFO: This part was replaced by the post processing method
        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return gae(
        # p_rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='syn_rews')

        # Standardize advantage function estimate
        # given by p_rollout['advs'] = standardized(p_rollout['advs'])

        # Create DataLoader objects to iterate over transitions in rollouts
        p_keys = [
            "obs0",
            "obs1",
            "acs",
            "env_rews",
        ]
        p_dataset = DatasetTorch({k: p_rollout[k] for k in p_keys}, device=self.device)
        p_dataloader = DataLoader(
            p_dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        if not self.hps.idm_pretrained:
            logger.info("[INFO] updating inverse dynamic model")
            idm_metrics = self.inv_dyn_model.train_until_converged(p_dataloader)
            metrics.update(idm_metrics)

        actions_e_shape = self.expert_dataset.data.pop("acs", np.array([])).shape
        assert (
            np.size(actions_e_shape) != 0
        ), "expected expert actions shape to be non-empty"

        expert_obs_before = torch.Tensor(self.expert_dataset.data["obs0"]).view(
            -1, self.ob_dim
        )
        expert_obs_after = torch.Tensor(self.expert_dataset.data["obs1"]).view(
            -1, self.ob_dim
        )

        if self.hps.wrap_absorb:
            expert_obs_before = expert_obs_before[:, :-1]
            expert_obs_after = expert_obs_after[:, :-1]

        action_e, actions_e_dist = self.inv_dyn_model.predict(
            expert_obs_before, expert_obs_after, return_distribution=True
        )

        if self.hps.wrap_absorb:
            act_shape = list(action_e.shape)
            act_shape[-1] += 1
            actions_ = torch.zeros(act_shape)
            actions_[:, 0:-1] = action_e
            action_e = actions_

        self.expert_dataset.data["acs"] = (
            action_e.cpu().detach().numpy().reshape(*actions_e_shape)
        )
        self.setup_expert_dataloader(
            self.expert_dataset, self.hps.batch_size, self.device
        )

        # print((self.expert_true_actions - self.expert_dataset.data["acs"]).mean())

        logger.info("[INFO] updating policy")
        for _ in range(self.hps.optim_epochs_per_iter):
            for p_batch in p_dataloader:  # go through the whole p_rollout
                # Transfer to device
                state = p_batch["obs0"].to(self.device)
                action = p_batch["acs"].to(self.device)

                # Update the observation normalizer
                self.rms_obs.update(state)

                # Policy loss only using regularizer

                bc_loss = 0
                if self.hps.gail_bc_loss == "mse":
                    # Reparametrization trick
                    # reparam. trick -> x = mean + std * sample(N(0,1))
                    ac_sampled = self.policy.rsample(expert_obs_before)
                    bc_loss = F.mse_loss(ac_sampled, action_e)

                elif self.hps.gail_bc_loss == "nll":
                    log_probs = self.policy.logp(expert_obs_before, action_e)
                    bc_loss = -torch.logsumexp(log_probs, dim=1).mean()

                elif self.hps.gail_bc_loss.endswith("kl"):
                    policy_dist = self.policy.dist(expert_obs_before)

                    if self.hps.gail_bc_loss == "forward_kl":
                        # sample action from inverse dynamics model = action_e
                        ac_sampled = action_e

                        dist_a = actions_e_dist
                        dist_b = policy_dist

                        logp_a = dist_a.log_prob(ac_sampled).detach()
                        logp_b = dist_b.log_prob(ac_sampled)

                    elif self.hps.gail_bc_loss == "reverse_kl":
                        # sample action from policy
                        ac_sampled = policy_dist.sample()

                        dist_a = policy_dist
                        dist_b = actions_e_dist

                        logp_a = dist_a.log_prob(ac_sampled)
                        logp_b = dist_b.log_prob(ac_sampled).detach()
                    else:
                        raise ValueError(
                            f"Gail BC loss '{self.hps.gail_bc_loss}' is not supported"
                        )

                    # compute estimated KL divergence
                    kl = (logp_a - logp_b).mean()

                    # compute rescaling
                    if hasattr(dist_b, "base_dist"):
                        scale = dist_b.base_dist.scale
                    else:
                        scale = dist_b._component_distribution.base_dist.scale

                    if hasattr(dist_a, "base_dist"):
                        scale_a = dist_a.base_dist.scale
                    else:
                        scale_a = dist_a._component_distribution.base_dist.scale

                    weight_kl = (2 * scale.detach() ** 2).mean()
                    # bc_loss += weight_kl * kl
                    bc_loss = kl
                    if self.hps.gail_bc_loss == "reverse_kl":
                        bc_loss = (
                            (
                                scale.log()
                                - scale_a.log()
                                + (scale_a.pow(2) + (dist_a.mean - dist_b.mean).pow(2))
                                / (2.0 * scale.pow(2))
                                - 0.5
                            )
                            .sum(dim=-1, keepdim=True)
                            .mean()
                        )
                        bc_loss *= weight_kl

                else:
                    raise ValueError(
                        f"Gail-BC loss '{self.hps.gail_bc_loss}' not supported."
                    )

                metrics["regularization/bc_loss"].append(bc_loss.item())
                metrics["regularization/actions_true_vs_pred_error"].append(
                    (self.expert_true_actions - self.expert_dataset.data["acs"]).mean()
                )
                metrics["regularization/regularized_bc_loss"].append(
                    self.hps.gail_bc_lambda * bc_loss.item()
                )
                metrics["regularization/bc_weight"].append(self.hps.gail_bc_lambda)

                # add bc loss to policy loss
                final_loss = self.hps.gail_bc_lambda * bc_loss

                # Log metrics
                # rewards
                metrics["batch_reward_env"].append(p_batch["env_rews"].mean().item())

                # actor
                metrics["actor_loss"].append(final_loss.item())

                # Update parameters
                self.p_optimizer.zero_grad()
                final_loss.backward()

                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)

                self.p_optimizer.step()

        self.scheduler.step()

        metrics = self.process_metrics(metrics)
        return metrics, self.scheduler.get_last_lr()

    def rollout_generator(self, rollout_len=None):
        rollout_len = rollout_len if rollout_len else self.hps.rollout_len

        self.simulator_steps = 0
        done = True
        # Reset agent's env
        ob = np.array(self.env.reset())

        # Init collections
        p_rollout = defaultdict(list)
        # Init current episode statistics
        cur_ep_len = 0
        cur_ep_env_ret = 0

        if self.is_discrete:
            ac_dim_ = 1
        else:
            ac_dim_ = self.ac_dim

        while True:
            # Predict
            ac, _, logp = self.predict(ob, True)

            if not isinstance(self.ac_space, spaces.Discrete):
                # NaN-proof and clip
                ac = np.nan_to_num(ac)
                ac = np.clip(ac, self.env.action_space.low, self.env.action_space.high)
            else:
                ac = ac if isinstance(ac, int) else ac.item()

            # (t > 0 and t % rollout_len == 0)
            if self.simulator_steps > 0 and self.simulator_steps % rollout_len == 0:
                for k in p_rollout.keys():
                    if k in ["obs0", "obs1"]:
                        p_rollout[k] = np.array(p_rollout[k]).reshape(
                            -1, *self.ob_shape
                        )
                    elif k == "acs":
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, ac_dim_)
                    elif k in ["vs", "logps", "env_rews", "syn_rews", "dones"]:
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, 1)
                    else:
                        p_rollout[k] = np.array(p_rollout[k])

                if self.hps.wrap_absorb:
                    ob_dim = self.ob_dim + 1
                    ac_dim = ac_dim_ + 1
                else:
                    ob_dim = self.ob_dim
                    ac_dim = ac_dim_
                yield p_rollout

                # Clear the collections
                p_rollout.clear()
                self.simulator_steps = 0

            # Interact with env(s)
            step = self.env.step(ac)
            if len(step) == 4:
                new_ob, env_rew, done, _ = step
            elif len(step) == 5:
                new_ob, env_rew, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                raise ValueError(f"unexpected step length: {len(step)}")
            self.simulator_steps += 1

            # Populate collections
            p_rollout["obs0"].append(ob)
            p_rollout["acs"].append(ac)
            p_rollout["obs1"].append(new_ob)
            p_rollout["logps"].append(logp)
            p_rollout["env_rews"].append(env_rew)
            p_rollout["dones"].append(done)

            # Update current episode statistics
            cur_ep_len += 1
            assert self.env._elapsed_steps == cur_ep_len  # sanity check
            cur_ep_env_ret += env_rew

            # Set current state with the next
            ob = np.array(deepcopy(new_ob))

            if done:
                # Update the global episodic statistics and
                # reset current episode statistics
                p_rollout["ep_lens"].append(cur_ep_len)
                cur_ep_len = 0
                p_rollout["ep_env_rets"].append(cur_ep_env_ret)
                cur_ep_env_ret = 0
                # Reset env
                ob = np.array(self.env.reset())

    def do_iteration(self, **kwargs):
        timed = get_kwarg(kwargs, "timed")
        roll_gen = get_kwarg(kwargs, "roll_gen")
        iters_so_far = get_kwarg(kwargs, "iters_so_far")
        d = get_kwarg(kwargs, "d")

        if self.hps.rot_auto_rew_scale and iters_so_far == 0:
            self.hps.rot_sinkhorn_rew_scale = 1.0

        for _ in range(self.hps.g_steps):
            with timed("interacting"):
                self.to("cpu")

                # Unpack (one rollout dict for policy training, one for reward training)
                p_rollout = roll_gen.__next__()

                d["roll_len"].append(p_rollout["ep_lens"].mean())
                roll_env_ret = p_rollout["ep_env_rets"].mean()
                d["roll_env_ret"].append(roll_env_ret)

                if self.hps.trainer_worker:
                    rollouts = sync_data({"p_rollout": p_rollout})

            with timed("policy training"):
                if self.hps.master and self.hps.trainer_worker:
                    p_rollout = concatenate_rollouts(rollouts, key="p_rollout")

                if self.hps.master:
                    self.to(self._device)
                    metrics, lrnow = self.update_policy(
                        p_rollout=p_rollout,
                        iters_so_far=iters_so_far,
                    )
                    self.to("cpu")

                    d["lrnow"].append(lrnow)
                    d.update(metrics)

        self.log_model_stats()

    def update_timesteps(self, timesteps_so_far):
        return timesteps_so_far + (self.hps.g_steps * self.hps.rollout_len)
