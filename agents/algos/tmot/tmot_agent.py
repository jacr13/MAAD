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
from agents.nets import CatPolicy, GaussPolicy, Value
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
from helpers.rewarder import (
    PWILRewarder,
    cosine_distance,
    euclidean_distance,
    optimal_transport_plan,
)


class TMOTAgent(BaseAgent):
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

        assert self.hps.shared_value or not self.hps.shared_value_policy_detached

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps, self.rms_obs).to(self.device)

        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps, self.rms_obs).to(self.device)

        if self.hps.algo in ["maad-ot"]:
            # self.debug_true_actions = self.expert_dataset.data.pop("acs", None)
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
        if not self.hps.shared_value:
            self.v_optimizer = torch.optim.Adam(
                self.value.parameters(),
                lr=self.hps.v_lr,
            )

        self.lr_decay = LRDecay(self.hps, self.hps.p_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.p_optimizer,
            self.lr_decay,
        )

        log_module_info(logger, "policy", self.policy)
        if not self.hps.shared_value:
            log_module_info(logger, "value", self.value)

        if self.hps.ot_rewards == "pwil":
            self.pwil_rewarder = PWILRewarder(self.expert_dataset)

    def _sync_weights(
        self,
        all_=False,
        policy=False,
        value=False,
        inv_model=False,
    ):
        if all_ or policy:
            sync_weights(self.policy, msg="[policy]")

        if not self.hps.shared_value:
            if all_ or value:
                sync_weights(self.value, msg="[value]")

        if self.hps.algo in ["maad-ot"]:
            if all_ or inv_model:
                sync_weights(self.inv_dyn_model, msg="[inv_dyn_model]")

    def to(self, device):
        if device != self.device:
            self.device = device
            self.policy = self.policy.to(device)
            if not self.hps.shared_value:
                self.value = self.value.to(device)
            if self.hps.algo in ["maad-ot"]:
                self.inv_dyn_model = self.inv_dyn_model.to(device)

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        if self.hps.shared_value:
            v = self.policy.value(ob).cpu().detach().numpy().flatten()
        else:
            v = self.value(ob).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, v, logp

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

    def update_policy_value(self, p_rollout, iters_so_far):
        """Update the policy and value networks"""

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
            "logps",
            "vs",
            "advs",
            "td_lam_rets",
            "env_rews",
            "syn_rews",
        ]
        p_dataset = DatasetTorch({k: p_rollout[k] for k in p_keys}, device=self.device)
        p_dataloader = DataLoader(
            p_dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        if self.hps.algo in ["maad-ot"]:
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
                logp_old = p_batch["logps"].to(self.device)
                v_old = p_batch["vs"].to(self.device)
                advantage = p_batch["advs"].to(self.device)
                td_lam_return = p_batch["td_lam_rets"].to(self.device)

                # Update the observation normalizer
                self.rms_obs.update(state)

                # Policy loss
                entropy_loss = (
                    -self.hps.p_ent_reg_scale * self.policy.entropy(state).mean()
                )
                logp = self.policy.logp(state, action)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(
                    1.0 - self.hps.eps,
                    1.0 + self.hps.eps,
                )
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = 0.5 * (logp - logp_old).pow(2).mean()
                kl_max = 0.5 * (logp - logp_old).pow(2).max()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                # Value loss
                if self.hps.shared_value:
                    v = self.policy.value(state)
                else:
                    v = self.value(state)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                v_loss_a = (clip_v - td_lam_return).pow(2)
                v_loss_b = (v - td_lam_return).pow(2)
                v_loss = torch.max(v_loss_a, v_loss_b).mean()
                if self.hps.shared_value:
                    p_loss = (
                        clip_loss + entropy_loss + (self.hps.baseline_scale * v_loss)
                    )
                else:
                    p_loss = clip_loss + entropy_loss

                # REG
                if self.hps.algo in ["maad-ot"]:
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
                        metrics["regularization/weight_kl"].append(weight_kl.item())
                        bc_loss = weight_kl * kl
                        # bc_loss = kl
                        # if self.hps.gail_bc_loss == "reverse_kl":
                        #     bc_loss = (
                        #         (
                        #             scale.log()
                        #             - scale_a.log()
                        #             + (
                        #                 scale_a.pow(2)
                        #                 + (dist_a.mean - dist_b.mean).pow(2)
                        #             )
                        #             / (2.0 * scale.pow(2))
                        #             - 0.5
                        #         )
                        #         .sum(dim=-1, keepdim=True)
                        #         .mean()
                        #     )
                        #     bc_loss *= weight_kl

                    else:
                        raise ValueError(
                            f"Gail-BC loss '{self.hps.gail_bc_loss}' not supported."
                        )

                    metrics["regularization/bc_loss"].append(bc_loss.item())
                    metrics["regularization/actions_true_vs_pred_error"].append(
                        (
                            self.expert_true_actions - self.expert_dataset.data["acs"]
                        ).mean()
                    )
                    metrics["regularization/regularized_bc_loss"].append(
                        self.hps.gail_bc_lambda * bc_loss.item()
                    )
                    metrics["regularization/bc_weight"].append(self.hps.gail_bc_lambda)

                    # add bc loss to policy loss
                    final_loss = p_loss + self.hps.gail_bc_lambda * bc_loss
                else:
                    final_loss = p_loss

                # Log metrics
                # rewards
                metrics["batch_reward_env"].append(p_batch["env_rews"].mean().item())
                metrics["batch_reward_syn"].append(p_batch["syn_rews"].mean().item())

                # critic
                metrics["critic_target_q"].append(td_lam_return.mean().item())
                metrics["critic_q"].append(v.mean().item())
                metrics["critic_loss"].append(v_loss.item())

                # actor
                metrics["actor_loss"].append(final_loss.item())
                metrics["actor_logprob"].append(logp.mean().item())
                metrics["actor_logprob_old"].append(logp_old.mean().item())
                metrics["actor_ent"].append(self.policy.entropy(state).mean().item())

                metrics["regularized_rl_loss"].append(p_loss.item())
                metrics["rl_loss"].append(clip_loss.item())

                # Update parameters
                self.p_optimizer.zero_grad()
                final_loss.backward()

                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)

                self.p_optimizer.step()
                if not self.hps.shared_value:
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    self.v_optimizer.step()

        self.scheduler.step()

        metrics = self.process_metrics(metrics)
        return metrics, self.scheduler.get_last_lr()

    def get_reward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)

        # Transfer to cpu
        obs = obs.cpu()

        if self.hps.ot_rewards == "pwil":
            self.pwil_rewarder.reset()
            rewards = [
                self.pwil_rewarder.compute_reward({"observation": ob, "action": None})
                for ob in obs
            ]
            rewards = np.array(rewards)
        else:
            rewards = self.rewarder(obs, self.timesteps_so_far)
        return rewards

    def rewarder(self, observations, step):
        # Adapted from https://github.com/siddhanthaldar/ROT/blob/41ef7b98ca3950b9f31dd174f306cbe6916a09c9/ROT/agent/potil.py#L307

        if step % self.hps.rot_update_target_every == 0:
            self.trunk_target.load_state_dict(self.policy.perception_stack.state_dict())
            self.target_updated = True

        scores_list = list()
        ot_rewards_list = list()
        for traj in range(self.expert_dataset.data["obs0"].shape[0]):
            obs = observations.clone().to(self.device).float()
            obs = self.trunk_target(obs)

            exp = self.expert_dataset.data["obs0"][traj].to(self.device).float()
            exp = self.trunk_target(exp)

            obs = obs.detach()
            exp = exp.detach()

            # assert obs.shape == exp.shape, "obs and exp should have the same shape"

            if self.hps.ot_rewards == "sinkhorn_cosine":
                cost_matrix = cosine_distance(
                    obs, exp
                )  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method="sinkhorn", niter=100
                ).float()  # Getting optimal coupling
                ot_rewards = (
                    -self.hps.rot_sinkhorn_rew_scale
                    * torch.diag(torch.mm(transport_plan, cost_matrix.T))
                    .detach()
                    .cpu()
                    .numpy()
                )

            elif self.hps.ot_rewards == "sinkhorn_euclidean":
                cost_matrix = euclidean_distance(
                    obs, exp
                )  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method="sinkhorn", epsilon=0.1, niter=100
                ).float()  # Getting optimal coupling
                ot_rewards = (
                    -self.hps.rot_sinkhorn_rew_scale
                    * torch.diag(torch.mm(transport_plan, cost_matrix.T))
                    .detach()
                    .cpu()
                    .numpy()
                )
            elif self.hps.ot_rewards == "cosine":
                ot_rewards = -(1.0 - F.cosine_similarity(obs, exp))
                ot_rewards *= self.hps.rot_sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()
            elif self.hps.ot_rewards == "euclidean":
                ot_rewards = -(obs - exp).norm(dim=1)
                ot_rewards *= self.hps.rot_sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()
            else:
                raise NotImplementedError()

            scores_list.append(np.sum(ot_rewards))
            ot_rewards_list.append(ot_rewards)

        closest_demo_index = np.argmax(scores_list)
        return ot_rewards_list[closest_demo_index]

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
            ac, v, logp = self.predict(ob, True)

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
                p_rollout["next_v"].append(v * (1 - done))

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
            p_rollout["vs"].append(v)
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

                new_rewards = self.get_reward(p_rollout["obs0"])
                p_rollout["syn_rews"] = new_rewards.reshape(-1, 1)
                p_rollout["ep_syn_rets"] = new_rewards.sum()

                d["roll_syn_ret"].append(p_rollout["ep_syn_rets"].mean())

                if self.hps.rot_auto_rew_scale:
                    if iters_so_far == 0 and _ == 0:
                        print("Rotating sinkhorn rew scale")
                        self.hps.rot_sinkhorn_rew_scale = (
                            self.hps.rot_sinkhorn_rew_scale
                            * self.hps.rot_auto_rew_scale_factor
                            / float(np.abs(np.mean(p_rollout["ep_syn_rets"])))
                        )

                        new_rewards = self.get_reward(p_rollout["obs0"])
                        p_rollout["syn_rews"] = new_rewards.reshape(-1, 1)
                        p_rollout["ep_syn_rets"] = new_rewards.sum()

                p_rollout = postprocess_rollout(
                    p_rollout,
                    self.hps.gamma,
                    self.hps.gae_lambda,
                    rew_key="syn_rews",
                )

                if self.hps.trainer_worker:
                    rollouts = sync_data({"p_rollout": p_rollout})

            with timed("policy and value training"):
                if self.hps.master and self.hps.trainer_worker:
                    p_rollout = concatenate_rollouts(rollouts, key="p_rollout")

                if self.hps.master:
                    self.to(self._device)
                    metrics, lrnow = self.update_policy_value(
                        p_rollout=p_rollout,
                        iters_so_far=iters_so_far,
                    )
                    self.to("cpu")

                    d["lrnow"].append(lrnow)
                    d.update(metrics)

        self.log_model_stats()

    def update_timesteps(self, timesteps_so_far):
        return timesteps_so_far + (self.hps.g_steps * self.hps.rollout_len)
