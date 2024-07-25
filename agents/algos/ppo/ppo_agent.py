import os.path as osp
from collections import defaultdict, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn.utils as U
from gym import spaces

from agents import get_kwarg
from agents.algos.base_agent import BaseAgent
from agents.gae import gae
from agents.nets import CatPolicy, GaussPolicy, Value
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
from helpers.math_util import LRDecay


class PPOAgent(BaseAgent):
    def __init__(self, env, device, hps):
        self.env = env

        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]
        self.device = device
        self.hps = hps
        if self.hps.clip_norm <= 0:
            logger.info(
                "[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm),
            )

        # Create observation normalizer that maintains running statistics
        self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=False)

        assert self.hps.shared_value or not self.hps.shared_value_policy_detached

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps, self.rms_obs).to(self.device)
        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps, self.rms_obs).to(self.device)
        self._sync_weights_master()

        # Set up the optimizer
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

    def _sync_weights_master(self):
        def sum_params(model):
            s = []
            for p in model.parameters():
                dims = list(p.size())
                n = p.cpu().data.numpy()
                s.append((dims, np.sum(n)))
            return s

        sync_weights(self.policy, msg="[policy]")
        if not self.hps.shared_value:
            sync_weights(self.value, msg="[value]")

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

    def update_policy_value(self, rollout, iters_so_far):
        """Train the agent"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key="env_rews")

        # Standardize advantage function estimate
        rollout["advs"] = (rollout["advs"] - rollout["advs"].mean()) / (
            rollout["advs"].std() + 1e-8
        )

        metrics["rollout_advs_std"] = rollout["advs"].std().item()
        metrics["rollout_advs_mean"] = rollout["advs"].mean().item()

        # Create DataLoader object to iterate over transitions in rollouts
        keys = ["obs0", "acs", "logps", "vs", "advs", "td_lam_rets"]

        dataset = DatasetTorch({k: rollout[k] for k in keys}, device=self.device)
        dataloader = DataLoader(
            dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        for _ in range(self.hps.optim_epochs_per_iter):
            for chunk in dataloader:
                # Transfer to device
                state = chunk["obs0"].to(self.device)
                action = chunk["acs"].to(self.device)
                logp_old = chunk["logps"].to(self.device)
                v_old = chunk["vs"].to(self.device)
                advantage = chunk["advs"].to(self.device)
                td_lam_return = chunk["td_lam_rets"].to(self.device)

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

                # Log metrics
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["clip_loss"].append(clip_loss.item())
                metrics["kl_approx"].append(kl_approx.item())
                metrics["kl_max"].append(kl_max.item())
                metrics["clip_frac"].append(clip_frac.item())
                metrics["v_loss"].append(v_loss.item())
                metrics["p_loss"].append(p_loss.item())
                metrics["advantage"].append(advantage.mean().item())
                metrics["rati_mean"].append(ratio.mean().item())
                metrics["rati_std"].append(ratio.std().item())
                metrics["ratio_clamped_mean"].append(
                    ratio.clamp(1.0 - self.hps.eps, 1.0 + self.hps.eps).mean().item(),
                )
                metrics["ratio_clamped_std"].append(
                    ratio.clamp(1.0 - self.hps.eps, 1.0 + self.hps.eps).std().item(),
                )
                metrics["surrogate_loss_a_mean"].append(surrogate_loss_a.mean().item())
                metrics["surrogate_loss_a_std"].append(surrogate_loss_a.std().item())
                metrics["surrogate_loss_b_mean"].append(surrogate_loss_b.mean().item())
                metrics["surrogate_loss_b_std"].append(surrogate_loss_b.std().item())

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                if not self.hps.trainer_worker:
                    average_gradients(self.policy, self.device)
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.p_optimizer.step()
                if not self.hps.shared_value:
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    if not self.hps.trainer_worker:
                        average_gradients(self.value, self.device)
                    self.v_optimizer.step()

        self.scheduler.step()

        metrics = self.process_metrics(metrics)
        return metrics, self.scheduler.get_last_lr()

    def rollout_generator(self, rollout_len=None):
        rollout_len = rollout_len if rollout_len else self.hps.rollout_len

        t = 0
        done = True
        # Reset agent's env
        rst = self.env.reset()
        if isinstance(rst, tuple) and len(rst) == 2:
            ob, _ = rst
        else:
            ob = rst
        ob = np.array(ob)
        # Init collections
        rollout = defaultdict(list)
        # Init current episode statistics
        cur_ep_len = 0
        cur_ep_env_ret = 0

        while True:
            # Predict
            ac, v, logp = self.predict(ob, True)

            if not isinstance(self.ac_space, spaces.Discrete):
                # NaN-proof and clip
                ac = np.nan_to_num(ac)
                ac = np.clip(ac, self.env.action_space.low, self.env.action_space.high)
            else:
                ac = ac if isinstance(ac, int) else ac.item()

            if t > 0 and t % rollout_len == 0:
                for k in rollout.keys():
                    if k in ["obs0", "obs1"]:
                        rollout[k] = np.array(rollout[k]).reshape(-1, *self.ob_shape)
                    elif k == "acs":
                        if self.is_discrete:
                            ac_dim = 1
                        else:
                            ac_dim = self.ac_dim
                        rollout[k] = np.array(rollout[k]).reshape(-1, ac_dim)
                    elif k in ["vs", "logps", "env_rews", "dones"]:
                        rollout[k] = np.array(rollout[k]).reshape(-1, 1)
                    else:
                        rollout[k] = np.array(rollout[k])
                rollout["next_v"].append(v * (1 - done))

                yield rollout

                # Clear the collections
                rollout.clear()

            # Interact with env(s)
            step = self.env.step(ac)
            if len(step) == 4:
                new_ob, env_rew, done, _ = step
            elif len(step) == 5:
                new_ob, env_rew, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                raise ValueError(f"unexpected step length: {len(step)}")

            # Populate collections
            rollout["obs0"].append(ob)
            rollout["acs"].append(ac)
            rollout["obs1"].append(new_ob)
            rollout["vs"].append(v)
            rollout["logps"].append(logp)
            rollout["env_rews"].append(env_rew)
            rollout["dones"].append(done)

            # Update current episode statistics
            cur_ep_len += 1
            if hasattr(self.env, "_elapsed_steps"):
                assert self.env._elapsed_steps == cur_ep_len  # sanity check
            cur_ep_env_ret += env_rew

            # Set current state with the next
            ob = np.array(deepcopy(new_ob))

            if done:
                # Update the global episodic statistics and
                # reset current episode statistics
                rollout["ep_lens"].append(cur_ep_len)
                cur_ep_len = 0
                rollout["ep_env_rets"].append(cur_ep_env_ret)
                cur_ep_env_ret = 0
                # Reset env
                rst = self.env.reset()
                if isinstance(rst, tuple) and len(rst) == 2:
                    ob, _ = rst
                else:
                    ob = rst
                ob = np.array(ob)

            t += 1

    def do_iteration(self, **kwargs):
        timed = get_kwarg(kwargs, "timed")
        roll_gen = get_kwarg(kwargs, "roll_gen")
        iters_so_far = get_kwarg(kwargs, "iters_so_far")
        d = get_kwarg(kwargs, "d")

        with timed("interacting"):
            rollout = roll_gen.__next__()
            d["roll_len"].append(mpi_mean_reduce(rollout["ep_lens"]))
            roll_env_ret = mpi_mean_reduce(rollout["ep_env_rets"])
            d["roll_env_ret"].append(roll_env_ret)

            if self.hps.trainer_worker:
                rollouts = sync_data(rollout)

        with timed("training"):
            if self.hps.master and self.hps.trainer_worker:
                rollout = {
                    key: np.concatenate([p_roll[key] for p_roll in rollouts])
                    for key in rollouts[0].keys()
                }

            if self.hps.master:
                metrics, lrnow = self.update_policy_value(
                    rollout=rollout,
                    iters_so_far=iters_so_far,
                )

                d["pol_losses"].append(metrics["p_loss"])
                d["val_losses"].append(metrics["v_loss"])
                d["entropy_loss"].append(metrics["entropy_loss"])
                d["clip_loss"].append(metrics["clip_loss"])
                d["kl_approx"].append(metrics["kl_approx"])
                d["kl_max"].append(metrics["kl_max"])
                d["clip_frac"].append(metrics["clip_frac"])
                d["advantage"].append(metrics["advantage"])
                d["rati_mean"].append(metrics["rati_mean"])
                d["rati_std"].append(metrics["rati_std"])
                d["ratio_clamped_mean"].append(metrics["ratio_clamped_mean"])
                d["ratio_clamped_std"].append(metrics["ratio_clamped_std"])
                d["surrogate_loss_a_mean"].append(metrics["surrogate_loss_a_mean"])
                d["surrogate_loss_a_std"].append(metrics["surrogate_loss_a_std"])
                d["surrogate_loss_b_mean"].append(metrics["surrogate_loss_b_mean"])
                d["surrogate_loss_b_std"].append(metrics["surrogate_loss_b_std"])
                d["lrnow"].append(lrnow)

            if self.hps.trainer_worker:
                self._sync_weights_master()

    def update_timesteps(self, timesteps_so_far):
        return timesteps_so_far + self.hps.rollout_len
