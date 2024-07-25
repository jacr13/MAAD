import json
import os
import os.path as osp
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.utils as U
import wandb
from gym import spaces
from torch.utils.data import DataLoader

from agents import get_kwarg
from agents.algos.base_agent import BaseAgent
from agents.models.inv_dyn_model import get_idm
from agents.nets import GaussPolicy
from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.dataset import DatasetTorch
from helpers.distributed_util import RunMoms
from helpers.math_util import LRDecay


def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return {str(k): v for k, v in data.items()}
    else:
        print(f"The file {file_path} does not exist.")
        return {}


def save_json_file(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


class BCAgent(BaseAgent):
    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]
        self.hps = hps

        self.device = device
        self._device = device

        self._timesteps_so_far = 0

        file_path = "data_bc.json"

        loaded_data = load_json_file(file_path)
        if "expert" not in loaded_data.keys():
            loaded_data["expert"] = {}
        if self.hps.env_id not in loaded_data["expert"].keys():
            loaded_data["expert"][self.hps.env_id] = {
                "return": {
                    "mean": np.mean(expert_dataset.stats["ep_ret"]),
                    "std": np.std(expert_dataset.stats["ep_ret"]),
                }
            }

        algo = (
            f"bco_{self.hps.rollout_len}_{self.hps.bco_post_rollout_len}"
            if self.hps.state_state
            else "bc"
        )
        self.algo = algo
        if algo not in loaded_data.keys():
            loaded_data[algo] = {}

        if str(self.hps.batch_size) not in loaded_data[algo].keys():
            loaded_data[algo][str(self.hps.batch_size)] = {}

        if self.hps.env_id not in loaded_data[algo][str(self.hps.batch_size)].keys():
            loaded_data[algo][str(self.hps.batch_size)][self.hps.env_id] = {}
        save_json_file(loaded_data, file_path)

        self.expert_dataset = expert_dataset

        if self.hps.clip_norm <= 0:
            logger.info(
                "[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm),
            )

        # Create observation normalizer that maintains running statistics
        self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

        assert self.hps.shared_value or not self.hps.shared_value_policy_detached

        # Create nets
        self.policy = GaussPolicy(self.env, self.hps, self.rms_obs).to(self.device)

        # Set up demonstrations dataset
        self.setup_expert_dataloader(
            self.expert_dataset, self.hps.batch_size, self.device
        )

        if self.hps.state_state:
            self.expert_dataset.data.pop("acs", None)
            self.inv_dyn_model = get_idm(self.hps.idm_model)(
                self.ob_dim, self.ac_dim, self.hps, self.rms_obs, self.ob_shape
            ).to(self.device)

        self.loss_fn = torch.nn.MSELoss()

        # Set up the optimizers
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)

        self.lr_decay = LRDecay(self.hps, self.hps.p_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.p_optimizer,
            self.lr_decay,
        )
        log_module_info(logger, "policy", self.policy)

    def to(self, device):
        if device != self.device:
            self.device = device
            self.policy = self.policy.to(device)

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, None, logp

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
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, self.ob_dim)
                    elif k == "acs":
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, self.ac_dim)
                    elif k in ["vs", "logps", "env_rews", "dones"]:
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, 1)
                    else:
                        p_rollout[k] = np.array(p_rollout[k])
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

    def update_timesteps(self, timesteps_so_far):
        if self.hps.state_state:
            timesteps = self.hps.rollout_len
            if timesteps_so_far == 0:
                self.hps.rollout_len = self.hps.bco_post_rollout_len
            self._timesteps_so_far = timesteps_so_far + timesteps
            return self._timesteps_so_far
        return 0

    def update_iters_so_far(self, iters_so_far):
        if not self.hps.state_state:
            # stop training but log evaluation

            # retreinving the max number of iterations
            num_iters = self.hps.num_timesteps // self.hps.rollout_len

            # calculating the number of iters so it is bigger than num_iters and
            # a multiple of eval_frequency so it logs evaluation
            iters_so_far = (
                (num_iters // self.hps.eval_frequency) + 1
            ) * self.hps.eval_frequency
        return iters_so_far + 1

    def do_iteration(self, **kwargs):
        timed = get_kwarg(kwargs, "timed")
        roll_gen = get_kwarg(kwargs, "roll_gen")

        if self.hps.state_state:
            with timed("interacting"):
                # Unpack (one rollout dict for policy training, one for reward training)
                self.to("cpu")
                p_rollout = roll_gen.__next__()
                print(p_rollout["obs0"].shape)
                self.to(self._device)

            with timed("Inverse dynamic model training"):
                p_keys = ["obs0", "obs1", "acs"]
                p_dataset = DatasetTorch(
                    {k: p_rollout[k] for k in p_keys}, device=self.device
                )
                p_dataloader = DataLoader(
                    p_dataset,
                    self.hps.batch_size,
                    shuffle=True,
                    drop_last=False,  # no compatibility issue, only used for policy alone
                )

                idm_metrics = self.inv_dyn_model.train_until_converged(
                    p_dataloader, patience=50
                )
                # print(idm_metrics["inverse_dynamic_model/loss"])

                self.expert_dataset.data.pop("acs", None)
                obs_before = torch.Tensor(self.expert_dataset.data["obs0"])
                obs_after = torch.Tensor(self.expert_dataset.data["obs1"])

                if self.hps.wrap_absorb:
                    obs_before = obs_before[:, :-1]
                    obs_after = obs_after[:, :-1]

                actions_sampled = self.inv_dyn_model.sample(obs_before, obs_after)
                if self.hps.wrap_absorb:
                    act_shape = list(actions_sampled.shape)
                    act_shape[-1] += 1
                    actions_ = torch.zeros(act_shape)
                    actions_[:, 0:-1] = actions_sampled
                    actions_sampled = actions_

                self.expert_dataset.data["acs"] = actions_sampled.cpu().detach().numpy()
                self.setup_expert_dataloader(self.expert_dataset, self.hps.batch_size)

        with timed("BC policy training"):
            loss_history = []
            for epoch in range(self.hps.bc_epochs):
                losses = []
                for e_batch in self.e_dataloader:
                    # Transfer to device
                    state = e_batch["obs0"].to(self.device)
                    action = e_batch["acs"].to(self.device)
                    if self.hps.wrap_absorb:
                        state = state[:, :-1]
                        action = action[:, :-1]

                    self.rms_obs.update(state)

                    pred_action = self.policy.rsample(state)
                    p_loss = self.loss_fn(pred_action, action)

                    # Update parameters
                    self.p_optimizer.zero_grad()
                    p_loss.backward()
                    if self.hps.clip_norm > 0:
                        U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                    self.p_optimizer.step()
                    losses.append(p_loss.item())

                # print(
                #     f"Epoch {epoch+1}/{self.hps.bc_epochs}, loss: {np.mean(losses)}",
                #     " " * 30,
                #     end="\r",
                # )
                # d["bc_loss"].append(np.mean(loss_history))
                self.scheduler.step()

                if not self.hps.state_state:
                    wandb.log(
                        {"bc_loss": np.mean(losses)},
                        step=epoch,
                    )

                loss_history.append(np.mean(losses))

                # check for early stopping
                if epoch > self.hps.bc_patience:
                    if all(
                        loss_history[-1] >= loss_history[-i - 1]
                        for i in range(self.hps.bc_patience)
                    ):
                        print(
                            f"Stopping early. No improvement in the last {self.hps.bc_patience} epochs."
                        )
                        break
            if self.hps.state_state:
                wandb.log(
                    {"bc_loss": np.mean(loss_history)}, step=self._timesteps_so_far
                )
