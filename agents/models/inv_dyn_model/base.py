import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.modules.loss import _Loss as Loss

from agents.algos.base_agent import _format_str
from agents.memory import PrioritizedReplayBuffer, ReplayBuffer, UnrealReplayBuffer
from helpers import logger
from helpers.config_helper import Config
from helpers.distributed_util import (
    COMM,
    RunMoms,
    average_gradients,
    send_data_to_all,
    sync_data,
)
from helpers.mpi_util import MPI


class InvDynModelBase(nn.Module):
    loss_fn: Loss = NotImplemented
    optimizer: optim.Optimizer = NotImplemented

    def __init__(
        self,
        obs_size: int,
        actions_size: int,
        hps: Config,
        rms_obs: RunMoms,
        obs_shape: tuple = None,
    ):
        super(InvDynModelBase, self).__init__()
        self.hps = hps
        self.obs_size = obs_size
        self.actions_size = actions_size
        self.rms_obs = rms_obs
        self.obs_shape = obs_shape
        self.device = self.hps.device

        self.replay_buffer = self.setup_replay_buffer()

    @abstractmethod
    def forward(self, obs_before, obs_after):
        raise NotImplementedError

    @abstractmethod
    def sample(self, obs_before, obs_after, return_distribution=False):
        raise NotImplementedError

    def average_gradients(self):
        if not self.hps.trainer_worker:
            average_gradients(self, self.device)

    def predict(self, obs_before, obs_after, return_distribution=False):
        self.eval()
        with torch.no_grad():
            return self.sample(
                obs_before, obs_after, return_distribution=return_distribution
            )

    def train_until_converged(self, dataloader, patience=10):
        metrics = defaultdict(list)

        iter_wo_improvement = 0
        loss_min = np.iinfo(int).max

        # add new rollout
        self._store_rollout(dataloader)

        self.train()
        while patience > iter_wo_improvement:
            batch = self.sample_batch()
            obs_before = torch.Tensor(batch["obs0"]).to(self.device)
            obs_after = torch.Tensor(batch["obs1"]).to(self.device)
            actions = torch.Tensor(batch["acs"]).to(self.device)

            output = self.forward(obs_before, obs_after)

            loss = self.loss_fn(output, actions=actions)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()

            self.average_gradients()

            self.optimizer.step()

            loss = loss.item()
            loss_mse = nn.functional.mse_loss(
                self.sample(obs_before, obs_after), actions
            )
            metrics["inverse_dynamic_model/loss"].append(loss)
            metrics["inverse_dynamic_model/mse"].append(loss_mse)

            if loss >= loss_min:
                iter_wo_improvement += 1
            else:
                iter_wo_improvement = 0
                loss_min = loss

            iter_wo_improvement = np.min(sync_data(iter_wo_improvement, to_all=True))
        return metrics

    def _setup_replay_buffer_shapes(self):
        shapes = {
            "obs0": self.obs_shape,
            "obs1": self.obs_shape,
            "acs": (self.actions_size,),
        }
        return shapes

    def setup_replay_buffer(self):
        """Setup experiental memory unit"""
        logger.info(">>>> setting up replay buffer")
        shapes = self._setup_replay_buffer_shapes()
        # Create the buffer
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                replay_buffer = UnrealReplayBuffer(
                    self.hps.mem_size, shapes, device=self.device
                )
            else:  # Vanilla prioritized experience replay
                replay_buffer = PrioritizedReplayBuffer(
                    self.hps.mem_size,
                    shapes,
                    alpha=self.hps.alpha,
                    beta=self.hps.beta,
                    device=self.device,
                    ranked=self.hps.ranked,
                )
        else:  # Vanilla experience replay
            replay_buffer = ReplayBuffer(self.hps.mem_size, shapes, self.device)
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("{} configured".format(replay_buffer))
        return replay_buffer

    def _store_rollout(self, dataloader):
        dataset = dataloader.dataset
        for i in range(len(dataset)):
            transition = {
                k: v
                for k, v in dataset[i].items()
                if k in self.replay_buffer.ring_buffers.keys()
            }
            self.replay_buffer.append(transition)

    def sample_batch(self, batch_size=None, patcher=None):
        if batch_size is None:
            batch_size = self.hps.batch_size
        return self.replay_buffer.sample(batch_size, patcher=patcher)

    def save(self, path):
        pass

    def load(self, path=None, iters="best"):
        if iters == "best":
            iters = f"_{iters}"
        else:
            assert isinstance(
                iters,
                int,
            ), "Invalid number of iterations, num_iter must be an integer or best"

        assert (
            path is not None or self.hps.idm_pretrained_path is not None
        ), "Either path or idm_pretrained_path is required"

        if COMM.Get_rank() == 0:
            if self.hps.idm_pretrained_from_wandb:
                wandb_api = wandb.Api()
                run = wandb_api.run(self.hps.idm_pretrained_path)
                ckpt_dir = run.config["checkpoint_dir"].replace(run.config["root"], "")
                if ckpt_dir.startswith("/"):
                    ckpt_dir = ckpt_dir[1:]

                model_path = osp.join(ckpt_dir, run.config["uuid"])
                for file in run.files():
                    if iters in file.name:
                        print(file.name)
                        file.download(
                            root=model_path, replace=self.hps.wandb_download_replace
                        )
            else:
                model_path = path if path is not None else self.hps.idm_pretrained_path

            send_data_to_all(model_path)
        else:
            model_path = send_data_to_all(None)

        path_to_model = osp.join(model_path, "model_best", f"iter{iters}.pth")
        objects_loaded = torch.load(path_to_model)

        self.load_state_dict(objects_loaded["inv_dyn_model"])

        self.rms_obs = RunMoms(shape=self.obs_size, use_mpi=True)
        self.rms_obs.load_state_dict(objects_loaded["rms_obs"])
