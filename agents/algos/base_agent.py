import abc
import os
import os.path as osp
import random
from typing import Literal, Optional

import numpy as np
import torch

import wandb
from helpers import wandb_util
from helpers.dataloader import DataLoader
from helpers.dataset import DatasetTorch
from helpers.math_util import is_number


def _format_str(text):
    return "" if text is None else f"_{text}"


def get_local_rank():
    # TODO: move this somewhere else
    from helpers.mpi_util import MPI

    return MPI.COMM_WORLD.Get_rank()


def wandb_runing(function):
    def wrapper(*args, **kwargs):
        if get_local_rank() == 0:
            if wandb.run is not None:
                function(*args, **kwargs)
            else:
                raise RuntimeError(
                    "Your are trying to watch a model, but wandb is not initialized."
                )

    return wrapper


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self):
        return

    @abc.abstractmethod
    def rollout_generator(self):
        return

    @abc.abstractmethod
    def do_iteration(self):
        return

    @abc.abstractmethod
    def update_timesteps(self):
        return

    def process_metrics(self, metrics):
        metrics = {
            k: (
                torch.stack(v).mean().cpu().data.numpy()
                if not is_number(v) and not all([is_number(item) for item in v])
                else np.mean(v)
            )
            for k, v in metrics.items()
        }
        return metrics

    def setup_expert_dataloader(self, expert_dataset, batch_size, device=None):
        device = self.device

        self.e_batch_size = min(len(expert_dataset), batch_size)
        self.e_dataloader = DataLoader(
            DatasetTorch(expert_dataset.data, device=device),
            self.e_batch_size,
            shuffle=False,
            drop_last=True,
        )
        assert len(self.e_dataloader) > 0, "No expert data available"

    def _get_models(self):
        models = {
            attr_name: attr_value
            for attr_name, attr_value in self.__dict__.items()
            if isinstance(attr_value, torch.nn.Module)
            if len(attr_value.state_dict()) > 0
        }
        return models

    @wandb_runing
    def log_model_stats(
        self, log: Optional[Literal["gradients", "parameters", "all"]] = "all"
    ):
        models = self._get_models()
        wandb_util.log_params(models, log=log)

    @wandb_runing
    def watch(
        self,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        log_freq: int = 100,
    ) -> None:
        log_freq = wandb.config.get("wandb_log_freq", log_freq)
        wandb_util.watch(
            self._get_models(),
            log=log,
            log_freq=log_freq,
        )

    def to(self, device):
        for attr_value in self.__dict__.values():
            if hasattr(attr_value, "to"):
                attr_value.to(device)

    def train(self):
        for attr_value in self.__dict__.values():
            if hasattr(attr_value, "train"):
                attr_value.train()

    def eval(self):
        for attr_value in self.__dict__.values():
            if hasattr(attr_value, "eval"):
                attr_value.eval()

    def save(self, path, iters, random_state=False, rank=None, **kwargs):
        os.makedirs(path, exist_ok=True)
        objects_to_save = {}

        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, "state_dict"):
                state_dict = attr_value.state_dict()
                if len(state_dict) > 0:
                    objects_to_save[attr_name] = state_dict

        if random_state:
            self.save_random_states(path, iters, rank=rank)
            # INFO: following code is for saving random states within same dic but
            # it is not straightforward to save them using mpi

            # # Check if there is no overlaping keys
            # assert (
            #     len(set(random_states).intersection(set(objects_to_save))) == 0
            # ), "Duplicated keys found in save method."
            # # No overlaping keys, so put all together
            # objects_to_save.update(random_states)

        if len(objects_to_save) > 0:
            torch.save(
                objects_to_save, osp.join(path, f"iter{iters}{_format_str(rank)}.pth")
            )

    def load(self, path, iters, random_state=False, rank=None, **kwargs):
        if iters == "best":
            iters = f"_{iters}"
        else:
            assert isinstance(
                iters,
                int,
            ), "Invalid number of iterations, num_iter must be an integer or best"

        #add handling of rank ?
        objects_loaded = torch.load(
            osp.join(path, f"iter{iters}.pth")
        )

        for attr_name, attr_value in objects_loaded.items():
            if attr_name in self.__dict__:
                self.__dict__[attr_name].load_state_dict(attr_value)

        if random_state:
            self.load_random_states(path, iters, rank=rank)

    def save_random_states(self, path, iters, rank=None):
        random_state = {
            "torch_rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "rng_state": random.getstate(),
        }

        # cuda
        if torch.cuda.is_available():
            random_state["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
            # for all gpus -> torch.cuda.get_rng_state_all()

        # env
        # TODO:the env rand state should also be present
        torch.save(
            random_state,
            osp.join(path, f"rand_state_iter{iters}{_format_str(rank)}.pth"),
        )

    def load_random_states(self, path, iters, rank=None):
        random_state = torch.load(
            osp.join(path, f"rand_state_iter{iters}{_format_str(rank)}.pth")
        )

        # setting random states loaded
        torch.set_rng_state(random_state["torch_rng_state"])
        np.random.set_state(random_state["np_rng_state"])
        random.setstate(random_state["rng_state"])

        if torch.cuda.is_available():
            torch.cuda.set_rng_state(random_state["torch_cuda_rng_state"])
            # for all gpus -> torch.cuda.set_rng_state_all(state)
