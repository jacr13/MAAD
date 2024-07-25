import glob
import os.path as osp
import pickle
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import h5py
import numpy as np
import torch

from helpers import logger
from helpers.env_makers import get_benchmark


def save_dict_h5py(data, fname):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, "w") as hf:
        for key in data.keys():
            hf.create_dataset(key, data=data[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = {}
    with h5py.File(fname, "r") as hf:
        for key in hf.keys():
            data[key] = hf[key][()]
    return data


def load_dict_pkl(fname):
    # shapes:
    # (10, 500, 9, 84, 84) (10, 500, 24) (10, 500, 6) (10, 500)
    data = pickle.load(open(fname, "rb"))

    for i, d in enumerate(data):
        if isinstance(d, list):
            data[i] = np.stack(d, axis=0)

    obs_pixels, obs_features, actions, rewards = data

    return {
        "pix_obs0": obs_pixels,
        "obs0": obs_features,
        "acs": actions,
        "env_rews": rewards,
    }


def load_demos(fname):
    if fname.endswith(".h5"):
        return load_dict_h5py(fname)
    elif fname.endswith(".pkl"):
        return load_dict_pkl(fname)
    elif fname.endswith(".npy"):
        return np.load(fname)
    else:
        raise ValueError("File type not supported")


def readMotionFile(filename):
    """Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not osp.exists(filename):
        print("file do not exists")

    file_id = open(filename, "r")

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nr = 0
    while "endheader" not in next_line:
        if "datarows" in next_line:
            nr = int(next_line[next_line.index(" ") + 1 : len(next_line)])
        elif "nRows" in next_line:
            nr = int(next_line[next_line.index("=") + 1 : len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace():
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for _ in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data


class DatasetTorch(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        self.data = data
        self.device = device

        for k, v in self.data.items():
            self.data[k] = torch.Tensor(v).to(self.device)

        assert all(
            list(self.data.values())[0].size(0) == tensor.size(0)
            for tensor in self.data.values()
        ), "Size mismatch between tensors"

    def to(self, device):
        self.data = {k: v.to(device) for k, v in self.data.items()}

    def get_batch(self, batch_idx):
        return {k: v[batch_idx] for k, v in self.data.items()}

    def __getitem__(self, i):
        return {k: v[i, ...] for k, v in self.data.items()}

    def __len__(self):
        return list(self.data.values())[0].size(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return {k: v[i, ...].astype(np.float32) for k, v in self.data.items()}

    def __len__(self):
        return list(self.data.values())[0].shape[0]


class DemoDataset(Dataset):
    def __init__(
        self, hps, env_id, expert_path, num_demos, env, wrap_absorb
    ):  # noqa: C901
        self.hps = hps
        self.num_demos = num_demos
        self.data = defaultdict(list)
        self.stats = defaultdict(list)
        logger.info(">>>> loading demos")

        benchmark = get_benchmark(env_id)

        # Load data from Mujoco env
        if self.hps.algo in ["tmot", "maad-ot"]:
            demonstrations = glob.glob(osp.join(expert_path, "*.h5"))
            random.shuffle(demonstrations)
            for i, f in enumerate(demonstrations[: self.hps.num_demos]):
                demo = load_demos(f)
                sub_rate = self.hps.expert_demo_subsample_rate
                if self.hps.ot_rewards in [
                    "sinkhorn_cosine",
                    "sinkhorn_euclidean",
                    "cosine",
                    "euclidean",
                ]:
                    self.hps.expert_demo_subsample = False
                    warnings.warn(
                        f"Modifying expert demo subsample to False since {self.hps.ot_rewards} requires full expert trajectory"
                    )

                if self.hps.expert_demo_subsample:
                    random_offset = random.randint(0, sub_rate - 1)
                    obs0 = demo["obs0"][random_offset::sub_rate]
                    acs = demo["acs"][random_offset::sub_rate]
                    obs1 = demo["obs1"][random_offset::sub_rate]
                else:
                    obs0 = demo["obs0"]
                    acs = demo["acs"]
                    obs1 = demo["obs1"]

                    # if obs0.shape[0] != 1000:
                    #     continue

                self.data["obs0"].append(obs0)
                self.data["acs"].append(acs)
                self.data["obs1"].append(obs1)

            for k, v in self.data.items():
                self.data[k] = np.stack(v)

            return

        elif (
            benchmark == "mujoco"
            or "MZ" in expert_path
            or "dmc" in expert_path
            or "my_metaworld" in expert_path
        ):
            # Go over the demos, sorted in alphabetical order
            for i, f in enumerate(sorted(glob.glob(osp.join(expert_path, "*.h5")))):
                # If the desired number of demos has been aggregated, leave
                if i == num_demos:
                    break
                # Log the location of the loaded demo
                logger.info(
                    "[INFO] demo #{} loaded from: {}".format(str(i).zfill(3), f),
                )
                # Load the demo from the file
                tmp = load_demos(f)
                # Remove undesirable keys (at least in this application)
                assert tmp["dones1"][
                    -1
                ], "by construction"  # making sure every ep ends with done
                tmp.pop("pix_obs0", None)
                tmp.pop("pix_obs1", None)
                # Extract and display content dims
                dims = {k: tmp[k].shape[1:] for k in tmp.keys()}
                dims = " | ".join(["{}={}".format(k, v) for k, v in dims.items()])
                logger.info("[INFO]      dims: {}".format(dims))
                # Get episode statistics
                ep_len = tmp.pop("ep_lens", None)  # return and delete key
                ep_ret = tmp.pop("ep_env_rets", None)  # return and delete key
                assert isinstance(
                    ep_len,
                    np.int64,
                ), "each file should contain only one episode"
                assert isinstance(
                    ep_ret,
                    np.float64,
                ), "each file should contain only one episode"
                logger.info("[INFO]      {}{}".format("ep_len".ljust(20, "-"), ep_len))
                logger.info("[INFO]      {}{}".format("ep_ret".ljust(20, "-"), ep_ret))
                self.stats["ep_len"].append(ep_len)
                self.stats["ep_ret"].append(ep_ret)
                # Determine if terminal because of timeout or real termination
                terminal = not ep_len == env._max_episode_steps

                # Subsample trajectory
                if self.hps.expert_demo_subsample:
                    if self.hps.algo not in ["bc", "rot"] and self.hps.task != "eval":
                        sub_rate = (
                            self.hps.expert_demo_subsample_rate
                        )  # N=20 in the original GAIL paper
                        start = np.random.randint(low=0, high=sub_rate)
                        indices = [
                            start + (i * sub_rate) for i in range(ep_len // sub_rate)
                        ]
                        ep_len = len(indices)  # overwrite ep_len
                        logger.info(
                            "[INFO]      {}{}".format(
                                "subsample".ljust(20, "-"),
                                "{}(sub_rate={})".format(ep_len, sub_rate),
                            ),
                        )
                        for k in tmp.keys():
                            tmp[k] = tmp[k][indices]

                # Collect the demo's content
                if wrap_absorb:
                    if tmp["dones1"][-1] and terminal:
                        # If the last subsampled transition is done, then it must be
                        # the very last transition of the episode, and testing whether it is
                        # a true terminal state is given by 'terminal' determined above.
                        logger.info(
                            "[INFO] >>>> wrapping with absorbing transition <<<<",
                        )
                        # Wrap with an absorbing state
                        obs0 = np.concatenate(
                            [tmp["obs0"], np.zeros((ep_len, 1))],
                            axis=-1,
                        )
                        acs = np.concatenate(
                            [tmp["acs"], np.zeros((ep_len, 1))],
                            axis=-1,
                        )
                        obs1 = np.concatenate(
                            [
                                tmp["obs1"],
                                np.concatenate(
                                    [np.zeros((ep_len - 1, 1)), np.ones((1, 1))],
                                    axis=0,
                                ),
                            ],
                            axis=-1,
                        )
                        # Add absorbing transition
                        obs0 = np.concatenate(
                            [
                                obs0,
                                np.expand_dims(
                                    np.append(np.zeros_like(tmp["obs0"][-1]), 1),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        )
                        acs = np.concatenate(
                            [
                                acs,
                                np.expand_dims(
                                    np.append(np.zeros_like(tmp["acs"][-1]), 1),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        )
                        obs1 = np.concatenate(
                            [
                                obs1,
                                np.expand_dims(
                                    np.append(np.zeros_like(tmp["obs1"][-1]), 1),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        )
                        self.data["obs0"].append(obs0)
                        self.data["acs"].append(acs)
                        self.data["obs1"].append(obs1)
                    else:
                        self.data["obs0"].append(
                            np.concatenate(
                                [tmp["obs0"], np.zeros((ep_len, 1))],
                                axis=-1,
                            ),
                        )
                        self.data["acs"].append(
                            np.concatenate(
                                [tmp["acs"], np.zeros((ep_len, 1))],
                                axis=-1,
                            ),
                        )
                        self.data["obs1"].append(
                            np.concatenate(
                                [tmp["obs1"], np.zeros((ep_len, 1))],
                                axis=-1,
                            ),
                        )
                else:
                    self.data["obs0"].append(tmp["obs0"])
                    self.data["acs"].append(tmp["acs"])
                    self.data["obs1"].append(tmp["obs1"])

        else:
            raise ValueError("benchmark not supported")

        """
        # Log demos' statistics
        logger.info("[INFO] keys extracted: {}".format(list(self.data.keys())))
        lens, rets = self.stats['ep_len'], self.stats['ep_ret']
        logger.info("[INFO] got {} transitions, from {} eps".format(len(self), self.num_demos))
        logger.info("[INFO] episodic length: {}({})".format(np.mean(lens), np.std(lens)))
        logger.info("[INFO] episodic return: {}({})".format(np.mean(rets), np.std(rets)))
        """
