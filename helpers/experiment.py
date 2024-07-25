import hashlib
import json
import os
import os.path as osp
import random
import string
import subprocess
from typing import Any, Dict

import numpy as np
import yaml

from agents import ALGOS_WITH_DEMONSTRATIONS
from helpers import humanhash, logger
from helpers.argparsers import argparser


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def generate_salt(length):
    """Generate a random salt with the specified length."""
    characters = string.ascii_letters + string.digits
    salt = "".join(random.choice(characters) for _ in range(length))
    return salt


def get_gitsha():
    _gitsha = "gitSHA_{}"
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        gitsha = _gitsha.format(out.strip().decode("ascii"))
    except OSError:
        gitsha = "noGitSHA"
    return gitsha


def replace_batch_norm_to_global(args):
    if args["batch_norm_all"]:
        batch_norm_keys = [key for key in args.keys() if "_batch_norm" in key]
        for key in batch_norm_keys:
            args[key] = True
    return args


def create_uuid(*args, **kwargs):
    """Builds the uuid of the experiment."""
    uuid = uuid_basis(*args, **kwargs)
    config = kwargs.get("config", None)
    seed = kwargs.get("seed", None)

    assert seed is not None, "seed must be specified"
    assert config is not None, "config must be specified"

    algo = config["algo"]
    need_demos = algo in ALGOS_WITH_DEMONSTRATIONS

    # Enrich the uuid with extra information
    uuid = f"{config['uuid']}.{algo}.{get_gitsha()}.{config['env_id']}"
    if need_demos:
        uuid += f".demos{str(config['num_demos']).zfill(3)}"
    uuid += f".seed{str(seed).zfill(2)}"
    return uuid


def uuid_basis(*args, **kwargs):
    method = kwargs.get("method", "human_hash")
    add_salt = kwargs.get("add_salt", False)

    uuid = None
    if method == "syllables":
        uuid = uuid_syllables(*args, **kwargs)
    elif method == "hash":
        uuid = uuid_hash(*args, **kwargs)
    elif method == "human_hash":
        uuid = humanhash.humanize(uuid_hash(*args, **kwargs), words=5)
    elif method == "parameters":
        uuid = uuid_parameters(*args, **kwargs)
    else:
        raise NotImplementedError

    if add_salt:
        salt_len = kwargs.get("salt_len", 4)
        salt = generate_salt(salt_len)
        uuid = f"{uuid}-{salt}"
    return uuid


def is_valid_key(key):
    if key.startswith("wandb"):
        return False
    if key in ["seed", "execution_time", "uuid", "task", "num_timestep", "expert_path"]:
        return False

    return True


def get_min_config(*args, **kwargs):
    config = kwargs.get("config", None)
    assert config is not None, "config must be specified"

    config_default = vars(argparser().parse_args(args=""))

    config_min = {
        key: value
        for key, value in config.items()
        if key in config_default and value != config_default[key]
    }
    return dict(sorted(config_min.items()))


def uuid_parameters(*args, **kwargs):
    config_min = get_min_config(*args, **kwargs)
    uuid = ""
    for key, value in config_min.items():
        if is_valid_key(key):
            uuid += f"{key}={value}-"

    uuid = uuid[:-1]
    return uuid


def uuid_hash(*args, **kwargs):
    min_config = get_min_config(*args, **kwargs)
    uuid = dict_hash(min_config)
    return uuid


def uuid_syllables(*args, **kwargs):
    """Randomly create a semi-pronounceable uuid"""
    num_syllables = kwargs.get("num_syllables", 2)
    num_parts = kwargs.get("num_parts", 3)
    part1 = [
        "s",
        "t",
        "r",
        "ch",
        "b",
        "c",
        "w",
        "z",
        "h",
        "k",
        "p",
        "ph",
        "sh",
        "f",
        "fr",
    ]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


class ConfigDumper:
    def __init__(self, args, path=None):
        """Log the job config into a file"""
        self.args = args
        os.makedirs(path, exist_ok=True)
        self.path = path

    def dump(self):
        hpmap = self.args.to_dict()
        with open(osp.join(self.path, "hyperparameters.yaml"), "w") as outfile:
            yaml.safe_dump(hpmap, outfile, default_flow_style=False)


class ExperimentInitializer:
    def __init__(self, args, rank=None, world_size=None):
        """Initialize the experiment"""
        self.uuid_provided = args.uuid is not None
        self.uuid = (
            args.uuid
            if self.uuid_provided
            else uuid_basis(config=args, method="human_hash")
        )
        self.args = args
        self.rank = rank
        self.world_size = world_size
        # Set printing options
        np.set_printoptions(precision=3)
        # update batch_norm if batch_norm_all is true
        self.args.update(replace_batch_norm_to_global(args))

    def configure_logging(self):
        """Configure the experiment"""
        if self.rank is None:  # eval
            logger.info("configuring logger for evaluation")
            logger.configure(dir_=None, format_strs=["stdout"])

        elif self.rank == 0:  # train, master
            log_path = osp.join(self.args.log_dir, self.get_name())
            formats_strs = ["stdout", "log", "csv"]
            fmtstr = "configuring logger"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.configure(dir_=log_path, format_strs=formats_strs)
            fmtstr = "logger configured"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.info("  directory: {}".format(log_path))
            logger.info("  output formats: {}".format(formats_strs))
            # In the same log folder, log args in a YAML file
            self.args.save(path=log_path)
            fmtstr = "experiment configured"
            fmtstr += " [{} MPI workers]".format(self.world_size)
            logger.info(fmtstr)

        else:  # train, worker
            logger.info("configuring logger [worker #{}]".format(self.rank))
            logger.configure(dir_=None, format_strs=None)
            logger.set_level(logger.DISABLED)

    def get_name(self):
        """Assemble long experiment name"""
        if self.uuid_provided:
            # If the uuid has been provided, use it.
            return self.uuid
        # Assemble the uuid
        name = self.uuid + "."
        try:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            name += "gitSHA_{}.".format(out.strip().decode("ascii"))
        except OSError:
            pass
        if self.args.task == "eval":
            name += "{}.".format(self.args.task)
            name += "num_trajs_{}.".format(self.args.num_trajs)
        name += self.args.env_id
        name += ".{}".format(self.args.reward_function)
        name += ".seed{}".format(str(self.args.seed).zfill(2))
        return name
