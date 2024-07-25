import os
import os.path as osp
import random
import shutil
import time

import numpy as np
import torch
import wandb
import yaml

from agents import agent_wrapper, orchestrator
from helpers import logger
from helpers.argparsers import argparser
from helpers.config_helper import Config
from helpers.env_makers import make_env
from helpers.experiment import ExperimentInitializer, uuid_basis
from helpers.mpi_util import MPI


def wandb_init(args: Config, experiment_name: str) -> None:
    # Group by everything except the seed, which is last, hence index -1
    # For 'gail', it groups by uuid + gitSHA + env_id + num_demos,
    # while for 'ppo', it groups by uuid + gitSHA + env_id
    seed_idx = [
        idx
        for idx, segment in enumerate(experiment_name.split("."))
        if "seed" in segment
    ][0]
    group = ".".join(experiment_name.split(".")[:seed_idx])
    # group = uuid_basis(method="hash", config=args.to_dict())

    log_dir = osp.join(args.log_dir, experiment_name)
    # Set up wandb
    while True:
        try:
            wandb.init(
                project=args.wandb_project,
                name=experiment_name,
                id=experiment_name,
                group=group,
                config=args.to_dict(),
                dir=log_dir,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                resume="allow",
                # job_type=args.task,  # TODO: is it needed ? grouping weird
            )
        except ConnectionRefusedError:
            pause = 5
            logger.info("wandb co error. Retrying in {} secs.".format(pause))
            time.sleep(pause)
            continue
        logger.info("wandb co established!")
        break
    # Save hyperparameters
    shutil.copy(
        osp.join(log_dir, "hyperparameters.yaml"),
        os.path.join(wandb.run.dir, "hyperparameters.yaml"),
    )
    wandb.save(osp.join(log_dir, "hyperparameters.yaml"), base_path=log_dir)


def wandb_download(args):
    # get default args
    defaults = vars(argparser().parse_args([]))

    # connect to wandb api and get run
    wandb_api = wandb.Api()
    run = wandb_api.run(args.wandb_run_path)
    ckpt_dir = run.config["checkpoint_dir"].replace(run.config["root"], "")
    if ckpt_dir.startswith("/"):
        ckpt_dir = ckpt_dir[1:]

    model_path = osp.join(ckpt_dir, run.config["uuid"])
    for file in run.files():
        if str(args.iter_num) in file.name:
            if args.iter_num == "model_timeout":
                args.iter_num = (
                    file.name.split("/")[-1].split(".")[0].replace("iter", "")
                )
                if args.iter_num is not None and args.iter_num.isdigit():
                    args.iter_num = int(args.iter_num)
            file.download(root=model_path, replace=args.wandb_download_replace)
    args.update(run.config, defaults=defaults)

    # set path to the model downloaded from wandb
    args.model_path = model_path


def get_device(args):
    # Set device-related knobs
    if args.cuda and args.master:
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = "cuda"
        # should call setup_mpi_gpus() ?
        return torch.device(device)
    else:
        # kill any possibility of usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    return torch.device(device)


def train(args):
    """Train an agent"""

    # update args if wandb_run_path exists
    if args.wandb_run_path is not None:
        wandb_download(args)

    # Get the current process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    args.rank = rank

    args.master = True
    if args.trainer_worker:
        if rank == 0:
            args.master = True
        else:
            args.master = False

    torch.set_num_threads(1)

    # num_timesteps = {
    #     "Cheetah-Run-Feat-v0": 8e6,
    #     "Walker-Walk-Feat-v0": 22e6,
    #     "Walker-Run-Feat-v0": 22e6,
    #     "Quadruped-Walk-Feat-v0": 5e6,
    #     "Quadruped-Run-Feat-v0": 13e6,
    #     "Dog-Run-Feat-v0": 3e6,
    #     "Cartpole-Swingup-Feat-v0": 7e6,
    #     "Hopper-v3": 16e6,
    #     "Walker2d-v3": 31e6,
    #     "HalfCheetah-v3": 41e6,
    #     "Ant-v3": 21e6,
    #     "Humanoid-v3": 10e6,
    # }

    # args.num_timesteps = num_timesteps[args.env_id]

    # Initialize experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
    # Create experiment name
    experiment_name = experiment.get_name()

    # Configure experiment
    experiment.configure_logging()

    # get the updated args
    args = experiment.args
    print(args.num_timesteps)

    args.device = get_device(args)
    logger.info("device in use: {}".format(args.device))

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    worker_seed = args.seed + (1000000 * (rank + 1))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    eval_seed = args.seed + 1000000

    # initialize wandb
    if rank == 0:
        wandb_init(args, experiment_name)

    # Create environment
    env = make_env(args, worker_seed, experiment_name=experiment_name, mode="training")

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(
            args, eval_seed, experiment_name=experiment_name, mode="evaluation"
        )

    # Train
    orchestrator.learn(
        args=args,
        rank=rank,
        env=env,
        eval_env=eval_env,
        agent=agent_wrapper(env=env, device=args.device, args=args),
        experiment_name=experiment_name,
    )

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate(args):
    """Evaluate an agent"""

    # update args if wandb_run_path exists
    if args.wandb_run_path is not None:
        wandb_download(args)

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    # Create experiment name
    experiment_name = experiment.get_name()

    # Configure experiment
    experiment.configure_logging()

    # get the updated args
    args = experiment.args

    args.device = get_device(args)
    logger.info("device in use: {}".format(args.device))

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args, args.seed, experiment_name=experiment_name)

    # Evaluate
    orchestrator.evaluate(
        args=args,
        env=env,
        agent=agent_wrapper(env=env, device="cpu", args=args),
        experiment_name=experiment_name,
    )

    # Close environment
    env.close()


if __name__ == "__main__":
    args = Config(argparser().parse_args())
    args.__start_time = time.time()

    # Make the paths absolute
    args.root = osp.dirname(osp.abspath(__file__))

    if args.config_path is not None:
        with open(args.config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        # get default args
        defaults = vars(argparser().parse_args([]))
        args.update(config, defaults=defaults, change_uuid=args.resume_training)

    if args.iter_num is not None and args.iter_num.isdigit():
        args.iter_num = int(args.iter_num)

    if args.task == "train":
        train(args)
    elif args.task == "eval":
        evaluate(args)
    else:
        raise NotImplementedError
