import os
import shutil
import sys

import gym

import environments
from helpers import logger


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    benchmark = None
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


def make_env(args, seed, experiment_name=None, mode="training"):
    """Create an environment"""

    benchmark = get_benchmark(args.env_id)

    if benchmark.startswith("dmc"):
        import helpers.dmc2gym as dmc2gym

        environment_kwargs = {}

        from_pixels = False
        frame_skip = 1
        height = 84
        width = 84
        channels_first = True

        domain_name, task_name, obs_type = args.env_id.lower().split("-")[:3]

        if obs_type == "pixels":
            from_pixels = True
        elif obs_type == "feat":
            from_pixels = False
        else:
            raise ValueError("Observation type not supported")

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            from_pixels=from_pixels,
            frame_skip=frame_skip,
            height=height,
            width=width,
            channels_first=channels_first,
            seed=seed,
            environment_kwargs=environment_kwargs,
        )

        return env

    elif benchmark == "mujoco":
        # Remove the lockfile if it exists
        conda_path = shutil.which("conda")
        # remove /bin/conda on linux and windows
        conda_path = conda_path.replace("/bin/conda", "").replace("\\bin\\conda", "")
        lockfile = os.path.join(
            conda_path,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "mujoco_py",
            "generated",
            "mujocopy-buildlock.lock",
        )
        try:
            os.remove(lockfile)
            logger.warn("[WARN] Removed mujoco lockfile successfully")
        except OSError:
            logger.warn("[WARN] FAILLED to remove mujoco lockfile")
            pass

        if args.env_id in environments.MYO_SUITE:
            import myosuite  # noqa: F401
        env = gym.make(args.env_id)
        env.seed(seed)  # weird, but struct kept general if adding other envs

    else:
        raise ValueError("unsupported benchmark")
    return env
