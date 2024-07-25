import time
from contextlib import contextmanager

from helpers.misc_util import prettify_time, zipsame


def colorize(string, color, bold=False, highlight=False):
    color2num = {
        "gray": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "crimson": 38,
    }
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def log_env_info(logger, env):
    logger.info(4 * ">" + ' logging env "{}" specs'.format(env))
    shapes = {
        "ob_shape": env.observation_space.shape,
        "ac_shape": (
            env.action_space.n
            if hasattr(env.action_space, "n")
            else env.action_space.shape
        ),
    }
    logger.info("[INFO] shapes: {}.".format(shapes))


def log_module_info(logger, name, model):
    def _fmt(n):
        if n // 10**6 > 0:
            return str(round(n / 10**6, 2)) + " M"
        elif n // 10**3:
            return str(round(n / 10**3, 2)) + " k"
        else:
            return str(n)

    logger.info(4 * ">" + " logging {} specs".format(name))

    lists = [
        list(model.state_dict()),
        [list(model.state_dict()[x].size()) for x in model.state_dict()],
        [model.state_dict()[x].numel() for x in model.state_dict()],
        [
            True
            if x in [n for (n, p) in model.named_parameters() if p.requires_grad]
            else False
            for x in model.state_dict()
        ],
    ]
    for i, (p, s, n, t) in enumerate(zipsame(*lists)):
        logger.info(
            "[INFO] ({}:".format(str(i).zfill(2))
            + " param_name={},".format(str(p))
            + " size={},".format(str(s))
            + " numel={},".format(str(n))
            + " trainable={})".format(str(t)),
        )
    num_params = [p.numel() for (n, p) in model.named_parameters() if p.requires_grad]
    logger.info("[INFO] trainable params: {}.".format(_fmt(sum(num_params))))


def timed_cm_wrapper(logger, color_message="magenta", color_elapsed_time="cyan"):
    """Wraps a context manager that records the time taken by encapsulated ops"""

    @contextmanager
    def _timed(message):
        """Display the time it took for the mpi master
        to perform the task within the context manager
        """
        logger.info(
            colorize(">>>> {}".format(message).ljust(50, "."), color=color_message),
        )
        tstart = time.time()
        yield
        logger.info(
            colorize(
                "[done in {:.3f} seconds]".format(time.time() - tstart).rjust(50, "."),
                color=color_elapsed_time,
            ),
        )

    return _timed


def log_iter_info(logger, iters_so_far, num_iters, tstart, timesteps=None):
    """Display the current iteration and elapsed time"""
    elapsed = prettify_time(time.time() - tstart)
    fmtstr = " iteration [{}/{}] {}| elapsed time: {}"

    timesteps = f"| [{timesteps}] " if timesteps else ""

    logger.info(
        colorize(
            fmtstr.format(iters_so_far, num_iters, timesteps, elapsed).rjust(75, ">"),
            color="blue",
        ),
    )
