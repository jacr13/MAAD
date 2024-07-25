import numpy as np

from helpers.math_util import standardized
from agents.gae import gae, gae_v2, gae_v2_ev


def compute_advantages(rollout, gamma, lambda_, rew_key, method="gae_v2"):
    if method == "gae":
        return gae(rollout, gamma, lambda_, rew_key)
    elif method == "gae_v2":
        return gae_v2(rollout, gamma, lambda_, rew_key)
    elif method == "gae_v2_ev":
        return gae_v2_ev(rollout, gamma, lambda_, rew_key)
    else:
        ValueError(f"Unknown method ({method}) to compute advantage")


def postprocess_rollout(rollout, gamma, lambda_, rew_key, method="gae_v2"):
    assert len(rollout["next_v"]) == 1, "next_v should be equal to 1"
    rollout = compute_advantages(rollout, gamma, lambda_, rew_key, method=method)
    rollout["advs"] = standardized(rollout["advs"])
    return rollout


def concatenate_rollouts(rollouts, key=None):
    assert isinstance(rollouts, list), "rollouts should be a list"

    if len(rollouts) == 0:
        return rollouts

    if isinstance(rollouts[0], dict):
        assert key is not None, "key must be specified"
        assert key in rollouts[0].keys(), "key must be in rollouts list"
        rollout = {
            k: np.concatenate([rollout[key][k] for rollout in rollouts])
            for k in rollouts[0][key].keys()
        }
    elif isinstance(rollouts[0], list):
        assert isinstance(key, int), "key be the index"
        rollout = {
            k: np.concatenate([rollout[key][k] for rollout in rollouts])
            for k in rollouts[0][key].keys()
        }
    else:
        ValueError(f"no method implemented for rollouts type {type(rollouts)}")
    return rollout
