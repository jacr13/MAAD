"""This file is a modified version of https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_watch.py, to allow wandb.watch() to handle model names"""

import logging
import os
from typing import Optional

import torch
from matplotlib import pyplot as plt

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import wandb
from wandb.sdk.lib import telemetry

logger = logging.getLogger("wandb")

_global_watch_idx = 0


def log_params(
    models,
    log: Optional[Literal["gradients", "parameters", "all"]] = "gradients",
):
    """Manually log histograms of gradients and parameters to wandb."""
    assert isinstance(models, dict), "models must be a dict"

    for model_name, model in models.items():
        for param_name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if log in ("parameters", "all"):
                    wandb.run._torch.log_tensor_stats(
                        parameter, f"parameters/{model_name}/{param_name}"
                    )
                if log in ("gradients", "all"):
                    if parameter.grad is not None:
                        wandb.run._torch.log_tensor_stats(
                            parameter.grad.data, f"gradients/{model_name}/{param_name}"
                        )


def watch(
    models,
    model_names=None,
    criterion=None,
    log: Optional[Literal["gradients", "parameters", "all"]] = "gradients",
    log_freq: int = 1000,
    idx: Optional[int] = None,
    log_graph: bool = False,
):
    """Hook into the torch model to collect gradients and the topology.

    Should be extended to accept arbitrary ML models.

    Args:
        models: (torch.Module) The model to hook, can be a tuple or dict
        model_names: (str) The model names to watch, can be a tuple or dict
        criterion: (torch.F) An optional loss value being optimized
        log: (str) One of "gradients", "parameters", "all", or None
        log_freq: (int) log gradients and parameters every N batches
        idx: (int) an index to be used when calling wandb.watch on multiple models
        log_graph: (boolean) log graph topology

    Returns:
        `wandb.Graph`: The graph object that will populate after the first backward pass

    Raises:
        ValueError: If called before `wandb.init` or if any of models is not a torch.nn.Module.
    """
    global _global_watch_idx

    with telemetry.context() as tel:
        tel.feature.watch = True

    logger.info("Watching")

    if wandb.run is None:
        raise ValueError("You must call `wandb.init` before calling watch")

    if log not in {"gradients", "parameters", "all", None}:
        raise ValueError("log must be one of 'gradients', 'parameters', 'all', or None")

    log_parameters = log in {"parameters", "all"}
    log_gradients = log in {"gradients", "all"}

    # Ensure models are list or tuple if not dict
    if not isinstance(models, (tuple, list, dict)):
        models = (models,)

    # Set default model_names if not provided (only used if models are not a dict)
    if model_names is None:
        model_names = ["graph_{idx}"] * len(models)

    assert len(models) == len(
        model_names
    ), "The number of models must match the number of model names"

    if not isinstance(models, dict):
        models = {model_name: model for (model_name, model) in zip(model_names, models)}

    torch = wandb.util.get_module(
        "torch", required="wandb.watch only works with pytorch, couldn't import torch."
    )

    for model in models.values():
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Expected a pytorch model (torch.nn.Module). Received "
                + str(type(model))
            )

    graphs = []
    prefix = ""

    if idx is None:
        idx = _global_watch_idx

    for local_idx, (model_name, model) in enumerate(models.items()):
        global_idx = idx + local_idx
        _global_watch_idx += 1

        prefix = model_name.format(idx=global_idx) + "/"

        if log_parameters:
            wandb.run._torch.add_log_parameters_hook(
                model,
                prefix=prefix,
                log_freq=log_freq,
            )

        if log_gradients:
            wandb.run._torch.add_log_gradients_hook(
                model,
                prefix=prefix,
                log_freq=log_freq,
            )

        if log_graph:
            graph = wandb.run._torch.hook_torch(model, criterion, graph_idx=global_idx)
            graphs.append(graph)
            # NOTE: the graph is set in run.summary by hook_torch on the backward pass
    return graphs


def unwatch(models=None):
    """Remove pytorch model topology, gradient and parameter hooks.

    Args:
        models: (list) Optional list of pytorch models that have had watch called on them
    """
    if models:
        if not isinstance(models, (tuple, list)):
            models = (models,)
        for model in models:
            if not hasattr(model, "_wandb_hook_names"):
                wandb.termwarn("%s model has not been watched" % model)
            else:
                for name in model._wandb_hook_names:
                    wandb.run._torch.unhook(name)
                delattr(model, "_wandb_hook_names")
                # TODO: we should also remove recursively model._wandb_watch_called

    else:
        wandb.run._torch.unhook_all()


def _remove_infs_nans(tensor: "torch.Tensor") -> "torch.Tensor":
    if not torch.isfinite(tensor).all():
        tensor = tensor[torch.isfinite(tensor)]

    return tensor


def _no_finite_values(tensor: "torch.Tensor") -> bool:
    return tensor.shape == torch.Size([0]) or (~torch.isfinite(tensor)).all().item()


def show_tensor_hist(tensor, name, step, _is_cuda_histc_supported=None, _num_bins=64):
    """Add distribution statistics on a tensor's elements to the current History entry"""
    # TODO Handle the case of duplicate names.
    # print("in show")
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        while (isinstance(tensor, tuple) or isinstance(tensor, list)) and (
            isinstance(tensor[0], tuple) or isinstance(tensor[0], list)
        ):
            tensor = [item for sublist in tensor for item in sublist]
        tensor = torch.cat([t.reshape(-1) for t in tensor])

    # checking for inheritance from _TensorBase didn't work for some reason
    if not hasattr(tensor, "shape"):
        cls = type(tensor)
        raise TypeError(f"Expected Tensor, not {cls.__module__}.{cls.__name__}")

    # HalfTensors on cpu do not support view(), upconvert to 32bit
    if isinstance(tensor, torch.HalfTensor):
        tensor = tensor.clone().type(torch.FloatTensor).detach()

    # Sparse tensors have a bunch of implicit zeros. In order to histo them correctly,
    # we have to count them up and add them to the histo ourselves.
    sparse_zeros = None
    if tensor.is_sparse:
        # Have to call this on a sparse tensor before most other ops.
        tensor = tensor.cpu().coalesce().clone().detach()

        backing_values = tensor._values()
        non_zero_values = backing_values.numel()
        all_values = tensor.numel()
        sparse_zeros = all_values - non_zero_values
        tensor = backing_values

    flat = tensor.reshape(-1)

    if flat.is_cuda:
        # TODO(jhr): see if pytorch will accept something upstream to check cuda support for ops
        # until then, we are going to have to catch a specific exception to check for histc support.
        if _is_cuda_histc_supported is None:
            _is_cuda_histc_supported = True
            check = torch.cuda.FloatTensor(1).fill_(0)
            try:
                check = flat.histc(bins=_num_bins)
            except RuntimeError as e:
                # Only work around missing support with specific exception
                # if str(e).startswith("_th_histc is not implemented"):
                #    _is_cuda_histc_supported = False
                # On second thought, 0.4.1 doesnt have support and maybe there are other issues
                # lets disable more broadly for now
                _is_cuda_histc_supported = False

        if not _is_cuda_histc_supported:
            flat = flat.cpu().clone().detach()

        # As of torch 1.0.1.post2+nightly, float16 cuda summary ops are not supported (convert to float32)
        if isinstance(flat, torch.cuda.HalfTensor):
            flat = flat.clone().type(torch.cuda.FloatTensor).detach()

    if isinstance(flat, torch.HalfTensor):
        flat = flat.clone().type(torch.FloatTensor).detach()

    # Skip logging if all values are nan or inf or the tensor is empty.
    if _no_finite_values(flat):
        return

    # Remove nans and infs if present. There's no good way to represent that in histograms.
    flat = _remove_infs_nans(flat)

    tmin = flat.min().item()
    tmax = flat.max().item()
    if sparse_zeros:
        # If we've got zeros to add in, make sure zero is in the hist range.
        tmin = 0 if tmin > 0 else tmin
        tmax = 0 if tmax < 0 else tmax
    # Anecdotally, this can somehow happen sometimes. Maybe a precision error
    # in min()/max() above. Swap here to prevent a runtime error.
    if tmin > tmax:
        tmin, tmax = tmax, tmin
    tensor = flat.histc(bins=_num_bins, min=tmin, max=tmax)
    tensor = tensor.cpu().clone().detach()
    bins = torch.linspace(tmin, tmax, steps=_num_bins + 1)

    # Add back zeroes from a sparse tensor.
    if sparse_zeros:
        bins_np = bins.numpy()
        tensor_np = tensor.numpy()
        bin_idx = 0
        num_buckets = len(bins_np) - 1
        for i in range(num_buckets):
            start = bins_np[i]
            end = bins_np[i + 1]
            # There are 3 cases to consider here, all of which mean we've found the right bucket
            # 1. The bucket range contains zero.
            # 2. The bucket range lower bound *is* zero.
            # 3. This is the last bucket and the bucket range upper bound is zero.
            if (start <= 0 and end > 0) or (i == num_buckets - 1 and end == 0):
                bin_idx = i
                break

        tensor_np[bin_idx] += sparse_zeros
        tensor = torch.Tensor(tensor_np)
        bins = torch.Tensor(bins_np)

    # print(f"----------{step}-------------")
    # print(name, tensor.tolist(), bins.tolist())
    # print()
    plt.hist(bins.tolist())
    plt.title(name)
    os.makedirs(f"debug_fig", exist_ok=True)
    os.makedirs(f"debug_fig/{step}", exist_ok=True)
    plt.savefig(f"debug_fig/{step}/{name}.png")
    plt.clf()
