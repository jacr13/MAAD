import math
from collections import OrderedDict
from copy import copy

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from gym import spaces

from helpers import logger
from helpers.distributed_util import RunMoms

STANDARDIZED_OB_CLAMPS = [-5.0, 5.0]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> GNN.


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0],
        max_index,
        dtype=torch.float32,
        device=indices.device,
    )
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    # Init empty result tensor.
    result = tensor.new_full(result_shape, 0).to(tensor.device)
    segment_ids = segment_ids.to(tensor.device).unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def snwrap(use_sn=False):
    """Spectral normalization wrapper"""

    def _snwrap(m):
        assert isinstance(m, nn.Linear)
        if use_sn:
            return U.spectral_norm(m)
        else:
            return m

    return _snwrap


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Distributional toolkits.


class NormalToolkit(object):
    @staticmethod
    def logp(x, mean, std):
        neglogp = (
            0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True)
            + 0.5 * math.log(2 * math.pi)
            + std.log().sum(dim=-1, keepdim=True)
        )
        return -neglogp

    @staticmethod
    def entropy(std):
        return (std.log() + 0.5 * math.log(2.0 * math.pi * math.e)).sum(
            dim=-1,
            keepdim=True,
        )

    @staticmethod
    def sample(mean, std):
        # Reparametrization trick
        eps = torch.empty(mean.size()).normal_().to(mean.device)
        return mean + std * eps

    @staticmethod
    def mode(mean):
        return mean

    @staticmethod
    def kl(mean, std, other_mean, other_std):
        return (
            other_std.log()
            - std.log()
            + (std.pow(2) + (mean - other_mean).pow(2)) / (2.0 * other_std.pow(2))
            - 0.5
        ).sum(dim=-1, keepdim=True)


class CatToolkit(object):
    @staticmethod
    def logp(x, logits):
        x = x[None] if len(x.size()) == 1 else x
        eye = torch.eye(logits.size()[-1]).to(logits.device)
        one_hot_ac = F.embedding(input=x.long(), weight=eye).to(logits.device)
        # Softmax loss (or Softmax Cross-Entropy loss)
        neglogp = -(one_hot_ac[:, 0, :].detach() * F.log_softmax(logits, dim=-1)).sum(
            dim=-1,
            keepdim=True,
        )
        return -neglogp

    @staticmethod
    def entropy(logits):
        a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        entropy = (p0 * (torch.log(z0) - a0)).sum(dim=-1)
        return entropy

    @staticmethod
    def sample(logits):
        # Gumbel-Max trick (>< Gumbel-Softmax trick)
        u = torch.empty(logits.size()).uniform_().to(logits.device)
        return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    @staticmethod
    def mode(logits):
        probs = torch.sigmoid(logits)
        return torch.argmax(probs, dim=-1)

    @staticmethod
    def kl(logits, other_logits):
        a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        a1 = other_logits - torch.max(other_logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        z1 = torch.sum(ea1, dim=-1, keepdim=True)
        p0 = ea0 / z0
        kl = (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(dim=-1)
        return kl


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Perception stacks.


class ShallowMLP(nn.Module):
    def __init__(self, env, hps, hidsize, extrahid=False):
        """MLP layer stack as usually used in Deep RL"""
        super(ShallowMLP, self).__init__()

        ob_dim = env.observation_space.shape
        if len(env.observation_space.shape) == 3:
            ob_dim = ob_dim[0] * ob_dim[1]
        else:
            ob_dim = ob_dim[0]

        self.extrahid = extrahid
        # Assemble fully-connected encoder
        self.encoder_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(ob_dim, hidsize)),
                                    ("ln", nn.LayerNorm(hidsize)),
                                    ("nl", nn.Tanh()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        if self.extrahid:
            self.encoder_2 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc_block",
                            nn.Sequential(
                                OrderedDict(
                                    [
                                        ("fc", nn.Linear(hidsize, hidsize)),
                                        ("ln", nn.LayerNorm(hidsize)),
                                        ("nl", nn.Tanh()),
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            )
            # Create skip connection
            self.skip_co = nn.Sequential()

        # Perform initialization
        self.apply(weight_init)

    def forward(self, x):
        # x = x.view(-1, x.shape[0])
        x = self.encoder_1(x)
        if self.extrahid:
            x = self.skip_co(x) + self.encoder_2(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ShallowCNN(nn.Module):
    def __init__(self, env, hps, hidsize, extrahid=False):
        super(ShallowCNN, self).__init__()

        self.ob_dim = env.observation_space.shape

        num_inputs = self.ob_dim[-1]
        self.extrahid = extrahid

        self.p_decoder = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, hidsize),
            nn.ReLU(),
        )

        # Perform initialization
        self.apply(weight_init)

    def forward(
        self,
        inputs,
    ):
        inputs = inputs.permute(0, 3, 1, 2)
        x = self.p_decoder(inputs / 255.0)
        return x


def perception_stack_parser(x):
    if "mlp" in x:
        _, hidsize = x.split("_")
        return (
            (lambda u, v: ShallowMLP(u, v, hidsize=int(hidsize), extrahid=False)),
            int(hidsize),
        )
    if "cnn" in x:
        _, hidsize = x.split("_")
        return (
            (lambda u, v: ShallowCNN(u, v, hidsize=int(hidsize), extrahid=False)),
            int(hidsize),
        )
    else:
        raise NotImplementedError("invalid perception stack")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Networks.


class GaussPolicy(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(GaussPolicy, self).__init__()
        ac_dim = env.action_space.shape[0]
        self.hps = hps

        if self.hps.p_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs

        # Define perception stack
        net_lambda, fc_in = perception_stack_parser(self.hps.p_perception_stack)
        self.perception_stack = net_lambda(env, self.hps)

        # Assemble the last layers and output heads
        self.p_decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(fc_in, fc_in)),
                                    ("ln", nn.LayerNorm(fc_in)),
                                    ("nl", nn.Tanh()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.p_head = nn.Linear(fc_in, ac_dim)

        if self.hps.shared_value:
            self.v_decoder = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc_block",
                            nn.Sequential(
                                OrderedDict(
                                    [
                                        ("fc", nn.Linear(fc_in, fc_in)),
                                        ("ln", nn.LayerNorm(fc_in)),
                                        ("nl", nn.Tanh()),
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            )
            modules = []
            modules.append(nn.Linear(fc_in, 1))
            if self.hps.osim_muscles:
                modules.append(nn.Sigmoid())
            self.v_head = nn.Sequential(*modules)

        # Perform initialization
        self.apply(weight_init)

        self.ac_logstd_head = nn.Parameter(
            torch.full((ac_dim,), math.log(self.hps.policy_std_init))
        )

    def dist(self, ob):
        ac_mu, ac_std = self.forward(ob)
        return td.Independent(td.Normal(ac_mu, ac_std), 1)

    def logp(self, ob, ac):
        out = self.forward(ob)
        return NormalToolkit.logp(ac, *out[0:2])  # mean, std

    def entropy(self, ob):
        out = self.forward(ob)
        return NormalToolkit.entropy(out[1])  # std

    def rsample(self, ob):
        # reparameterized samples allowing gradients
        out = self.forward(ob)
        ac = NormalToolkit.sample(*out[0:2])  # mean, std
        return ac

    def sample(self, ob, return_dist=False):
        with torch.no_grad():
            out = self.forward(ob)
            ac = NormalToolkit.sample(*out[0:2])  # mean, std
        if return_dist:
            return ac, out
        return ac

    def mode(self, ob):
        with torch.no_grad():
            out = self.forward(ob)
            ac = NormalToolkit.mode(out[0])  # mean
        return ac

    def kl(self, ob, other):
        assert isinstance(other, GaussPolicy)
        with torch.no_grad():
            out_a = self.forward(ob)
            out_b = other.forward(ob)
            kl = NormalToolkit.kl(*out_a[0:2], *out_b[0:2])  # mean, std
        return kl

    def value(self, ob):
        if self.hps.shared_value:
            out = self.forward(ob)
            return out[2]  # value
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        if self.hps.p_batch_norm:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)

        x = self.perception_stack(ob)

        if "mlp" in self.hps.p_perception_stack:
            if self.hps.shared_value:
                value = self.v_head(self.v_decoder(x))
                if self.hps.shared_value_policy_detached:
                    # Prevent the policy from sending gradient backwards beyond its decoder
                    x = x.detach()

            ac_mean = self.p_head(self.p_decoder(x))

        else:
            raise NotImplementedError

        ac_std = self.ac_logstd_head.expand_as(ac_mean).exp()
        out = [ac_mean, ac_std]
        if self.hps.shared_value:
            out.append(value)

        return out


class CatPolicy(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(CatPolicy, self).__init__()
        self.hps = hps
        ac_dim = env.action_space.n
        if self.hps.p_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs

        ob_dim = env.observation_space.shape

        net_lambda, fc_in = perception_stack_parser(self.hps.p_perception_stack)
        self.perception_stack = net_lambda(env, self.hps)

        # Assemble the last layers and output heads
        self.p_decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(fc_in, fc_in)),
                                    ("ln", nn.LayerNorm(fc_in)),
                                    ("nl", nn.Tanh()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.p_head = nn.Linear(fc_in, ac_dim)
        if self.hps.shared_value:
            # Policy and value share their feature extractor
            self.v_decoder = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc_block",
                            nn.Sequential(
                                OrderedDict(
                                    [
                                        ("fc", nn.Linear(fc_in, fc_in)),
                                        ("ln", nn.LayerNorm(fc_in)),
                                        ("nl", nn.Tanh()),
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            )
            self.v_head = nn.Linear(fc_in, 1)

        # Perform initialization
        self.apply(weight_init)

    def logp(self, ob, ac):
        out = self.forward(ob)
        return CatToolkit.logp(ac, out[0])  # ac_logits

    def entropy(self, ob):
        out = self.forward(ob)
        return CatToolkit.entropy(out[0])  # ac_logits

    def rsample(self, ob):
        # reparameterized samples allowing gradients
        out = self.forward(ob)
        ac = CatToolkit.sample(out[0])  # ac_logits
        return ac[:, None]

    def sample(self, ob):
        # Gumbel-Max trick (>< Gumbel-Softmax trick)
        with torch.no_grad():
            out = self.forward(ob)
            ac = CatToolkit.sample(out[0])  # ac_logits
        return ac

    def mode(self, ob):
        with torch.no_grad():
            out = self.forward(ob)
            ac = CatToolkit.mode(out[0])  # ac_logits
        return ac

    def kl(self, ob, other):
        assert isinstance(other, CatPolicy)
        with torch.no_grad():
            out_a = self.forward(ob)
            out_b = other.forward(ob)
            kl = CatToolkit.kl(out_a[0], out_b[0])  # ac_logits
        return kl

    def value(self, ob):
        if self.hps.shared_value:
            out = self.forward(ob)
            return out[1]  # value
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        if self.hps.p_batch_norm:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.perception_stack(ob)
        ac_logits = self.p_head(self.p_decoder(x))
        out = [ac_logits]
        if self.hps.shared_value:
            value = self.v_head(self.v_decoder(x))
            out.append(value)
        return out


class Value(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(Value, self).__init__()
        self.hps = hps

        if self.hps.v_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs

        # Define perception stack
        net_lambda, fc_in = perception_stack_parser(self.hps.v_perception_stack)
        self.perception_stack = net_lambda(env, self.hps)

        # Assemble the last layers and output heads
        self.v_decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(fc_in, fc_in)),
                                    ("ln", nn.LayerNorm(fc_in)),
                                    ("nl", nn.Tanh()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.v_head = nn.Linear(fc_in, 1)

        # Perform initialization
        self.apply(weight_init)

    def forward(self, ob):
        if self.hps.v_batch_norm:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.perception_stack(ob)

        value = self.v_head(self.v_decoder(x))
        return value


class Discriminator(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(Discriminator, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_space = env.action_space
        self.is_discrete = isinstance(ac_space, spaces.Discrete)

        ac_dim = ac_space.n if self.is_discrete else ac_space.shape[0]

        if hps.osim_muscles:
            ob_dim = env.observation_space_no_muscles.shape[0]

        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hps = hps
        self.leak = 0.1
        apply_sn = snwrap(use_sn=self.hps.spectral_norm)
        if self.hps.d_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
            if hps.osim_muscles:
                self.rms_obs = RunMoms(shape=ob_dim - 1, use_mpi=True)

        assert not (
            self.hps.state_only and self.hps.state_state
        ), "state_only and state_state should not be true at the same time"
        # Define the input dimension
        if self.hps.d_perception_stack.startswith("cnn"):
            ob_dim = int(self.hps.d_perception_stack.split("_")[-1])
        if self.hps.state_only:
            in_dim = ob_dim
        elif self.hps.state_state:
            in_dim = ob_dim + ob_dim
        else:
            in_dim = ob_dim + ac_dim

        # Assemble the layers and output heads
        net_lambda, hidsize = perception_stack_parser(self.hps.d_perception_stack)
        if self.hps.d_perception_stack.startswith("cnn"):
            self.perception_stack = net_lambda(env, self.hps)

        self.fc_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", apply_sn(nn.Linear(in_dim, hidsize))),
                                    ("nl", nn.LeakyReLU(negative_slope=self.leak)),
                                ],
                            ),
                        ),
                    ),
                    (
                        "fc_block_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", apply_sn(nn.Linear(hidsize, hidsize))),
                                    ("nl", nn.LeakyReLU(negative_slope=self.leak)),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.d_head = nn.Linear(hidsize, 1)

        # Perform initialization
        self.apply(weight_init)

    def D(self, input_a, input_b):
        return self.forward(input_a, input_b)

    def forward(self, input_a, input_b):
        if self.hps.osim_muscles:
            input_a = input_a[:, : self.ob_dim]
            if self.hps.state_state:
                input_b = input_b[:, : self.ob_dim]
        if self.hps.d_batch_norm:
            # Apply normalization
            if self.hps.wrap_absorb:
                # Normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(
                    *STANDARDIZED_OB_CLAMPS,
                )
                input_a = torch.cat([input_a_, input_a[:, -1].unsqueeze(-1)], dim=-1)
                if self.hps.state_state:
                    # Normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(
                        *STANDARDIZED_OB_CLAMPS,
                    )
                    input_b = torch.cat(
                        [input_b_, input_b[:, -1].unsqueeze(-1)],
                        dim=-1,
                    )
            else:
                # Normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(
                    *STANDARDIZED_OB_CLAMPS,
                )
                if self.hps.state_state:
                    # Normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(
                        *STANDARDIZED_OB_CLAMPS,
                    )
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.hps.state_state:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)

        if self.hps.d_perception_stack.startswith("cnn"):
            input_a = self.perception_stack(input_a)
            if self.hps.state_state:
                input_b = self.perception_stack(input_b)

        # Concatenate
        if self.hps.state_only:
            x = input_a
        else:
            x = torch.cat([input_a, input_b], dim=-1)

        x = self.fc_stack(x)
        score = self.d_head(x)  # no sigmoid here
        return score


class Actor(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        if self.hps.a_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs

        # Assemble the last layers and output heads
        hidsize = self.hps.actor_hidden_size
        self.fc_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(ob_dim, hidsize)),
                                    (
                                        "ln",
                                        (
                                            nn.LayerNorm
                                            if hps.layer_norm
                                            else nn.Identity
                                        )(hidsize),
                                    ),
                                    ("nl", nn.ReLU()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.a_fc_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(hidsize, hidsize)),
                                    (
                                        "ln",
                                        (
                                            nn.LayerNorm
                                            if hps.layer_norm
                                            else nn.Identity
                                        )(hidsize),
                                    ),
                                    ("nl", nn.ReLU()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.a_head = nn.Linear(hidsize, ac_dim)
        if self.hps.kye_p:
            self.r_fc_stack = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc_block_1",
                            nn.Sequential(
                                OrderedDict(
                                    [
                                        ("fc", nn.Linear(hidsize, hidsize)),
                                        (
                                            "ln",
                                            (
                                                nn.LayerNorm
                                                if hps.layer_norm
                                                else nn.Identity
                                            )(hidsize),
                                        ),
                                        ("nl", nn.ReLU()),
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            )
            self.r_head = nn.Linear(hidsize, 1)

        # Perform initialization
        self.apply(weight_init)

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def auxo(self, ob):
        if self.hps.kye_p:
            out = self.forward(ob)
            return out[1]  # aux
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        if self.hps.a_batch_norm:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_fc_stack(x)))
        out = [ac]
        if self.hps.kye_p:
            aux = self.r_head(self.r_fc_stack(x))
            out.append(aux)
        return out

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" in n]


class Critic(nn.Module):
    def __init__(self, env, hps, rms_obs):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.use_c51:
            num_heads = hps.c51_num_atoms
        elif hps.use_qr:
            num_heads = hps.num_tau
        else:
            num_heads = 1
        self.hps = hps
        if self.hps.c_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        hidsize = self.hps.critic_hidden_size
        self.fc_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_block_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(ob_dim + ac_dim, hidsize)),
                                    (
                                        "ln",
                                        (
                                            nn.LayerNorm
                                            if hps.layer_norm
                                            else nn.Identity
                                        )(hidsize),
                                    ),
                                    ("nl", nn.ReLU()),
                                ],
                            ),
                        ),
                    ),
                    (
                        "fc_block_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("fc", nn.Linear(hidsize, hidsize)),
                                    (
                                        "ln",
                                        (
                                            nn.LayerNorm
                                            if hps.layer_norm
                                            else nn.Identity
                                        )(hidsize),
                                    ),
                                    ("nl", nn.ReLU()),
                                ],
                            ),
                        ),
                    ),
                ],
            ),
        )
        self.head = nn.Linear(hidsize, num_heads)

        # Perform initialization
        self.apply(weight_init)

    def QZ(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        if self.hps.c_batch_norm:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x

    @property
    def out_params(self):
        return list(self.head.parameters())
