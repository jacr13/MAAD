import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
import torch.nn.utils as U
from gym import spaces
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from agents import get_kwarg
from agents.algos.base_agent import BaseAgent, get_local_rank
from agents.mine import Mine
from agents.models.inv_dyn_model import get_idm
from agents.nets import CatPolicy, Discriminator, GaussPolicy, Value
from agents.postprocessing import concatenate_rollouts, postprocess_rollout
from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.dataset import Dataset
from helpers.distributed_util import (
    RunMoms,
    average_gradients,
    mpi_mean_reduce,
    sync_data,
    sync_weights,
)
from helpers.math_util import LRDecay, gradients_norm


class GAILAgent(BaseAgent):
    def __init__(self, env, device, hps, expert_dataset):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[-1]  # num dims
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]

        self.device = "cpu"
        self._device = device

        self.hps = hps
        self.expert_dataset = expert_dataset

        self.simulator_steps = 0

        if self.hps.clip_norm <= 0:
            logger.info(
                "[WARN] clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm),
            )

        # Create observation normalizer that maintains running statistics
        self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

        assert self.hps.shared_value or not self.hps.shared_value_policy_detached

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps, self.rms_obs).to(self.device)

        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps, self.rms_obs).to(self.device)

        self.discriminator = Discriminator(self.env, self.hps, self.rms_obs).to(
            self.device,
        )

        if self.hps.algo == "idmm":
            input_size = self.ob_dim + (self.ac_dim + self.ob_dim)
            self.mine = Mine(input_size, self.hps, self.rms_obs)

        if self.hps.algo == "maad":
            self.debug_true_actions = self.expert_dataset.data.pop("acs", None)
            self.inv_dyn_model = get_idm(self.hps.idm_model)(
                self.ob_dim, self.ac_dim, self.hps, self.rms_obs, self.ob_shape
            )

        self._sync_weights(all_=True)

        # Set up demonstrations dataset
        self.setup_expert_dataloader(
            self.expert_dataset, self.hps.batch_size, self.device
        )

        # Set up the optimizers
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)
        if not self.hps.shared_value:
            self.v_optimizer = torch.optim.Adam(
                self.value.parameters(),
                lr=self.hps.v_lr,
            )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hps.d_lr,
        )

        self.lr_decay = LRDecay(self.hps, self.hps.p_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.p_optimizer,
            self.lr_decay,
        )

        log_module_info(logger, "policy", self.policy)
        if not self.hps.shared_value:
            log_module_info(logger, "value", self.value)
        log_module_info(logger, "discriminator", self.discriminator)

        # Set up wandb.watch on all models
        # TODO: For now watch is not logging parameters for all models, please use self.log_model_stats() instead. (cf do_iteration)
        # self.watch()

    def _sync_weights(
        self,
        all_=False,
        policy=False,
        value=False,
        discriminator=False,
        mine=False,
        inv_model=False,
    ):
        if all_ or policy:
            sync_weights(self.policy, msg="[policy]")

        if not self.hps.shared_value:
            if all_ or value:
                sync_weights(self.value, msg="[value]")

        if all_ or discriminator:
            sync_weights(self.discriminator, msg="[discriminator]")

        if self.hps.algo == "idmm":
            if all_ or mine:
                sync_weights(self.mine, msg="[mine]")

        if self.hps.algo == "maad":
            if all_ or inv_model:
                sync_weights(self.inv_dyn_model, msg="[inv_dyn_model]")

    def to(self, device):
        if device != self.device:
            self.device = device
            self.policy = self.policy.to(device)
            if not self.hps.shared_value:
                self.value = self.value.to(device)
            self.discriminator = self.discriminator.to(device)
            if self.hps.algo == "idmm":
                self.mine = self.mine.to(device)
            if self.hps.algo == "maad":
                self.inv_dyn_model = self.inv_dyn_model.to(device)

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        if self.hps.shared_value:
            v = self.policy.value(ob).cpu().detach().numpy().flatten()
        else:
            v = self.value(ob).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, v, logp

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(
                torch.eq(
                    row,
                    torch.cat(
                        [
                            torch.zeros_like(row[0:-1]),
                            torch.Tensor([1.0]).to(self.device),
                        ],
                        dim=-1,
                    ),
                ),
            ):
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def pretrain(self):
        # Container for all the metrics
        metrics = defaultdict(list)

        loss_fn = torch.nn.MSELoss()

        logger.info("[INFO] pretraining policy")
        for epoch in range(self.hps.pretrain_epochs):
            loss_history = []
            for e_batch in self.e_dataloader:
                # Transfer to device
                state = e_batch["obs_pre"].to(self.device)
                action = e_batch["acs"].to(self.device)
                if self.hps.wrap_absorb:
                    state = state[:, :-1]
                    action = action[:, :-1]

                self.rms_obs.update(state)

                mean, std = self.policy(state)
                # Reparametrization trick
                # reparam. trick -> x = mean + std * sample(N(0,1))
                eps = torch.empty(mean.size()).normal_().to(mean.device)
                pred_action = mean + std * eps
                p_loss = loss_fn(pred_action, action)

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.p_optimizer.step()
                loss_history.append(p_loss.item())

            logger.info(
                f"Epoch {epoch+1}/{self.hps.pretrain_epochs}, loss: {np.mean(loss_history)}",
            )
            metrics["pretrain/loss"].append(np.mean(loss_history))
        return metrics

    def update_policy_value(self, p_rollout, iters_so_far):
        """Update the policy and value networks"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # INFO: This part was replaced by the post processing method
        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return gae(
        # p_rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='syn_rews')

        # Standardize advantage function estimate
        # given by p_rollout['advs'] = standardized(p_rollout['advs'])

        # Create DataLoader objects to iterate over transitions in rollouts
        p_keys = [
            "obs0",
            "obs1",
            "acs",
            "logps",
            "vs",
            "advs",
            "td_lam_rets",
        ]
        p_dataset = Dataset({k: p_rollout[k] for k in p_keys})
        p_dataloader = DataLoader(
            p_dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        if self.hps.algo == "maad":
            if not self.hps.idm_pretrained:
                logger.info("[INFO] updating inverse dynamic model")
                idm_metrics = self.inv_dyn_model.train_until_converged(p_dataloader)
                metrics.update(idm_metrics)

            self.expert_dataset.data.pop("acs", None)
            obs_before = torch.Tensor(self.expert_dataset.data["obs0"])
            obs_after = torch.Tensor(self.expert_dataset.data["obs1"])

            if self.hps.wrap_absorb:
                obs_before = obs_before[:, :-1]
                obs_after = obs_after[:, :-1]

            actions_sampled = self.inv_dyn_model.sample(obs_before, obs_after)

            if self.hps.wrap_absorb:
                act_shape = list(actions_sampled.shape)
                act_shape[-1] += 1
                actions_ = torch.zeros(act_shape)
                actions_[:, 0:-1] = actions_sampled
                actions_sampled = actions_

            self.expert_dataset.data["acs"] = actions_sampled.cpu().detach().numpy()
            self.setup_expert_dataloader(
                self.expert_dataset, self.hps.batch_size, self.device
            )

        logger.info("[INFO] updating policy")
        _start = time.time()
        for _ in range(self.hps.optim_epochs_per_iter):
            debug_dist = True
            for p_batch in p_dataloader:  # go through the whole p_rollout
                # Transfer to device
                state = p_batch["obs0"].to(self.device)
                action = p_batch["acs"].to(self.device)
                logp_old = p_batch["logps"].to(self.device)
                v_old = p_batch["vs"].to(self.device)
                advantage = p_batch["advs"].to(self.device)
                td_lam_return = p_batch["td_lam_rets"].to(self.device)

                # Update the observation normalizer
                self.rms_obs.update(state)

                # Policy loss
                entropy_loss = (
                    -self.hps.p_ent_reg_scale * self.policy.entropy(state).mean()
                )
                logp = self.policy.logp(state, action)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(
                    1.0 - self.hps.eps,
                    1.0 + self.hps.eps,
                )
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = 0.5 * (logp - logp_old).pow(2).mean()
                kl_max = 0.5 * (logp - logp_old).pow(2).max()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                # Value loss
                if self.hps.shared_value:
                    v = self.policy.value(state)
                else:
                    v = self.value(state)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                v_loss_a = (clip_v - td_lam_return).pow(2)
                v_loss_b = (v - td_lam_return).pow(2)
                v_loss = torch.max(v_loss_a, v_loss_b).mean()
                if self.hps.shared_value:
                    p_loss = (
                        clip_loss + entropy_loss + (self.hps.baseline_scale * v_loss)
                    )
                else:
                    p_loss = clip_loss + entropy_loss
                if self.hps.algo == "idmm":
                    x = state
                    y = torch.cat([p_batch["obs1"].to(self.device), action], dim=-1)
                    mi = self.mine(x, y)  # state_after, action
                    p_loss -= self.hps.p_mi_reg_scale * mi
                    metrics["mi"].append(mi)

                if self.hps.algo in ["gail-bc", "maad"]:
                    state = torch.Tensor(self.expert_dataset.data["obs0"]).to(
                        self.device
                    )
                    if self.hps.algo == "gail-bc":
                        action_e = torch.Tensor(self.expert_dataset.data["acs"]).to(
                            self.device
                        )
                    elif self.hps.algo == "maad":
                        state_before = state
                        state_after = torch.Tensor(self.expert_dataset.data["obs1"]).to(
                            self.device
                        )
                        if self.hps.wrap_absorb:
                            state_before = state[:, :-1]
                            state_after = state_after[:, :-1]
                        action_e, actions_e_dist = self.inv_dyn_model.predict(
                            state_before, state_after, return_distribution=True
                        )

                    if self.hps.wrap_absorb:
                        state = state[:, :-1]
                        action_e = (
                            action_e[:, :-1] if self.hps.algo == "gail-bc" else action_e
                        )

                    bc_loss = 0
                    if self.hps.gail_bc_loss == "mse":
                        # Reparametrization trick
                        # reparam. trick -> x = mean + std * sample(N(0,1))
                        ac_sampled = self.policy.rsample(state)
                        bc_loss = F.mse_loss(ac_sampled, action_e)

                    elif self.hps.gail_bc_loss == "nll":
                        log_probs = self.policy.logp(state_before, action_e)
                        bc_loss = -torch.logsumexp(log_probs, dim=1).mean()

                    elif self.hps.gail_bc_loss.endswith("kl"):
                        policy_dist = self.policy.dist(state)

                        if self.hps.gail_bc_loss == "forward_kl":
                            # sample action from inverse dynamics model = action_e
                            ac_sampled = action_e
                            dist_a = actions_e_dist
                            dist_b = policy_dist
                        elif self.hps.gail_bc_loss == "reverse_kl":
                            # sample action from policy
                            ac_sampled = policy_dist.sample()
                            dist_a = policy_dist
                            dist_b = actions_e_dist
                        else:
                            raise ValueError(
                                f"BC loss '{self.hps.gail_bc_loss}' is not supported"
                            )

                        # Check if distributions are instances of torch.distributions.independent.Independent
                        if isinstance(
                            dist_a, torch.distributions.Independent
                        ) and isinstance(dist_b, torch.distributions.Independent):
                            # If yes, directly compute KL divergence
                            kl = dist.kl_divergence(dist_a, dist_b).mean()
                        else:
                            # If not, approximate KL divergence using Monte Carlo estimation
                            logp_a = dist_a.log_prob(ac_sampled).mean()
                            logp_b = dist_b.log_prob(ac_sampled).mean()
                            kl = logp_a - logp_b

                        # compute rescaling
                        if hasattr(dist_b, "base_dist"):
                            scale = dist_b.base_dist.scale
                        else:
                            scale = dist_b._component_distribution.base_dist.scale

                        if hasattr(dist_a, "base_dist"):
                            scale_a = dist_a.base_dist.scale
                        else:
                            scale_a = dist_a._component_distribution.base_dist.scale

                        if self.hps.gail_bc_loss == "forward_kl":
                            metrics["gail_bc/sca_p_mean"].append(scale.mean().item())
                            metrics["gail_bc/sca_p_std"].append(scale.std().item())
                            metrics["gail_bc/sca_i_mean"].append(scale_a.mean().item())
                            metrics["gail_bc/sca_i_std"].append(scale_a.std().item())

                        elif self.hps.gail_bc_loss == "reverse_kl":
                            metrics["gail_bc/sca_i_mean"].append(scale.mean().item())
                            metrics["gail_bc/sca_i_std"].append(scale.std().item())
                            metrics["gail_bc/sca_p_mean"].append(scale_a.mean().item())
                            metrics["gail_bc/sca_p_std"].append(scale_a.std().item())

                        metrics["gail_bc/kl_estimated"].append(kl.item())
                        weight_kl = (2 * scale.detach() ** 2).mean()
                        metrics["gail_bc/weight_kl"].append(weight_kl.item())
                        bc_loss = weight_kl * kl

                    else:
                        raise ValueError(
                            f"Gail-BC loss '{self.hps.gail_bc_loss}' not supported."
                        )

                    metrics["gail_bc/loss"].append(self.hps.gail_bc_lambda * bc_loss)
                    metrics["gail_bc/p_loss_simple"].append(p_loss.item())
                    metrics["gail_bc/loss_no_lambda"].append(bc_loss.item())

                    # add bc loss to policy loss
                    p_loss += self.hps.gail_bc_lambda * bc_loss

                # Log metrics
                metrics["entropy_loss"].append(entropy_loss)
                metrics["clip_loss"].append(clip_loss.item())
                metrics["kl_approx"].append(kl_approx)
                metrics["kl_max"].append(kl_max)
                metrics["clip_frac"].append(clip_frac)
                metrics["v_loss"].append(v_loss)
                metrics["p_loss"].append(p_loss.item())

                metrics["ratio_mean"].append(ratio.mean().item())
                metrics["ratio_std"].append(ratio.std().item())
                metrics["surrogate_loss_a_mean"].append(surrogate_loss_a.mean().item())
                metrics["surrogate_loss_a_std"].append(surrogate_loss_a.std().item())
                metrics["surrogate_loss_b_mean"].append(surrogate_loss_b.mean().item())
                metrics["surrogate_loss_b_std"].append(surrogate_loss_b.std().item())
                metrics["logp_mean"].append(logp.mean().item())
                metrics["logp_std"].append(logp.std().item())
                metrics["logp_old_mean"].append(logp_old.mean().item())
                metrics["logp_old_std"].append(logp_old.std().item())

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                if not self.hps.trainer_worker:
                    average_gradients(self.policy, self.device)
                metrics["norm_before_clip"].append(gradients_norm(self.policy))
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                metrics["norm_after_clip"].append(gradients_norm(self.policy))
                self.p_optimizer.step()
                if not self.hps.shared_value:
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    if not self.hps.trainer_worker:
                        average_gradients(self.value, self.device)
                    self.v_optimizer.step()

        self.scheduler.step()

        metrics = self.process_metrics(metrics)
        return metrics, self.scheduler.get_last_lr()

    def update_discriminator(self, d_rollout, iters_so_far):
        """Update the discriminator network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Create DataLoader object to iterate over transitions in rollouts
        d_keys = ["obs0"]
        if self.hps.state_state:
            if self.hps.n_step_returns and "sam" in self.hps.algo:
                d_keys.append("obs1_td1")
            else:
                d_keys.append("obs1")
        else:
            d_keys.append("acs")

        d_dataset = Dataset({k: d_rollout[k] for k in d_keys})
        d_dataloader = DataLoader(
            d_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )

        logger.info("[INFO] updating discriminator")
        for e_batch in self.e_dataloader:
            # Get a minibatch of policy data
            d_batch = next(iter(d_dataloader))

            # Transfer to device
            p_input_a = d_batch["obs0"].to(self.device)[:, : self.discriminator.ob_dim]
            e_input_a = e_batch["obs0"].to(self.device)

            if self.hps.state_state:
                if self.hps.n_step_returns:
                    p_input_b = d_batch["obs1_td1"].to(self.device)
                else:
                    p_input_b = d_batch["obs1"][:, : self.discriminator.ob_dim].to(
                        self.device,
                    )
                e_input_b = e_batch["obs1"].to(self.device)
            else:
                p_input_b = d_batch["acs"].to(self.device)
                e_input_b = e_batch["acs"].to(self.device)
                if self.is_discrete:
                    p_input_b = (
                        F.one_hot(p_input_b.long(), num_classes=self.ac_dim)
                        .float()
                        .view(-1, self.ac_dim)
                    )
                    e_input_b = (
                        F.one_hot(e_input_b.long(), num_classes=self.ac_dim)
                        .float()
                        .view(-1, self.ac_dim)
                    )

            # Update running moments
            _state = torch.cat([p_input_a, e_input_a], dim=0)
            if self.hps.wrap_absorb:
                _state = self.remove_absorbing(_state)[0][:, 0:-1]
            if self.hps.d_batch_norm:
                self.discriminator.rms_obs.update(_state)

            # Compute scores
            p_scores = self.discriminator.D(p_input_a, p_input_b)
            e_scores = self.discriminator.D(e_input_a, e_input_b)

            # Create entropy loss
            scores = torch.cat([p_scores, e_scores], dim=0)
            entropy = F.binary_cross_entropy_with_logits(
                input=scores,
                target=torch.sigmoid(scores),
            )
            entropy_loss = -self.hps.d_ent_reg_scale * entropy

            # Create labels
            fake_labels = 0.0 * torch.ones_like(p_scores).to(self.device)
            real_labels = 1.0 * torch.ones_like(e_scores).to(self.device)

            # Create positive-negative binary classification (cross-entropy) losses
            p_loss = F.binary_cross_entropy_with_logits(
                input=p_scores,
                target=fake_labels,
                reduction="none",
            )
            e_loss = F.binary_cross_entropy_with_logits(
                input=e_scores,
                target=real_labels,
                reduction="none",
            )
            p_e_loss = p_loss + e_loss
            # Averate out over the batch
            p_e_loss = p_e_loss.mean()

            # Aggregated loss
            d_loss = p_e_loss + entropy_loss

            # Log metrics
            metrics["entropy_loss"].append(entropy_loss)
            metrics["p_e_loss"].append(p_e_loss)
            metrics["p_loss"].append(p_loss)
            metrics["e_loss"].append(e_loss)

            if self.hps.grad_pen:
                # Create gradient penalty loss (coefficient from the original paper)
                grad_pen = 10.0 * self.grad_pen(
                    p_input_a,
                    p_input_b,
                    e_input_a,
                    e_input_b,
                )
                d_loss += grad_pen
                # Log metrics
                metrics["grad_pen"].append(grad_pen)

            metrics["d_loss"].append(d_loss)

            # Update parameters
            self.d_optimizer.zero_grad()
            d_loss.backward()
            if not self.hps.trainer_worker:
                average_gradients(self.discriminator, self.device)
            self.d_optimizer.step()

        metrics = {
            k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()
        }
        return metrics

    def grad_pen(self, p_input_a, p_input_b, e_input_a, e_input_b):
        """Gradient penalty regularizer (motivation from Wasserstein GANs (Gulrajani),
        but empirically useful in JS-GANs (Lucic et al. 2017)) and later in (Karol et al. 2018).
        """
        # Assemble interpolated state-action pair
        ob_shape = (
            self.discriminator.perception_stack.ob_dim
            if hasattr(self.discriminator, "perception_stack")
            else self.discriminator.ob_dim
        )
        if self.hps.state_state:
            input_a_dim = ob_shape
            input_b_dim = ob_shape
        else:
            input_a_dim = ob_shape
            input_b_dim = self.discriminator.ac_dim

        """
        if self.hps.wrap_absorb:
            input_a_dim += 1
            input_b_dim += 1
        """

        input_a_eps = torch.rand(input_a_dim).to(p_input_a.device)
        input_b_eps = torch.rand(input_b_dim).to(p_input_a.device)

        input_a_interp = input_a_eps * p_input_a + ((1.0 - input_a_eps) * e_input_a)
        input_b_interp = input_b_eps * p_input_b + ((1.0 - input_b_eps) * e_input_b)

        # Set `requires_grad=True` to later have access to
        # gradients w.r.t. the inputs (not populated by default)
        input_a_interp = Variable(input_a_interp, requires_grad=True)
        input_b_interp = Variable(input_b_interp, requires_grad=True)

        # Create the operation of interest
        score = self.discriminator.D(input_a_interp, input_b_interp)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(
            outputs=score,
            inputs=[input_a_interp, input_b_interp],
            only_inputs=True,
            grad_outputs=torch.ones(score.size()).to(p_input_a.device),
            retain_graph=True,
            create_graph=True,
            allow_unused=self.hps.state_only,
        )
        assert len(list(grads)) == 2, "length must be exactly 2"

        # Return the gradient penalty (try to induce 1-Lipschitzness)
        if self.hps.state_only:
            grads = grads[0]

        grads = [grad.reshape(grad.shape[0], -1) for grad in grads]
        grads_concat = torch.cat(list(grads), dim=-1)
        return (grads_concat.norm(2, dim=-1) - 1.0).pow(2).mean()

    def get_reward(self, curr_ob, ac, next_ob):
        # Define the obeservation to get the reward of
        # Craft surrogate reward
        assert sum([isinstance(x, torch.Tensor) for x in [curr_ob, ac, next_ob]]) in [
            0,
            3,
        ]
        if not isinstance(curr_ob, torch.Tensor):  # then ac is not neither
            curr_ob = torch.Tensor(curr_ob)
            ac = torch.Tensor(ac)
            next_ob = torch.Tensor(next_ob)

        # Transfer to cpu
        curr_ob = curr_ob.cpu()
        ac = ac.cpu()
        next_ob = next_ob.cpu()

        if self.hps.state_state:
            input_a = curr_ob
            input_b = next_ob
        else:
            input_a = curr_ob
            input_b = ac
            if self.is_discrete:
                input_b = (
                    F.one_hot(input_b.long(), num_classes=self.ac_dim)
                    .float()
                    .view(-1, self.ac_dim)
                )

        # Compure score
        score = self.discriminator.D(input_a, input_b).detach().view(-1, 1)
        # Counterpart of GAN's minimax (also called "saturating") loss
        # Numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1.0 - torch.sigmoid(score) + 1e-8)
        if self.hps.minimax_only:
            reward = minimax_reward
        else:
            # Counterpart of GAN's non-saturating loss
            # Recommended in the original GAN paper and later in (Fedus et al. 2017)
            # Numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = F.logsigmoid(score)
            # Return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # Numerics: might be better might be way worse
            reward = non_satur_reward + minimax_reward
        return self.hps.syn_rew_scale * reward

    def rollout_generator(self, rollout_len=None):
        rollout_len = rollout_len if rollout_len else self.hps.rollout_len

        self.simulator_steps = 0
        done = True
        # Reset agent's env
        ob = np.array(self.env.reset())
        # Init collections
        p_rollout = defaultdict(list)
        d_rollout = defaultdict(list)
        # Init current episode statistics
        cur_ep_len = 0
        cur_ep_env_ret = 0
        cur_ep_syn_ret = 0

        if self.is_discrete:
            ac_dim_ = 1
        else:
            ac_dim_ = self.ac_dim

        while True:
            # Predict
            ac, v, logp = self.predict(ob, True)

            if not isinstance(self.ac_space, spaces.Discrete):
                # NaN-proof and clip
                ac = np.nan_to_num(ac)
                ac = np.clip(ac, self.env.action_space.low, self.env.action_space.high)
            else:
                ac = ac if isinstance(ac, int) else ac.item()

            # (t > 0 and t % rollout_len == 0)
            if self.simulator_steps > 0 and self.simulator_steps % rollout_len == 0:
                for k in p_rollout.keys():
                    if k in ["obs0", "obs1"]:
                        p_rollout[k] = np.array(p_rollout[k]).reshape(
                            -1, *self.ob_shape
                        )
                    elif k == "acs":
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, ac_dim_)
                    elif k in ["vs", "logps", "env_rews", "syn_rews", "dones"]:
                        p_rollout[k] = np.array(p_rollout[k]).reshape(-1, 1)
                    else:
                        p_rollout[k] = np.array(p_rollout[k])
                p_rollout["next_v"].append(v * (1 - done))

                if self.hps.wrap_absorb:
                    ob_dim = self.ob_dim + 1
                    ac_dim = ac_dim_ + 1
                else:
                    ob_dim = self.ob_dim
                    ac_dim = ac_dim_
                for k in d_rollout.keys():
                    if k in ["obs0", "obs1"]:
                        d_rollout[k] = np.array(d_rollout[k]).reshape(
                            -1, *self.ob_shape
                        )
                    elif k == "acs":
                        d_rollout[k] = np.array(d_rollout[k]).reshape(-1, ac_dim)

                yield p_rollout, d_rollout

                # Clear the collections
                p_rollout.clear()
                d_rollout.clear()
                self.simulator_steps = 0

            # Interact with env(s)
            step = self.env.step(ac)
            if len(step) == 4:
                new_ob, env_rew, done, _ = step
            elif len(step) == 5:
                new_ob, env_rew, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                raise ValueError(f"unexpected step length: {len(step)}")
            self.simulator_steps += 1

            # Populate collections
            p_rollout["obs0"].append(ob)
            p_rollout["acs"].append(ac)
            p_rollout["obs1"].append(new_ob)
            p_rollout["vs"].append(v)
            p_rollout["logps"].append(logp)
            p_rollout["env_rews"].append(env_rew)
            p_rollout["dones"].append(done)

            if self.hps.wrap_absorb:
                _ob = np.append(ob, 0)
                _ac = np.append(ac, 0)
                d_rollout["obs0"].append(_ob)
                d_rollout["acs"].append(_ac)
                if done and not self.env._elapsed_steps == self.env._max_episode_steps:
                    # Wrap with an absorbing state
                    _new_ob = np.append(np.zeros(self.ob_shape), 1)
                    d_rollout["obs1"].append(_new_ob)
                    # Add absorbing transition
                    d_rollout["obs0"].append(np.append(np.zeros(self.ob_shape), 1))
                    d_rollout["acs"].append(np.append(np.zeros(self.ac_shape), 1))
                    d_rollout["obs1"].append(np.append(np.zeros(self.ob_shape), 1))
                else:
                    _new_ob = np.append(new_ob, 0)
                    d_rollout["obs1"].append(_new_ob)
                # Get synthetic rewards
                syn_rew = self.get_reward(_ob[None], _ac[None], _new_ob[None])
            else:
                d_rollout["obs0"].append(ob)
                d_rollout["acs"].append(ac)
                d_rollout["obs1"].append(new_ob)
                # Get synthetic rewards
                ob = np.array(ob)
                new_ob = np.array(new_ob)
                ac = np.array(ac)
                syn_rew = self.get_reward(ob[None], ac[None], new_ob[None])
            syn_rew = syn_rew.detach().cpu().numpy().flatten().item()
            p_rollout["syn_rews"].append(syn_rew)

            # Update current episode statistics
            cur_ep_len += 1
            assert self.env._elapsed_steps == cur_ep_len  # sanity check
            cur_ep_env_ret += env_rew
            cur_ep_syn_ret += syn_rew

            # Set current state with the next
            ob = np.array(deepcopy(new_ob))

            if done:
                # Update the global episodic statistics and
                # reset current episode statistics
                p_rollout["ep_lens"].append(cur_ep_len)
                cur_ep_len = 0
                p_rollout["ep_env_rets"].append(cur_ep_env_ret)
                cur_ep_env_ret = 0
                p_rollout["ep_syn_rets"].append(cur_ep_syn_ret)
                cur_ep_syn_ret = 0
                # Reset env
                ob = np.array(self.env.reset())

    def do_iteration(self, **kwargs):
        timed = get_kwarg(kwargs, "timed")
        roll_gen = get_kwarg(kwargs, "roll_gen")
        iters_so_far = get_kwarg(kwargs, "iters_so_far")
        d = get_kwarg(kwargs, "d")

        for _ in range(self.hps.g_steps):
            with timed("interacting"):
                if self.hps.trainer_worker:
                    self._sync_weights(all_=True)
                # Unpack (one rollout dict for policy training, one for reward training)
                p_rollout, d_rollout = roll_gen.__next__()
                d["roll_len"].append(mpi_mean_reduce(p_rollout["ep_lens"]))
                roll_env_ret = mpi_mean_reduce(p_rollout["ep_env_rets"])
                d["roll_env_ret"].append(roll_env_ret)

                p_rollout = postprocess_rollout(
                    p_rollout,
                    self.hps.gamma,
                    self.hps.gae_lambda,
                    rew_key="syn_rews",
                )

                if self.hps.trainer_worker:
                    rollouts = sync_data(
                        {"p_rollout": p_rollout, "d_rollout": d_rollout},
                    )

            with timed("policy and value training"):
                if self.hps.master and self.hps.trainer_worker:
                    p_rollout = concatenate_rollouts(rollouts, key="p_rollout")
                    d_rollout = concatenate_rollouts(rollouts, key="d_rollout")

                if self.hps.master:
                    self.to(self._device)
                    metrics, lrnow = self.update_policy_value(
                        p_rollout=p_rollout,
                        iters_so_far=iters_so_far,
                    )
                    self.to("cpu")

                    d["lrnow"].append(lrnow)
                    d.update(metrics)

        for _ in range(self.hps.d_steps):
            with timed("discriminator training"):
                if self.hps.master:
                    metrics = self.update_discriminator(
                        d_rollout=d_rollout,
                        iters_so_far=iters_so_far,
                    )
                    d["discriminator/losses"].append(metrics["d_loss"])
                    d["discriminator/grad_pen"].append(metrics["grad_pen"])
                    d["discriminator/entropy"].append(metrics["entropy_loss"])
                    d["discriminator/p_e_loss"].append(metrics["p_e_loss"])
                    d["discriminator/p_loss"].append(metrics["p_loss"])
                    d["discriminator/e_loss"].append(metrics["e_loss"])

        self.log_model_stats()

    def update_timesteps(self, timesteps_so_far):
        return timesteps_so_far + (self.hps.g_steps * self.hps.rollout_len)
