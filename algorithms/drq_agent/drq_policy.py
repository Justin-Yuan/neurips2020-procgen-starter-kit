from gym.spaces import Discrete
import logging

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.sac.sac_tf_policy import build_sac_model, \
    postprocess_trajectory
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta)
from ray.rllib.utils import try_import_torch
import kornia

torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)


def build_sac_model_and_action_dist(policy, obs_space, action_space, config):
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = get_dist_class(config, action_space)
    return model, action_dist_class


def get_dist_class(config, action_space):
    if isinstance(action_space, Discrete):
        return TorchCategorical
    else:
        if config["normalize_actions"]:
            return TorchSquashedGaussian if \
                not config["_use_beta_distribution"] else TorchBeta
        else:
            return TorchDiagGaussian


def action_distribution_fn(policy,
                           model,
                           obs_batch,
                           *,
                           state_batches=None,
                           seq_lens=None,
                           prev_action_batch=None,
                           prev_reward_batch=None,
                           explore=None,
                           timestep=None,
                           is_training=None):
    model_out, _ = model({
        "obs": obs_batch,
        "is_training": is_training,
    }, [], None)
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)

    return distribution_inputs, action_dist_class, []


def actor_critic_loss(policy, model, _, train_batch):
    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    ################################################################################################
    # model_out_t, _ = model({
    #     "obs": train_batch[SampleBatch.CUR_OBS],
    #     "is_training": True,
    # }, [], None)

    # model_out_tp1, _ = model({
    #     "obs": train_batch[SampleBatch.NEXT_OBS],
    #     "is_training": True,
    # }, [], None)

    # target_model_out_tp1, _ = policy.target_model({
    #     "obs": train_batch[SampleBatch.NEXT_OBS],
    #     "is_training": True,
    # }, [], None)

    # NOTE: augmentation 
    aug_num = policy['aug_num']
    model_out_t_augs, model_out_tp1_augs, target_model_out_tp1_augs = [], [], []
    for _ in range(aug_num):
        # augmented obs 
        augCurSampleBatch = policy.trans(train_batch[SampleBatch.CUR_OBS])
        augNextSampleBatch = policy.trans(train_batch[SampleBatch.NEXT_OBS])

        # cur obs embeddings
        model_out_t, _ = model({
            "obs": augCurSampleBatch,
            "is_training": True,
        }, [], None)
        model_out_t_augs.append(model_out_t)

        # next obs embeddings 
        model_out_tp1, _ = model({
            "obs": augNextSampleBatch,
            "is_training": True,
        }, [], None)
        model_out_tp1_augs.append(model_out_tp1)

        # target next obs embeddings 
        target_model_out_tp1, _ = policy.target_model({
            "obs": augNextSampleBatch,
            "is_training": True,
        }, [], None)
        target_model_out_tp1_augs.append(target_model_out_tp1)
    ################################################################################################

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        ################################################################################################
        # # Get all action probs directly from pi and form their logp.
        # log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
        # policy_t = torch.exp(log_pis_t)
        # log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
        # policy_tp1 = torch.exp(log_pis_tp1)
        # # Q-values.
        # q_t = model.get_q_values(model_out_t)
        # # Target Q-values.
        # q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        # if policy.config["twin_q"]:
        #     twin_q_t = model.get_twin_q_values(model_out_t)
        #     twin_q_tp1 = policy.target_model.get_twin_q_values(
        #         target_model_out_tp1)
        #     q_tp1 = torch.min(q_tp1, twin_q_tp1)
        # q_tp1 -= alpha * log_pis_tp1

        # # Actually selected Q-values (from the actions batch).
        # one_hot = F.one_hot(
        #     train_batch[SampleBatch.ACTIONS], num_classes=q_t.size()[-1])
        # q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        # if policy.config["twin_q"]:
        #     twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # # Discrete case: "Best" means weighted by the policy (prob) outputs.
        # q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        # q_tp1_best_masked = \
        #     (1.0 - train_batch[SampleBatch.DONES].float()) * \
        #     q_tp1_best

        # NOTE: Q values with augmented obs embeddings 
        q_t_selected_augs, twin_q_t_selected_augs = [], []
        q_tp1_best_masked_augs = []
        # for actor loss 
        log_pis_t_aug, policy_t_aug, q_t_aug= None, None, None 

        # repeat SAC Q value estimations for each augmented obs 
        for i in range(len(model_out_t_augs)):
            model_out_t = model_out_t_augs[i]
            model_out_tp1 = model_out_tp1_augs[i]
            target_model_out_tp1 = target_model_out_tp1_augs[i]

            log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
            policy_t = torch.exp(log_pis_t)
            log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
            policy_tp1 = torch.exp(log_pis_tp1)
            # Q-values.
            q_t = model.get_q_values(model_out_t)
            # Target Q-values.
            q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
            if policy.config["twin_q"]:
                twin_q_t = model.get_twin_q_values(model_out_t)
                twin_q_tp1 = policy.target_model.get_twin_q_values(
                    target_model_out_tp1)
                q_tp1 = torch.min(q_tp1, twin_q_tp1)
            q_tp1 -= alpha * log_pis_tp1

            # Actually selected Q-values (from the actions batch).
            one_hot = F.one_hot(
                train_batch[SampleBatch.ACTIONS], num_classes=q_t.size()[-1])
            q_t_selected = torch.sum(q_t * one_hot, dim=-1)
            if policy.config["twin_q"]:
                twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
            # Discrete case: "Best" means weighted by the policy (prob) outputs.
            q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
            q_tp1_best_masked = \
                (1.0 - train_batch[SampleBatch.DONES].float()) * \
                q_tp1_best

            # push to containers 
            q_t_selected_augs.append(q_t_selected)
            if policy.config["twin_q"]:
                twin_q_t_selected_augs.append(twin_q_t_selected)
            q_tp1_best_masked_augs.append(q_tp1_best_masked)

            # only use first augmented obs for actor loss updates
            if log_pis_t_aug is None:
                log_pis_t_aug = log_pis_t
                policy_t_aug = policy_t
                q_t_aug = q_t

        # get augmentation averaged target Q
        q_tp1_best_masked_avg = sum(q_tp1_best_masked_augs) / len(q_tp1_best_masked_augs)
        ################################################################################################
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * \
            q_tp1_best

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    ################################################################################################
    # # compute RHS of bellman equation
    # q_t_selected_target = (
    #     train_batch[SampleBatch.REWARDS] +
    #     (policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked
    # ).detach()

    # NOTE: use averaged Q target 
    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS] +
        (policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked_avg
    ).detach()
    ################################################################################################

    ################################################################################################
    # # Compute the TD-error (potentially clipped).
    # base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    # if policy.config["twin_q"]:
    #     twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
    #     td_error = 0.5 * (base_td_error + twin_td_error)
    # else:
    #     td_error = base_td_error

    # critic_loss = [
    #     0.5 * torch.mean(torch.pow(q_t_selected_target - q_t_selected, 2.0))
    # ]
    # if policy.config["twin_q"]:
    #     critic_loss.append(0.5 * torch.mean(
    #         torch.pow(q_t_selected_target - twin_q_t_selected, 2.0)))

    # NOTE: apply critic loss for each augmented obs 
    td_error = 0.0
    critic_loss = [0.0, 0.0] if policy.config["twin_q"] else [0.0]

    for i in range(len(q_t_selected_augs)):
        q_t_selected = q_t_selected_augs[i]

        # Compute the TD-error (potentially clipped).
        base_td_error = torch.abs(q_t_selected - q_t_selected_target)
        if policy.config["twin_q"]:
            twin_q_t_selected = twin_q_t_selected_augs[i]
            twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
            td_error += 0.5 * (base_td_error + twin_td_error)
        else:
            td_error += base_td_error

        critic_loss[0] += 0.5 * torch.mean(
            torch.pow(q_t_selected_target - q_t_selected, 2.0))
        
        if policy.config["twin_q"]:
            twin_q_t_selected = twin_q_t_selected_augs[i]
            critic_loss[1] += 0.5 * torch.mean(
                torch.pow(q_t_selected_target - twin_q_t_selected, 2.0)))

    # normalized critic loss across augmented obs 
    td_error /= len(q_t_selected_augs)
    for j in range(len(critic_loss)):
        critic_loss[j] /= len(q_t_selected_augs)
    ################################################################################################

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        ################################################################################################
        # weighted_log_alpha_loss = policy_t.detach() * (
        #     -model.log_alpha * (log_pis_t + model.target_entropy).detach())
        # # Sum up weighted terms and mean over all batch items.
        # alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # # Actor loss.
        # actor_loss = torch.mean(
        #     torch.sum(
        #         torch.mul(
        #             # NOTE: No stop_grad around policy output here
        #             # (compare with q_t_det_policy for continuous case).
        #             policy_t,
        #             alpha.detach() * log_pis_t - q_t.detach()),
        #         dim=-1))

        # NOTE: actor and alpha loss with augmented obs (only 1 of them used)
        weighted_log_alpha_loss = policy_t_aug.detach() * (
            -model.log_alpha * (log_pis_t_aug + model.target_entropy).detach())
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t_aug,
                    alpha.detach() * log_pis_t_aug - q_t_aug.detach()),
                dim=-1))
        ################################################################################################
    else:
        alpha_loss = -torch.mean(model.log_alpha *
                                 (log_pis_t + model.target_entropy).detach())
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Save for stats function.
    policy.q_t = q_t
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = alpha
    policy.target_entropy = model.target_entropy

    # Return all loss terms corresponding to our optimizers.
    return tuple([policy.actor_loss] + policy.critic_loss +
                 [policy.alpha_loss])


def stats(policy, train_batch):
    return {
        "td_error": policy.td_error,
        "mean_td_error": torch.mean(policy.td_error),
        "actor_loss": torch.mean(policy.actor_loss),
        "critic_loss": torch.mean(torch.stack(policy.critic_loss)),
        "alpha_loss": torch.mean(policy.alpha_loss),
        "alpha_value": torch.mean(policy.alpha_value),
        "log_alpha_value": torch.mean(policy.log_alpha_value),
        "target_entropy": policy.target_entropy,
        "policy_t": torch.mean(policy.policy_t),
        "mean_q": torch.mean(policy.q_t),
        "max_q": torch.max(policy.q_t),
        "min_q": torch.min(policy.q_t),
    }


def optimizer_fn(policy, config):
    """Creates all necessary optimizers for SAC learning.
    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            ))
    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    return tuple([policy.actor_optim] + policy.critic_optims +
                 [policy.alpha_optim])


class ComputeTDErrorMixin:
    def __init__(self):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_t,
                SampleBatch.ACTIONS: act_t,
                SampleBatch.REWARDS: rew_t,
                SampleBatch.NEXT_OBS: obs_tp1,
                SampleBatch.DONES: done_mask,
                PRIO_WEIGHTS: importance_weights,
            })
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            actor_critic_loss(self, self.model, None, input_dict)

            # Self.td_error is set within actor_critic_loss call.
            return self.td_error

        self.compute_td_error = compute_td_error


class TargetNetworkMixin:
    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")
        model_state_dict = self.model.state_dict()
        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        if tau != 1.0:
            target_state_dict = self.target_model.state_dict()
            model_state_dict = {
                k: tau * model_state_dict[k] + (1 - tau) * v
                for k, v in target_state_dict.items()
            }
        self.target_model.load_state_dict(model_state_dict)


def setup_late_mixins(policy, obs_space, action_space, config):
    policy.target_model = policy.target_model.to(policy.device)
    policy.model.log_alpha = policy.model.log_alpha.to(policy.device)
    policy.model.target_entropy = policy.model.target_entropy.to(policy.device)
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)


SACTorchPolicy = build_torch_policy(
    name="SACTorchPolicy",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    after_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)

################################################################################################
# NOTE: subclass SAC Policy to insert augmenteations

class DrqSACTorchPolicy(SACTorchPolicy):
    def __init__(self, obs_space, action_space, config):
        super.__init__(obs_space, action_space, config)

        image_pad = config["max_shift"]
        obs_shape = obs_space.shape[-1]
        self.trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape, obs_shape))
            
################################################################################################