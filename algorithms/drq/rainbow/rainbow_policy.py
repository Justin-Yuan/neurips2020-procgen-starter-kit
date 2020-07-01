from gym.spaces import Discrete

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.torch_ops import huber_loss, reduce_mean_ignore_inf
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional

# customs 
from algorithms.drq.rainbow.rainbow_model import DrqRainbowTorchModel


#######################################################################################################
#####################################   Models   #####################################################
#######################################################################################################

def build_q_model_and_distribution(policy, obs_space, action_space, config):

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    # # NOTE: using manually specified embedding layer, no need to switch num_outputs
    # if config["hiddens"]:
    #     # try to infer the last layer size, otherwise fall back to 256
    #     num_outputs = ([256] + config["model"]["fcnet_hiddens"])[-1]
    #     config["model"]["no_final_linear"] = True
    # else:
    #     num_outputs = action_space.n
    num_outputs = action_space.n    # NOTE: actually no used!!! 

    # TODO(sven): Move option to add LayerNorm after each Dense
    #  generically into ModelCatalog.
    add_layer_norm = (
        isinstance(getattr(policy, "exploration", None), ParameterNoise)
        or config["exploration_config"]["type"] == "ParameterNoise")

    policy.q_model = DrqRainbowTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_SCOPE,
        name="dqn_model",
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        use_noisy=config["noisy"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        num_atoms=config["num_atoms"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        # customs
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"],
        augmentation=config["augmentation"],
        aug_num=config["aug_num"],
        max_shift=config["max_shift"]
    )

    policy.q_func_vars = policy.q_model.variables()

    policy.target_q_model = DrqRainbowTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_TARGET_SCOPE,
        name="target_dqn_model",
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        use_noisy=config["noisy"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        num_atoms=config["num_atoms"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        # customs
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"],
        augmentation=config["augmentation"],
        aug_num=config["aug_num"],
        max_shift=config["max_shift"]
    )

    policy.target_q_func_vars = policy.target_q_model.variables()

    return policy.q_model, TorchCategorical


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      is_training=False,
                                      **kwargs):
    q_vals = compute_rainbow_q_values(policy, model, obs_batch, explore, is_training)
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    policy.q_values = q_vals
    return policy.q_values, TorchCategorical, []  # state-out


#######################################################################################################
#####################################   Loss funcs   #####################################################
#######################################################################################################

def compute_rainbow_q_values(policy, model, obs, explore, is_training=False):
    """ supports normal DQN and distributional DQN now 
    reference: https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_tf_policy.py
    """
    # # NOTE: LAZY DEV
    # if policy.config["num_atoms"] > 1:
    #     raise ValueError("torch DQN does not support distributional DQN yet!")
    config = policy.config

    # model_out, state = model({
    #     SampleBatch.CUR_OBS: obs,
    #     "is_training": is_training,
    # }, [], None)
    # NOTE: set noise with training flag 
    if is_training:
        model.train()
    else:
        model.eval()
    model_out, state = model.get_embeddings({
        SampleBatch.CUR_OBS: obs,
        "is_training": is_training,
    }, [], None)

    out = model.get_advantages_or_q_values(model_out)
    if config["num_atoms"] > 1:
        (advantages_or_q_values, z, support_logits_per_action, logits, dist) = out
    else:
        (advantages_or_q_values, logits, dist) = out

    if policy.config["dueling"]:
        state_value = model.get_state_value(model_out)
        
        if config["num_atoms"] > 1:
            support_logits_per_action_mean = torch.mean(support_logits_per_action, 1)

            support_logits_per_action_centered = support_logits_per_action - torch.unsqueeze(
                support_logits_per_action_mean, 1)

            support_logits_per_action = torch.unsqueeze(
                state_value, 1) + support_logits_per_action_centered

            support_prob_per_action = F.softmax(support_logits_per_action)

            q_values = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            dist = support_prob_per_action
        else:
            advantages_mean = reduce_mean_ignore_inf(advantages_or_q_values, 1)
            advantages_centered = advantages_or_q_values - torch.unsqueeze(
                advantages_mean, 1)
            q_values = state_value + advantages_centered
    else:
        q_values = advantages_or_q_values

    # return q_values
    return q_values, logits, dist 



class RainbowQLoss:
    """ accomodates both normal bellman error and distributional Q loss 
    reference: https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_tf_policy.py
            https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_tf_policy.py
    """
    def __init__(self,
                 q_t_selected,
                 q_logits_t_selected,   # for distributional 
                 q_tp1_best,
                 q_dist_tp1_best,   # for distributional
                 importance_weights,
                 rewards,
                 done_mask,
                 gamma=0.99,
                 n_step=1,
                 num_atoms=1,
                 v_min=-10.0,
                 v_max=10.0):

        if num_atoms > 1:
            # # NOTE: LAZY DEV
            # raise ValueError("Torch version of DQN does not support "
            #                  "distributional Q yet!")

            # Distributional Q-learning which corresponds to an entropy loss
            z = torch.arange(num_atoms).float().to(q_dist_tp1_best.device)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

            # (batch_size, 1) * (1, num_atoms) = (batch_size, num_atoms)
            r_tau = torch.unsqueeze(rewards, -1) + gamma**n_step * torch.unsqueeze(
                    1.0 - done_mask, -1) * torch.unsqueeze(z, 0)
            r_tau = torch.clamp(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = torch.floor(b)
            ub = torch.ceil(b)
             # indispensable judgement which is missed in most implementations
            # when b happens to be an integer, lb == ub, so pr_j(s', a*) will
            # be discarded because (ub-b) == (b-lb) == 0
            floor_equal_ceil = torch.le(ub - lb, 0.5).float()

            # (batch_size, num_atoms, num_atoms)
            l_project = F.one_hot(lb.long(), num_atoms)  
            # (batch_size, num_atoms, num_atoms)
            u_project = F.one_hot(ub.long(), num_atoms)  
            ml_delta = q_dist_tp1_best * (ub - b + floor_equal_ceil)
            mu_delta = q_dist_tp1_best * (b - lb)
            ml_delta = torch.sum(
                l_project * torch.unsqueeze(ml_delta, -1), dim=1)
            mu_delta = torch.sum(
                u_project * torch.unsqueeze(mu_delta, -1), dim=1)
            m = ml_delta + mu_delta

            # Rainbow paper claims that using this cross entropy loss for
            # priority is robust and insensitive to `prioritized_replay_alpha`

            # self.td_error = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=m, logits=q_logits_t_selected)
            # pytorch equivalent to tf.nn.softmax_cross_entropy_with_logits
            # https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
            self.td_error = -m * F.log_softmax(q_logits_t_selected, -1)
            self.loss = torch.mean(
                self.td_error * importance_weights.float())
            self.stats = {
                # TODO: better Q stats for dist dqn
                "mean_td_error": tf.reduce_mean(self.td_error),
            }
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked

            # compute the error (potentially clipped)
            self.td_error = q_t_selected - q_t_selected_target.detach()
            self.loss = torch.mean(
                importance_weights.float() * huber_loss(self.td_error))
            self.stats = {
                "mean_q": torch.mean(q_t_selected),
                "min_q": torch.min(q_t_selected),
                "max_q": torch.max(q_t_selected),
                "td_error": self.td_error,
                "mean_td_error": torch.mean(self.td_error),
            }



def build_rainbow_q_losses(policy, model, _, train_batch):
    """ full rainbow losses (with distributional but no DrQ)
    reference: https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_torch_policy.py
            https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_tf_policy.py
    """
    config = policy.config
    # q network evaluation
    q_t, q_logits_t, q_dist_t = compute_rainbow_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # target q network evalution
    q_tp1, q_logits_tp1, q_dist_tp1 = compute_rainbow_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                  policy.action_space.n)
    q_t_selected = torch.sum(q_t * one_hot_selection, 1)
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
    )

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        q_tp1_using_online_net = compute_rainbow_q_values(
            policy,
            policy.q_model,
            train_batch[SampleBatch.NEXT_OBS],
            explore=False,
            is_training=True)
        q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = F.one_hot(q_tp1_best_using_online_net,
                                                 policy.action_space.n)
        q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = torch.sum(
            q_dist_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )
    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = torch.sum(
            q_dist_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )

    policy.q_loss = RainbowQLoss(
        q_t_selected, 
        q_logits_t_selected,
        q_tp1_best, 
        q_dist_tp1_best,
        train_batch[PRIO_WEIGHTS],
        train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.DONES].float(),
        config["gamma"], 
        config["n_step"],
        config["num_atoms"], 
        config["v_min"],
        config["v_max"]
    )
    return policy.q_loss.loss



def build_drq_rainbow_q_losses(policy, model, _, train_batch):
    """ use input augmentation on Q target and Q updates
    """
    config = policy.config
    aug_num = config["aug_num"]
    
    # target q network evalution
    q_tp1_best_avg = None
    q_dist_tp1_best_avg = None 
    orig_nxt_obs = train_batch[SampleBatch.NEXT_OBS].clone()

    for _ in range(aug_num):
        # augment obs 
        aug_nxt_obs = model.trans(
            orig_nxt_obs.permute(0,3,1,2).float()
        ).permute(0,2,3,1)

        q_tp1, q_logits_tp1, q_dist_tp1 = compute_rainbow_q_values(
            policy,
            policy.target_q_model,
            aug_nxt_obs,
            explore=False,
            is_training=True)

        # compute estimate of best possible value starting from state at t + 1
        if config["double_q"]:
            q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net = compute_rainbow_q_values(
                policy,
                policy.q_model,
                aug_nxt_obs,
                explore=False,
                is_training=True)
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = F.one_hot(q_tp1_best_using_online_net,
                                                    policy.action_space.n)
            q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = torch.sum(
                q_dist_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )
        else:
            q_tp1_best_one_hot_selection = F.one_hot(
                torch.argmax(q_tp1, 1), policy.action_space.n)
            q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = torch.sum(
                q_dist_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )

        # accumulate target Q with augmented next obs 
        if q_tp1_best_avg is None:
            q_tp1_best_avg = q_tp1_best
            q_dist_tp1_best_avg = q_dist_tp1_best
        else:
            q_tp1_best_avg += q_tp1_best
            q_dist_tp1_best_avg += q_dist_tp1_best

    q_tp1_best_avg /= aug_num
    q_dist_tp1_best_avg /= aug_num

    # q network evaluation
    aug_loss = 0
    orig_cur_obs = train_batch[SampleBatch.CUR_OBS].clone()
    for _ in range(aug_num):
        # augment obs 
        aug_cur_obs = model.trans(
            orig_cur_obs.permute(0,3,1,2).float()
        ).permute(0,2,3,1)

        q_t, q_logits_t, q_dist_t = compute_rainbow_q_values(
            policy,
            policy.q_model,
            aug_cur_obs,
            explore=False,
            is_training=True)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                    policy.action_space.n)
        q_t_selected = torch.sum(q_t * one_hot_selection, 1)
        q_logits_t_selected = torch.sum(
            q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
        )

        # Bellman error 
        policy.q_loss = RainbowQLoss(
            q_t_selected, 
            q_logits_t_selected,
            q_tp1_best_avg, 
            q_dist_tp1_best_avg,
            train_batch[PRIO_WEIGHTS],
            train_batch[SampleBatch.REWARDS],
            train_batch[SampleBatch.DONES].float(),
            config["gamma"], 
            config["n_step"],
            config["num_atoms"], 
            config["v_min"],
            config["v_max"]
        )
        # accumulate loss with augmented obs 
        aug_loss += policy.q_loss.loss
    return aug_loss / aug_num 



def adam_optimizer(policy, config):
    return torch.optim.Adam(
        policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"])


def build_q_stats(policy, batch):
    return dict({
        "cur_lr": policy.cur_lr,
    }, **policy.q_loss.stats)


#######################################################################################################
#####################################   Mixins   #####################################################
#######################################################################################################

class ComputeTDErrorMixin:
    def __init__(self):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.DONES] = done_mask
            input_dict[PRIO_WEIGHTS] = importance_weights

            # Do forward pass on loss to update td error attribute
            # build_q_losses(self, self.model, None, input_dict)
            # NOTE: customs but not sure if works 
            self._loss(self, self.model, None, input_dict)

            return self.q_loss.td_error

        self.compute_td_error = compute_td_error


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def after_init(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    # Move target net to device (this is done autoatically for the
    # policy.model, but not for any other models the policy has).
    policy.target_q_model = policy.target_q_model.to(policy.device)


def grad_process_and_td_error_fn(policy, optimizer, loss):
    # Clip grads if configured.
    info = apply_grad_clipping(policy, optimizer, loss)
    # Add td-error to info dict.
    info["td_error"] = policy.q_loss.td_error
    return info


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    return {"q_values": policy.q_values}


#######################################################################################################
#####################################   Policy   #####################################################
#######################################################################################################

# hack to avoid cycle imports 
import algorithms.drq.rainbow.rainbow_trainer

NoAugRainbowTorchPolicy = build_torch_policy(
    name="NoAugRainbowTorchPolicy",
    loss_fn=build_rainbow_q_losses,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    # shared 
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.drq.rainbow.rainbow_trainer.RAINBOW_CONFIG,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])

DrqRainbowTorchPolicy = build_torch_policy(
    name="DrqRainbowTorchPolicy",
    loss_fn=build_drq_rainbow_q_losses,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    # shared 
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.drq.rainbow.rainbow_trainer.RAINBOW_CONFIG,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])