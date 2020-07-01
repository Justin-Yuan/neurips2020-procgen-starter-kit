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
from ray.rllib.agents.dqn.dqn_torch_policy import QLoss, build_q_losses, compute_q_values
from algorithms.drq.dqn.dqn_model import DrqDQNTorchModel

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

    policy.q_model = DrqDQNTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_SCOPE,
        name="dqn_model",
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        # use_noisy=config["noisy"],    # not implemented 
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        # customs 
        embed_dim=config["embed_dim"],
        encoder_type=config["encoder_type"],
        augmentation=config["augmentation"],
        aug_num=config["aug_num"],
        max_shift=config["max_shift"]
    )

    policy.q_func_vars = policy.q_model.variables()

    policy.target_q_model = DrqDQNTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_TARGET_SCOPE,
        name="target_dqn_model",
        dueling=config["dueling"],
        q_hiddens=config["hiddens"],
        # use_noisy=config["noisy"],    # not implemented  
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        # customs 
        embed_dim=config["embed_dim"],
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
    q_vals = compute_q_values(policy, model, obs_batch, explore, is_training)
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    policy.q_values = q_vals
    return policy.q_values, TorchCategorical, []  # state-out


#######################################################################################################
#####################################   DQN Loss funcs   #####################################################
#######################################################################################################

def build_drq_q_losses(policy, model, _, train_batch):
    """ use input augmentation on Q target and Q updates
    """
    config = policy.config
    aug_num = config["aug_num"]
    
    # target q network evalution
    q_tp1_best_avg = 0
    orig_nxt_obs = train_batch[SampleBatch.NEXT_OBS].clone()
    for _ in range(aug_num):
        # augment obs 
        aug_nxt_obs = model.trans(
            orig_nxt_obs.permute(0,3,1,2).float()
        ).permute(0,2,3,1)

        q_tp1 = compute_q_values(
            policy,
            policy.target_q_model,
            aug_nxt_obs,
            explore=False,
            is_training=True)

        # compute estimate of best possible value starting from state at t + 1
        if config["double_q"]:
            q_tp1_using_online_net = compute_q_values(
                policy,
                policy.q_model,
                aug_nxt_obs,
                explore=False,
                is_training=True)
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = F.one_hot(q_tp1_best_using_online_net,
                                                    policy.action_space.n)
            q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        else:
            q_tp1_best_one_hot_selection = F.one_hot(
                torch.argmax(q_tp1, 1), policy.action_space.n)
            q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)

        # accumulate target Q with augmented next obs 
        q_tp1_best_avg += q_tp1_best
    q_tp1_best_avg /= aug_num

    # q network evaluation
    aug_loss = 0
    orig_cur_obs = train_batch[SampleBatch.CUR_OBS].clone()
    for _ in range(aug_num):
        # augment obs 
        aug_cur_obs = model.trans(
            orig_cur_obs.permute(0,3,1,2).float()
        ).permute(0,2,3,1)

        q_t = compute_q_values(
            policy,
            policy.q_model,
            aug_cur_obs,
            explore=False,
            is_training=True)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                    policy.action_space.n)
        q_t_selected = torch.sum(q_t * one_hot_selection, 1)

        # Bellman error 
        policy.q_loss = QLoss(q_t_selected, q_tp1_best_avg, 
                            train_batch[PRIO_WEIGHTS],
                            train_batch[SampleBatch.REWARDS],
                            train_batch[SampleBatch.DONES].float(),
                            config["gamma"], config["n_step"],
                            config["num_atoms"], config["v_min"],
                            config["v_max"])
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
import algorithms.drq.dqn.dqn_trainer

NoAugDQNTorchPolicy = build_torch_policy(
    name="NoAugDQNTorchPolicy",
    loss_fn=build_q_losses,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    # shared 
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.drq.dqn.dqn_trainer.DQN_CONFIG,
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

DrqDQNTorchPolicy = build_torch_policy(
    name="DrqDQNTorchPolicy",
    loss_fn=build_drq_q_losses,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    # shared 
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.drq.dqn.dqn_trainer.DQN_CONFIG,
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

