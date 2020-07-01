from gym.spaces import Discrete

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
# from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
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
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.torch_policy import TorchPolicy
from algorithms.curl.rainbow.rainbow_model import CurlRainbowTorchModel
from algorithms.curl.sac.sac_model import CURL
from utils.utils import update_params, update_model_params


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

    policy.q_model = CurlRainbowTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_SCOPE,
        name="rainbow_model",
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
        embed_dim=config["embed_dim"],
        encoder_type=config["encoder_type"],
        num_layers=config["num_layers"],
        num_filters=config["num_filters"],
        cropped_image_size=config["cropped_image_size"]
    )

    # policy.q_func_vars = policy.q_model.variables()
    # only for Q net params (excluding encoder params)
    policy.q_func_vars = policy.q_model.q_variables()

    policy.target_q_model = CurlRainbowTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        # name=Q_TARGET_SCOPE,
        name="target_rainbow_model",
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
        embed_dim=config["embed_dim"],
        encoder_type=config["encoder_type"],
        num_layers=config["num_layers"],
        num_filters=config["num_filters"],
        cropped_image_size=config["cropped_image_size"]
    )

    # policy.target_q_func_vars = policy.target_q_model.variables()
    policy.target_q_func_vars = policy.target_q_model.q_variables()

    # NOTE: customs 
    policy.curl = CURL(
        policy.q_model.embed_dim,
        policy.q_model.encoder,
        policy.target_q_model.encoder 
    )

    return policy.q_model, TorchCategorical


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      is_training=False,
                                      **kwargs):
    # crop input 
    cropped_obs_batch = model.center_crop(
        obs_batch.permute(0,3,1,2).float()
    )
    q_vals = compute_rainbow_q_values(policy, model, cropped_obs_batch, explore, is_training)
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
    }, [], None, permute=False)     # already permuted from cropping

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


# def adam_optimizer(policy, config):
#     return torch.optim.Adam(
#         policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"])

# NOTE: customs 
def optimizer_fn(policy, config):
    """Creates all necessary optimizers for SAC learning.
    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.
    """
    # critic optimizer 
    policy.critic_optim = torch.optim.Adam(
            params=policy.q_model.parameters(),
            # lr=config["optimization"]["critic_learning_rate"],
            lr=policy.cur_lr,
            # eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
            eps=config["adam_epsilon"],
            betas=(config["critic_beta"], 0.999)
        )

    # cpc / encoder optimizer 
    policy.encoder_optim = torch.optim.Adam(
        params=policy.q_model.encoder.parameters(),
        lr=config["encoder_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )
    policy.cpc_optim = torch.optim.Adam(
        params=policy.curl.parameters(),
        lr=config["encoder_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    return tuple([policy.critic_optim] + [policy.encoder_optim, policy.cpc_optim])


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

            # NOTE: use eval mode cropping (center), 64 -> cropped_image_size (54)
            cur_obs = input_dict[SampleBatch.CUR_OBS].permute(0,3,1,2).float()
            nxt_obs = input_dict[SampleBatch.NEXT_OBS].permute(0,3,1,2).float()

            input_dict[SampleBatch.CUR_OBS] = self.q_model.center_crop(cur_obs)
            input_dict[SampleBatch.NEXT_OBS] = self.q_model.center_crop(nxt_obs)

            # Do forward pass on loss to update td error attribute
            # build_q_losses(self, self.model, None, input_dict)
            build_rainbow_q_losses(self, self.q_model, None, input_dict)

            return self.q_loss.td_error

        self.compute_td_error = compute_td_error


class TargetNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        # def do_update():
        #     # Update_target_fn will be called periodically to copy Q network to
        #     # target Q network.
        #     assert len(self.q_func_vars) == len(self.target_q_func_vars), \
        #         (self.q_func_vars, self.target_q_func_vars)
        #     self.target_q_model.load_state_dict(self.q_model.state_dict())

        # self.update_target = do_update

        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        """ different update rates for different model components
         + soft updates
        """
        # hard update for initialization
        if tau is not None:
            critic_tau = encoder_tau = tau 
        else:
            critic_tau = self.config["critic_tau"]
            encoder_tau = self.config["encoder_tau"]
        # update Q targets
        update_params(self.q_func_vars, self.target_q_func_vars, critic_tau)
        # update encoder 
        # update_params(
        #     self.q_model.encoder.parameters(), 
        #     self.target_model.encoder.parameters(), 
        #     encoder_tau
        # )
        update_model_params(
            self.q_model.encoder, 
            self.target_q_model.encoder, 
            encoder_tau
        )



# NOTE: customs 
class CurlMixin:
    """ methods (overrides) for Contrastive Unsupervised Representations for RL
    """
    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.POS_OBS = "pos"
        self.cpc_update_freq = self.config.get("cpc_update_freq", 1)
    
    def cpc_loss(self, train_batch):
        """ meat for CURL 
        """
        # collect positive and negative samples 
        obs_anchor = train_batch[SampleBatch.CUR_OBS]
        obs_pos = train_batch[self.POS_OBS]

        z_a = self.curl.encode(obs_anchor, permute=False)
        z_pos = self.curl.encode(obs_pos, ema=True, permute=False)
        
        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        # Save for stats function.
        policy.cpc_loss = loss 
        return loss 

    @override(TorchPolicy)
    def learn_on_batch(self, postprocessed_batch):
        """ CURL-style updates 
        """
        # convert to UsageTrackingDict from MultiagentBatch
        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        # crop inputs 64 -> cropped_image_size (54)
        cur_obs = train_batch[SampleBatch.CUR_OBS].permute(0,3,1,2).float()
        nxt_obs = train_batch[SampleBatch.NEXT_OBS].permute(0,3,1,2).float()

        train_batch[SampleBatch.CUR_OBS] = self.q_model.random_crop(cur_obs)
        train_batch[SampleBatch.NEXT_OBS] = self.q_model.random_crop(nxt_obs)
        # constrastive positives 
        train_batch[self.POS_OBS] = self.q_model.random_crop(cur_obs)

        # collect backprop stats
        grad_info = {"allreduce_latency": 0.0}
        step = self.global_timestep                                                                  

        # NOTE: loop through each loss and optimizer
        # optimize critics
        critic_loss = build_rainbow_q_losses(self, self.q_model, _, train_batch)

        self.critic_optim.zero_grad()
        critic_loss.backward()

        grad_info.update(self.extra_grad_process(self.critic_optim, critic_loss))
        latency = self.average_grad(self.critic_optim)
        grad_info["allreduce_latency"] += latency

        self.critic_optim.step()

        # optimize encoder and CURL 
        if step % self.cpc_update_freq == 0:
            cpc_loss = self.cpc_loss(train_batch, grad_info)
            
            self.encoder_optim.zero_grad()
            self.cpc_optim.zero_grad()
            cpc_loss.backward()

            grad_info.update(self.extra_grad_process(self.encoder_optim, cpc_loss))
            latency = self.average_grad(self.encoder_optim)
            grad_info["allreduce_latency"] += latency

            grad_info.update(self.extra_grad_process(self.cpc_optim, cpc_loss))
            latency = self.average_grad(self.cpc_optim)
            grad_info["allreduce_latency"] += latency

            self.encoder_optim.step()
            self.cpc_optim.step()

        # collect other grad info
        grad_info["allreduce_latency"] /= len(self._optimizers)
        grad_info.update(self.extra_grad_info(train_batch))
        return {LEARNER_STATS_KEY: grad_info}

    def average_grad(self, opt):
        """ for distributed setting, average gradients withh allreduce 
        Input:
            - opt: optimizer 
        Output:
            - latency
        """
        if self.distributed_world_size:
            grads = []
            for param_group in opt.param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        grads.append(p.grad)

            start = time.time()
            if torch.cuda.is_available():
                # Sadly, allreduce_coalesced does not work with CUDA yet.
                for g in grads:
                    torch.distributed.all_reduce(
                        g, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce_coalesced(
                    grads, op=torch.distributed.ReduceOp.SUM)

            for param_group in opt.param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        p.grad /= self.distributed_world_size

            latency = time.time() - start
            return latency
        else:
            return 0


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def after_init(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    # Move target net to device (this is done autoatically for the
    # policy.model, but not for any other models the policy has).
    policy.target_q_model = policy.target_q_model.to(policy.device)
    CurlMixin.__init__(policy)


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
import algorithms.curl.rainbow.rainbow_trainer

CurlRainbowTorchPolicy = build_torch_policy(
    name="CurlRainbowTorchPolicy",
    # loss updates shifted to policy.learn_on_batch
    loss_fn=None,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    optimizer_fn=optimizer_fn,
    # shared 
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.curl.rainbow.rainbow_trainer.RAINBOW_CONFIG,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    # added curl mixin 
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
        CurlMixin
    ])