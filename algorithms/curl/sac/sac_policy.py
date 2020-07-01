from gym.spaces import Discrete, Box
import logging

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.sac.sac_tf_policy import build_sac_model, postprocess_trajectory
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta)
from ray.rllib.utils import try_import_torch

# customs 
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.torch_policy import TorchPolicy

from algorithms.curl.sac.sac_model import CurlSACTorchModel, CURL
from utils.utils import update_model_params

torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)

#######################################################################################################
#####################################   Setup   #####################################################
#######################################################################################################

""" modified from sac_tf_policy.py
"""
def build_curl_sac_model(policy, obs_space, action_space, config):
    if config["model"].get("custom_model"):
        logger.warning(
            "Setting use_state_preprocessor=True since a custom model "
            "was specified.")
        config["use_state_preprocessor"] = True
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for SAC.".format(action_space))
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space has multiple dimensions "
            "{}. ".format(action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")

    # # infer num_outpus as action space dim (not embedding size!!)
    # _, num_outputs = ModelCatalog.get_action_dist(
    #     action_space, config["model"], framework="torch")
    num_outputs = action_space.n

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = CurlSACTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        # customs 
        embed_dim=config["embed_dim"],
        encoder_type=config["encoder_type"],
        num_layers=config["num_layers"],
        num_filters=config["num_filters"],
        cropped_image_size=config["cropped_image_size"]) 

    # target 
    policy.target_model = CurlSACTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        # customs 
        embed_dim=config["embed_dim"],
        encoder_type=config["encoder_type"],
        num_layers=config["num_layers"],
        num_filters=config["num_filters"],
        cropped_image_size=config["cropped_image_size"]) 

    # NOTE: customs 
    policy.curl = CURL(
        policy.model.embed_dim,
        policy.model.encoder,
        policy.target_model.encoder 
    )

    return policy.model


def build_curl_sac_model_and_action_dist(policy, obs_space, action_space, config):
    model = build_curl_sac_model(policy, obs_space, action_space, config)
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


def curl_action_distribution_fn(policy,
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
    # crop input 
    cropped_obs_batch = model.center_crop(
        obs_batch.permute(0,3,1,2).float()
    )
    # get action distrib
    model_out, _ = model.get_embeddings({
        "obs": cropped_obs_batch,
        "is_training": is_training,
    }, [], None, permute=False)
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)

    return distribution_inputs, action_dist_class, []


#######################################################################################################
#####################################   Loss   #####################################################
#######################################################################################################

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
        "cpc_loss": torch.mean(policy.cpc_loss),
    }


def optimizer_fn(policy, config):
    """Creates all necessary optimizers for SAC learning.
    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.
    """
    # actor optimizer 
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        betas=(config["optimization"]["actor_beta"], 0.999)
    )
    
    # critic optimizer 
    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
            betas=(config["optimization"]["critic_beta"], 0.999)
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            ))

    # entropy / alpha optimizer 
    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        betas=(config["optimization"]["alpha_beta"], 0.999)
    )

    # NOTE: customs 
    # cpc / encoder optimizer 
    policy.encoder_optim = torch.optim.Adam(
        params=policy.model.encoder.parameters(),
        lr=config["optimization"]["encoder_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )
    policy.cpc_optim = torch.optim.Adam(
        params=policy.curl.parameters(),
        lr=config["optimization"]["encoder_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    return tuple([policy.actor_optim] + policy.critic_optims +
                 [policy.alpha_optim] + [policy.encoder_optim, policy.cpc_optim])


#######################################################################################################
#####################################   Mixins   #####################################################
#######################################################################################################

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
            # NOTE: use eval mode cropping (center), 64 -> cropped_image_size (54)
            cur_obs = input_dict[SampleBatch.CUR_OBS].permute(0,3,1,2).float()
            nxt_obs = input_dict[SampleBatch.NEXT_OBS].permute(0,3,1,2).float()

            input_dict[SampleBatch.CUR_OBS] = self.model.center_crop(cur_obs)
            input_dict[SampleBatch.NEXT_OBS] = self.model.center_crop(nxt_obs)

            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            # build_q_losses(self, self.model, None, input_dict)
            self.critic_loss(input_dict)

            # Self.td_error is set within actor_critic_loss call.
            return self.td_error

        self.compute_td_error = comspute_td_error


class TargetNetworkMixin:
    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    # def update_target(self, tau=None):
    #     """ for here, 
    #     """
    #     # Update_target_fn will be called periodically to copy Q network to
    #     # target Q network, using (soft) tau-synching.
    #     tau = tau or self.config.get("tau")
    #     model_state_dict = self.model.state_dict()
    #     # Support partial (soft) synching.
    #     # If tau == 1.0: Full sync from Q-model to target Q-model.
    #     if tau != 1.0:
    #         target_state_dict = self.target_model.state_dict()
    #         model_state_dict = {
    #             k: tau * model_state_dict[k] + (1 - tau) * v
    #             for k, v in target_state_dict.items()
    #         }
    #     self.target_model.load_state_dict(model_state_dict)

    def update_target(self, tau=None):
        """ different update rates for different model components
        """
        # hard update for initialization
        if tau is not None:
            critic_tau = encoder_tau = tau 
        else:
            critic_tau = self.config["critic_tau"]
            encoder_tau = self.config["encoder_tau"]
        # update Q targets
        update_model_params(self.model.q_net, self.target_model.q_net, critic_tau)
        if self.config["twin_q"]:
            update_model_params(self.model.twin_q_net, self.target_model.twin_q_net, critic_tau)
        # update encoder 
        update_model_params(self.model.encoder, self.target_model.encoder, encoder_tau)



# NOTE: customs 
class CurlMixin:
    """ methods (overrides) for Contrastive Unsupervised Representations for RL
    """
    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.POS_OBS = "pos"

        self.actor_update_freq = self.config.get("actor_update_freq", 2) 
        self.cpc_update_freq = self.config.get("cpc_update_freq", 1)

    @property
    def alpha(self):
        """ learnable temperature parameter """
        return self.model.log_alpha.exp()

    def critic_loss(self, train_batch):
        """ critic loss based on bellman error 
        """
        # get embeddings (already permuted at cropping)
        model_out_t, _ = self.model.get_embeddings({
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": True
        }, [], None, permute=False)

        model_out_tp1, _ = self.model.get_embeddings({
            "obs": train_batch[SampleBatch.NEXT_OBS],
            "is_training": True
        }, [], None, permute=False)

        target_model_out_tp1, _ = self.target_model.get_embeddings({
            "obs": train_batch[SampleBatch.NEXT_OBS],
            "is_training": True
        }, [], None, permute=False)

        # Get all action probs directly from pi and form their logp.
        # action & logprob for cur_obs 
        log_pis_t = F.log_softmax(self.model.get_policy_output(model_out_t), dim=-1)
        policy_t = torch.exp(log_pis_t)
        # action & logprob for nxt_obs 
        log_pis_tp1 = F.log_softmax(self.model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = torch.exp(log_pis_tp1)

        # Q-values for cur_obs (discrete)
        q_t = self.model.get_q_values(model_out_t)       
        # Target Q-values for nxt_obs (discrete)
        q_tp1 = self.target_model.get_q_values(target_model_out_tp1)
        # double Q values for cur_obs (discrete)
        if self.config["twin_q"]:
            twin_q_t = self.model.get_twin_q_values(model_out_t)
            twin_q_tp1 = self.target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        # final Q-values with entropy  
        # q_tp1 -= self.alpha * log_pis_tp1
        q_tp1 -= self.alpha.detach() * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS], num_classes=q_t.size()[-1])
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        # actually selected double Q-values 
        if self.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = (
            train_batch[SampleBatch.REWARDS] +
            (self.config["gamma"]**self.config["n_step"]) * q_tp1_best_masked
        ).detach()

        # Compute the TD-error (potentially clipped).
        base_td_error = torch.abs(q_t_selected - q_t_selected_target)
        if self.config["twin_q"]:
            twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
            td_error = 0.5 * (base_td_error + twin_td_error)
        else:
            td_error = base_td_error

        critic_loss = [
            0.5 * torch.mean(torch.pow(q_t_selected_target - q_t_selected, 2.0))
        ]
        if self.config["twin_q"]:
            critic_loss.append(0.5 * torch.mean(
                torch.pow(q_t_selected_target - twin_q_t_selected, 2.0)))

        # save for status function 
        self.q_t = q_t
        self.policy_t = policy_t
        self.log_pis_t = log_pis_t
        self.td_error = td_error
        self.critic_loss = critic_loss
        return critic_loss

    def actor_and_alpha_loss(self, train_batch):
        """ actor loss based on KL for energy-based policy 
        """
        # get embeddings 
        embed_out = self.model.get_embeddings({
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": True
        }, [], None, permute=False)
        # detach encoder, so we don't update it with the actor loss
        model_out_t = embed_out[0].detach()

        # action & logprob for cur_obs 
        log_pis_t = F.log_softmax(self.model.get_policy_output(model_out_t), dim=-1)
        policy_t = torch.exp(log_pis_t)

        # Q-values for cur_obs (discrete)
        q_t = self.model.get_q_values(model_out_t) 
        if self.config["twin_q"]:
            twin_q_t = self.model.get_twin_q_values(model_out_t)
            q_t = torch.min(q_t, twin_q_t)
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS], num_classes=q_t.size()[-1])
        q_t = torch.sum(q_t * one_hot, dim=-1)

        # Actor loss.
        actor_loss = torch.mean(torch.sum(
            torch.mul(
                # NOTE: No stop_grad around policy output here
                # (compare with q_t_det_policy for continuous case).
                policy_t,
                self.alpha.detach() * log_pis_t - q_t.detach()
            ), dim=-1
        ))

        # Discrete case: Multiply the action probs as weights with the original loss terms (no expectations needed).
        # Note: In the papers, alpha is used directly, here we take the log
        weighted_log_alpha_loss = policy_t.detach() * (
            -self.model.log_alpha * (log_pis_t + self.model.target_entropy).detach())
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))

        # Save for stats function.
        policy.actor_loss = actor_loss
        policy.alpha_loss = alpha_loss
        policy.log_alpha_value = self.model.log_alpha
        policy.alpha_value = self.alpha
        policy.target_entropy = self.model.target_entropy
        return actor_loss, alpha_loss

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

        train_batch[SampleBatch.CUR_OBS] = self.model.random_crop(cur_obs)
        train_batch[SampleBatch.NEXT_OBS] = self.model.random_crop(nxt_obs)
        # constrastive positives 
        train_batch[self.POS_OBS] = self.model.random_crop(cur_obs)

        # collect backprop stats
        grad_info = {"allreduce_latency": 0.0}
        step = self.global_timestep                                                                  

        # NOTE: loop through each loss and optimizer
        # optimize critics
        critic_loss = self.critic_loss(train_batch)
        if self.config["twin_q"]:
            opt1, opt2 = self.critic_optims
            opt1.zero_grad()
            opt2.zero_grad()

            critic_loss[0].backward()
            grad_info.update(self.extra_grad_process(opt1, critic_loss[0]))
            latency = self.average_grad(opt1)
            grad_info["allreduce_latency"] += latency

            critic_loss[1].backward()
            grad_info.update(self.extra_grad_process(opt2, critic_loss[1]))
            latency = self.average_grad(opt2)
            grad_info["allreduce_latency"] += latency

            opt1.step()
            opt2.step()
        else:
            opt = self.critic_optims[0]
            opt.zero_grad()
            critic_loss[0].backward()
            grad_info.update(self.extra_grad_process(opt, critic_loss[0]))
            latency = self.average_grad(opt)
            grad_info["allreduce_latency"] += latency
            opt.step()

        # optimize actor and alpha 
        if step % self.actor_update_freq == 0:
            actor_loss, alpha_loss = self.actor_and_alpha_loss(train_batch)

            # optimize actor 
            self.actor_optim.zero_grad()
            actor_loss.backward()
            grad_info.update(self.extra_grad_process(self.actor_optim, actor_loss))
            latency = self.average_grad(self.actor_optim)
            grad_info["allreduce_latency"] += latency
            self.actor_optim.step()

            # optimize alpha 
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            grad_info.update(self.extra_grad_process(self.alpha_optim, alpha_loss))
            latency = self.average_grad(self.alpha_optim)
            grad_info["allreduce_latency"] += latency
            self.alpha_optim.step()
        
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


def setup_late_mixins(policy, obs_space, action_space, config):
    policy.target_model = policy.target_model.to(policy.device)
    policy.model.log_alpha = policy.model.log_alpha.to(policy.device)
    policy.model.target_entropy = policy.model.target_entropy.to(policy.device)
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)
    CurlMixin.__init__(policy)



#######################################################################################################
#####################################   Policy   #####################################################
#######################################################################################################

# hack to avoid cycle imports 
import algorithms.curl.sac.sac_trainer

CurlSACTorchPolicy = build_torch_policy(
    name="CurlSACTorchPolicy",
    # loss updates shifted to policy.learn_on_batch
    loss_fn=None,
    # get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.curl.sac.sac_trainer.SAC_CONFIG,
    stats_fn=stats,
    # called in a torch.no_grad scope, calls loss func again to update td error 
    postprocess_fn=postprocess_trajectory,
    # will clip grad in learn_on_batch if grad_clip is not None in config 
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    after_init=setup_late_mixins,
    make_model_and_action_dist=build_curl_sac_model_and_action_dist,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin, CurlMixin],
    action_distribution_fn=curl_action_distribution_fn,
)

