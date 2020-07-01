from gym.spaces import Discrete, Box
import logging

import ray
# from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta)
from ray.rllib.utils import try_import_torch

# custom imports 
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
from algorithms.baselines.sac.sac_model import BaselineSACTorchModel


torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)

#######################################################################################################
#####################################   Model  #####################################################
#######################################################################################################

def build_sac_model_and_action_dist(policy, obs_space, action_space, config):
    """ shared between NoAug & Drq SAC torch policy 
    """
    # make model 
    model = build_sac_model(policy, obs_space, action_space, config)
    # make action output distrib
    action_dist_class = get_dist_class(config, action_space)
    return model, action_dist_class


""" modified from `build_sac_model` from 
https://github.com/ray-project/ray/blob/master/rllib/agents/sac/sac_tf_policy.py
"""
def build_sac_model(policy, obs_space, action_space, config):
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
    policy.model = BaselineSACTorchModel(
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
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"]) 


    policy.target_model = BaselineSACTorchModel(
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
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"])

    return policy.model


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


#######################################################################################################
#####################################   Policy   #####################################################
#######################################################################################################

# hack to avoid cycle imports 
import algorithms.baselines.sac.sac_trainer

BaselineSACTorchPolicy = SACTorchPolicy.with_updates(
    name="BaselineSACTorchPolicy",
    make_model_and_action_dist=build_sac_model_and_action_dist,
    action_distribution_fn=action_distribution_fn,
    # get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.baselines.sac.sac_trainer.SAC_CONFIG
)
