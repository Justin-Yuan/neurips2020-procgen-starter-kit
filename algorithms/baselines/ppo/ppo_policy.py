from gym.spaces import Discrete, Box
import logging

import numpy as np
import ray
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta)    
# from ray.rllib.policy.torch_policy_template import build_torch_policy
# from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

# custom imports 
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from algorithms.baselines.ppo.ppo_model import BaselinePPOTorchModel


logger = logging.getLogger(__name__)


#######################################################################################################
#####################################   Model   #####################################################
#######################################################################################################

def build_ppo_model_and_action_dist(policy, obs_space, action_space, config):
    """ shared between NoAug & Drq PPO torch policy 
    """
    # make model
    # _, num_outputs = ModelCatalog.get_action_dist(
    #     action_space, config["model"], framework="torch"
    # ) 
    num_outputs = action_space.n

    policy.model = BaselinePPOTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name="ppo_model",
        # customs 
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"]
    ) 
    # make action output distrib
    action_dist_class = get_dist_class(config, action_space)
    return policy.model, action_dist_class


def get_dist_class(config, action_space):
    if isinstance(action_space, Discrete):
        return TorchCategorical
    else:
        if config["normalize_actions"]:
            return TorchSquashedGaussian if \
                not config["_use_beta_distribution"] else TorchBeta
        else:
            return TorchDiagGaussian


#######################################################################################################
#####################################   Policy   #####################################################
#######################################################################################################

# hack to avoid cycle imports 
import algorithms.baselines.ppo.ppo_trainer

BaselinePPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="BaselinePPOTorchPolicy",
    make_model_and_action_dist=build_ppo_model_and_action_dist,
    # get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.baselines.ppo.ppo_trainer.PPO_CONFIG
)

