from gym.spaces import Discrete

import ray
# from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
# from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional

# customs 
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from algorithms.baselines.dqn.dqn_model import BaselineDQNTorchModel


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

    policy.q_model = BaselineDQNTorchModel(
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
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"]
    )

    policy.q_func_vars = policy.q_model.variables()

    policy.target_q_model = BaselineDQNTorchModel(
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
        embed_dim =config["embed_dim"],
        encoder_type=config["encoder_type"]
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
#####################################   Policy   #####################################################
#######################################################################################################

# hack to avoid cycle imports 
import algorithms.baselines.dqn.dqn_trainer

BaselineDQNTorchPolicy = DQNTorchPolicy.with_updates(
    name="BaselineDQNTorchPolicy",
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    # get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    get_default_config=lambda: algorithms.baselines.dqn.dqn_trainer.DQN_CONFIG
)
