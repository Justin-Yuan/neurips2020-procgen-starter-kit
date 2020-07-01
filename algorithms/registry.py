"""
Registry of custom implemented algorithms names

Please refer to the following examples to add your custom algorithms : 

- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent

An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""


def _import_custom_random_agent():
    from .custom_random_agent.custom_random_agent import CustomRandomAgent
    return CustomRandomAgent

#####################################   Baselines  #####################################################

def _import_baseline_dqn_agent():
    from .baselines.dqn.dqn_trainer import BaselineDQNTrainer
    return BaselineDQNTrainer

def _import_baseline_rainbow_agent():
    from .baselines.rainbow.rainbow_trainer import BaselineRainbowTrainer
    return BaselineRainbowTrainer

def _import_baseline_ppo_agent():
    from .baselines.ppo.ppo_trainer import BaselinePPOTrainer
    return BaselinePPOTrainer

def _import_baseline_sac_agent():
    from .baselines.sac.sac_trainer import BaselineSACTrainer
    return BaselineSACTrainer


#####################################   DrQ  #####################################################
 
def _import_drq_sac_agent():
    from .drq.sac.sac_trainer import DrqSACTrainer
    return DrqSACTrainer

def _import_drq_ppo_agent():
    from .drq.ppo.ppo_trainer import DrqPPOTrainer
    return DrqPPOTrainer

def _import_drq_dqn_agent():
    from .drq.dqn.dqn_trainer import DrqDQNTrainer
    return DrqDQNTrainer

def _import_drq_rainbow_agent():
    from .drq.rainbow.rainbow_trainer import DrqRainbowTrainer
    return DrqRainbowTrainer


#####################################   SAC-AE  #####################################################

def _import_sac_ae_agent():
    from .sac_ae.sac_ae_trainer import SACAETrainer
    return SACAETrainer


#####################################   CURL  #####################################################

def _import_curl_sac_agent():
    from .curl.sac.sac_trainer import CurlSACTrainer
    return CurlSACTrainer

def _import_curl_rainbow_agent():
    from .curl.rainbow.rainbow_trainer import CurlRainbowTrainer
    return CurlRainbowTrainer


#####################################   Dreamer  #####################################################
 
# def _import_dreamer_agent():
#     from .dreamer_agent.dreamer_trainer import DreamerTrainer
#     return DreamerTrainer



#######################################################################################################
#####################################   Registry  #####################################################
#######################################################################################################

CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent": _import_custom_random_agent,

    # Baselines 
    "custom/BaselineSACTrainer": _import_baseline_sac_agent,
    "custom/BaselinePPOTrainer": _import_baseline_ppo_agent,
    "custom/BaselineDQNTrainer": _import_baseline_dqn_agent,
    "custom/BaselineRainbowTrainer": _import_baseline_rainbow_agent,

    # DrQ
    "custom/DrqSACTrainer": _import_drq_sac_agent,
    "custom/DrqPPOTrainer": _import_drq_ppo_agent,
    "custom/DrqDQNTrainer": _import_drq_dqn_agent,
    "custom/DrqRainbowTrainer": _import_drq_rainbow_agent,

    # SAC-AE
    "custom/SACAETrainer": _import_sac_ae_agent,

    # CURL
    "custom/CurlSACTrainer": _import_curl_sac_agent,
    "custom/CurlRainbowTrainer": _import_curl_rainbow_agent,

    # Dreamer
    # "custom/DreamerTrainer": _import_dreamer_agent,
}
