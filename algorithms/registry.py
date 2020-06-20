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

def _import_drq_sac_agent():
    from .drq_agent.sac_trainer import DrqSACTrainer
    return DrqSACTrainer

def _import_drq_ppo_agent():
    from .drq_agent.ppo_trainer import DrqPPOTrainer
    return DrqPPOTrainer

def _import_drq_dqn_agent():
    from .drq_agent.rainbow_trainer import DrqDQNTrainer
    return DrqDQNTrainer

def _import_drq_rainbow_agent():
    from .drq_agent.rainbow_trainer import DrqRainbowTrainer
    return DrqRainbowTrainer

def _import_sac_ae_agent():
    from .sac_ae_agent.sac_ae_trainer import SACAETrainer
# def _import_curl_agent():
#     from .curl_agent.curl_trainer import CurlTrainer
#     return CurlTrainer

# def _import_dreamer_agent():
#     from .dreamer_agent.dreamer_trainer import DreamerTrainer
#     return DreamerTrainer



CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent": _import_custom_random_agent,
    "custom/DrqSACTrainer": _import_drq_sac_agent,
    "custom/DrqPPOTrainer": _import_drq_ppo_agent,
    "custom/DrqDQNTrainer": _import_drq_dqn_agent,
    "custom/DrqRainbowTrainer": _import_drq_rainbow_agent,
    "custom/SACAETrainer": _import_sac_ae_agent,
    # "custom/CurlTrainer": _import_curl_agent,
    # "custom/DreamerTrainer": _import_dreamer_agent,
}
