import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE

# custom imports  
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from algorithms.drq.rainbow.rainbow_policy import NoAugRainbowTorchPolicy, DrqRainbowTorchPolicy


logger = logging.getLogger(__name__)


#######################################################################################################
#####################################   Config Template   #####################################################
#######################################################################################################

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": True,
    # Dense-layer setup for each the advantage branch and the value branch
    # in a dueling architecture.
    "hiddens": [256],
    # Whether to use double dqn
    "double_q": True,
    # N-step Q learning
    "n_step": 1,

    # === Exploration Settings (Experimental) ===
    "exploration_config": {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

        # For soft_q, use:
        # "exploration_config" = {
        #   "type": "SoftQ"
        #   "temperature": [float, e.g. 1.0]
        # }
    },
    # Switch to greedy actions in evaluation workers.
    "evaluation_config": {
        "explore": False,
    },

    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of iterations.
    "timesteps_per_iteration": 1000,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 500,
    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 50000,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Final value of beta (by default, we use constant beta=0.4).
    "final_prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    "prioritized_replay_beta_annealing_timesteps": 20000,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": False,
    # Callback to run before learning on a multi-agent batch of experiences.
    "before_learn_on_batch": None,
    # If set, this will fix the ratio of sampled to replayed timesteps.
    # Otherwise, replay will proceed at the native ratio determined by
    # (train_batch_size / rollout_fragment_length).
    "training_intensity": None,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 5e-4,
    # Learning rate schedule
    "lr_schedule": None,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    # If not None, clip gradients during optimization at this value
    "grad_clip": 40,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1000,
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 4,
    # Size of a batch sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 32,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,

    # DEPRECATED VALUES (set to -1 to indicate they have not been overwritten
    # by user's config). If we don't set them here, we will get an error
    # from the config-key checker.
    "schedule_max_timesteps": DEPRECATED_VALUE,
    "exploration_final_eps": DEPRECATED_VALUE,
    "exploration_fraction": DEPRECATED_VALUE,
    "beta_annealing_fraction": DEPRECATED_VALUE,
    "per_worker_exploration": DEPRECATED_VALUE,
    "softmax_temp": DEPRECATED_VALUE,
    "soft_q": DEPRECATED_VALUE,
    "parameter_noise": DEPRECATED_VALUE,
    "grad_norm_clipping": DEPRECATED_VALUE,
})
# __sphinx_doc_end__
# yapf: enable


#######################################################################################################
#####################################   Policy   #####################################################
#######################################################################################################

def validate_config(config):
    """Checks and updates the config based on settings.
    Rewrites rollout_fragment_length to take into account n_step truncation.
    """
    # TODO(sven): Remove at some point.
    #  Backward compatibility of epsilon-exploration config AND beta-annealing
    # fraction settings (both based on schedule_max_timesteps, which is
    # deprecated).
    if config.get("grad_norm_clipping", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning("grad_norm_clipping", "grad_clip")
        config["grad_clip"] = config.pop("grad_norm_clipping")

    schedule_max_timesteps = None
    if config.get("schedule_max_timesteps", DEPRECATED_VALUE) != \
            DEPRECATED_VALUE:
        deprecation_warning(
            "schedule_max_timesteps",
            "exploration_config.epsilon_timesteps AND "
            "prioritized_replay_beta_annealing_timesteps")
        schedule_max_timesteps = config["schedule_max_timesteps"]
    if config.get("exploration_final_eps", DEPRECATED_VALUE) != \
            DEPRECATED_VALUE:
        deprecation_warning("exploration_final_eps",
                            "exploration_config.final_epsilon")
        if isinstance(config["exploration_config"], dict):
            config["exploration_config"]["final_epsilon"] = \
                config.pop("exploration_final_eps")
    if config.get("exploration_fraction", DEPRECATED_VALUE) != \
            DEPRECATED_VALUE:
        assert schedule_max_timesteps is not None
        deprecation_warning("exploration_fraction",
                            "exploration_config.epsilon_timesteps")
        if isinstance(config["exploration_config"], dict):
            config["exploration_config"]["epsilon_timesteps"] = config.pop(
                "exploration_fraction") * schedule_max_timesteps
    if config.get("beta_annealing_fraction", DEPRECATED_VALUE) != \
            DEPRECATED_VALUE:
        assert schedule_max_timesteps is not None
        deprecation_warning(
            "beta_annealing_fraction (decimal)",
            "prioritized_replay_beta_annealing_timesteps (int)")
        config["prioritized_replay_beta_annealing_timesteps"] = config.pop(
            "beta_annealing_fraction") * schedule_max_timesteps
    if config.get("per_worker_exploration", DEPRECATED_VALUE) != \
            DEPRECATED_VALUE:
        deprecation_warning("per_worker_exploration",
                            "exploration_config.type=PerWorkerEpsilonGreedy")
        if isinstance(config["exploration_config"], dict):
            config["exploration_config"]["type"] = PerWorkerEpsilonGreedy
    if config.get("softmax_temp", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning(
            "soft_q", "exploration_config={"
            "type=StochasticSampling, temperature=[float]"
            "}")
        if config.get("softmax_temp", 1.0) < 0.00001:
            logger.warning("softmax temp very low: Clipped it to 0.00001.")
            config["softmax_temperature"] = 0.00001
    if config.get("soft_q", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning(
            "soft_q", "exploration_config={"
            "type=SoftQ, temperature=[float]"
            "}")
        config["exploration_config"] = {
            "type": "SoftQ",
            "temperature": config.get("softmax_temp", 1.0)
        }
    if config.get("parameter_noise", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning("parameter_noise", "exploration_config={"
                            "type=ParameterNoise"
                            "}")

    if config["exploration_config"]["type"] == "ParameterNoise":
        if config["batch_mode"] != "complete_episodes":
            logger.warning(
                "ParameterNoise Exploration requires `batch_mode` to be "
                "'complete_episodes'. Setting batch_mode=complete_episodes.")
            config["batch_mode"] = "complete_episodes"
        if config.get("noisy", False):
            raise ValueError(
                "ParameterNoise Exploration and `noisy` network cannot be "
                "used at the same time!")

    # Update effective batch size to include n-step
    adjusted_batch_size = max(config["rollout_fragment_length"],
                              config.get("n_step", 1))
    config["rollout_fragment_length"] = adjusted_batch_size

    if config.get("prioritized_replay"):
        if config["multiagent"]["replay_mode"] == "lockstep":
            raise ValueError("Prioritized replay is not supported when "
                             "replay_mode=lockstep.")
        elif config["replay_sequence_length"] > 1:
            raise ValueError("Prioritized replay is not supported when "
                             "replay_sequence_length > 1.")


def get_rainbow_policy_class(config):    
    if config["augmentation"] == True:
        return DrqRainbowTorchPolicy
    else:
        return NoAugRainbowTorchPolicy



#######################################################################################################
#####################################   Trainers   #####################################################
#######################################################################################################

"""
# original rainbow (https://arxiv.org/pdf/1710.02298.pdf)
num_atoms: 51   #(min max +-10)
noisy, double_q, dueling: True
n_steps: 3
learning_starts: 1000

# ray rllib rainbow (https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/tests/test_dqn.py)
num_atoms: 10 
noisy, double_q, dueling: True
n_steps: 5
learning_starts: 1000

# efficient rainbow (https://arxiv.org/pdf/1906.05243.pdf)
num_atoms: 51
noisy, double_q, dueling: True
n_step: 20
learning_starts: 1600
"""

# here we use efficient rainbow with partial params from rllib rainbow
new_config = {

    "num_atoms": 51,
    "noisy": True,
    "double_q": True,
    "dueling": True,
    "n_step": 20,
    "learning_starts": 1600,
    
    # "critic_learning_rate": 1e-3,
    "lr": 1e-3,
    "critic_beta": 0.9,
    "encoder_learning_rate": 1e-3, 
    # "adam_epsilon": 1e-7,

    "cpc_update_freq": 1,
    "target_network_update_freq": 2,

    "critic_tau": 0.01,
    "encoder_tau": 0.05,

    "train_batch_size": 32,
    "gamma": 0.99,

    # customs 
    "embed_dim": 128,
    "encoder_type": "impala",

    "augmentation": True,
    "aug_num": 2,
    "max_shift": 4,
}
RAINBOW_CONFIG = DEFAULT_CONFIG.copy()
RAINBOW_CONFIG.update(new_config)



DrqRainbowTrainer = GenericOffPolicyTrainer.with_updates(
    name="DrqRainbow",
    default_config=RAINBOW_CONFIG,
    validate_config=validate_config,
    default_policy=DrqRainbowTorchPolicy,
    get_policy_class=get_rainbow_policy_class,
)




