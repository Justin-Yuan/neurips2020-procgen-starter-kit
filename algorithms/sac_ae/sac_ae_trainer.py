from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.agents.trainer_template import build_trainer
# custom imports
from algorithms.sac_ae_agent.sac_ae_policy import SACAETorchPolicy, DrqSACAETorchPolicy


#######################################################################################################
#####################################   Config Template  #####################################################
#######################################################################################################

OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha",
    "prioritized_replay_beta", "prioritized_replay_eps",
    "rollout_fragment_length", "train_batch_size", "learning_starts"
]

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Model ===
    "twin_q": True,
    "use_state_preprocessor": False,
    # RLlib model options for the Q function(s).
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 64],
        "hidden_activation": DEPRECATED_VALUE,
        "hidden_layer_sizes": DEPRECATED_VALUE,
    },
    # RLlib model options for the policy function.
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 64],
        "hidden_activation": DEPRECATED_VALUE,
        "hidden_layer_sizes": DEPRECATED_VALUE,
    },
    # Unsquash actions to the upper and lower bounds of env's action space.
    # Ignored for discrete action spaces.
    "normalize_actions": True,

    # === Customs ===
    "augmentation": True,
    "aug_num": 2,
    "max_shift": 4,  

    # === Learning ===
    # Disable setting done=True at end of episode. This should be set to True
    # for infinite-horizon MDPs (e.g., many continuous control problems).
    "no_done_at_end": False,
    # Update the target by \tau * policy + (1-\tau) * target_policy.
    "tau": 5e-3,
    # Initial value to use for the entropy weight alpha.
    "initial_alpha": 1.0,
    # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
    # Discrete(2), -3.0 for Box(shape=(3,))).
    # This is the inverse of reward scale, and will be optimized automatically.
    "target_entropy": "auto",
    # N-step target updates.
    "n_step": 1,

    # Number of env steps to optimize for before returning.
    "timesteps_per_iteration": 100,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "prioritized_replay_beta_annealing_timesteps": 20000,
    "final_prioritized_replay_beta": 0.4,
    # ae configs 
    "decoder_update_freq": 1, 
    "decoder_latent_lambda": 0.0,
    "decoder_weight_lambda": 0.0,
    # Whether to LZ4 compress observations
    "compress_observations": False,
    # If set, this will fix the ratio of sampled to replayed timesteps.
    # Otherwise, replay will proceed at the native ratio determined by
    # (train_batch_size / rollout_fragment_length).
    "training_intensity": None,

    # === Optimization ===
    "optimization": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
        "encoder_learning_rate": 0.001,
        "decoder_learning_rate": 0.001,
       
    },
    # If not None, clip gradients during optimization at this value.
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 256,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,

    # === Parallelism ===
    # Whether to use a GPU for local optimization.
    "num_gpus": 0,
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span.
    "min_iter_time_s": 1,

    # Whether the loss should be calculated deterministically (w/o the
    # stochastic action sampling step). True only useful for cont. actions and
    # for debugging!
    "_deterministic_loss": False,
    # Use a Beta-distribution instead of a SquashedGaussian for bounded,
    # continuous action spaces (not recommended, for debugging only).
    "_use_beta_distribution": False,

    # DEPRECATED VALUES (set to -1 to indicate they have not been overwritten
    # by user's config). If we don't set them here, we will get an error
    # from the config-key checker.
    "grad_norm_clipping": DEPRECATED_VALUE,
})
# __sphinx_doc_end__
# yapf: enable


#######################################################################################################
#####################################   Helper funcs  #####################################################
#######################################################################################################

def get_policy_class(config):
    ################################################################################################
    # if config["framework"] == "torch":
    #     from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
    #     return SACTorchPolicy
    # else:
    #     return SACTFPolicy

    # NOTE: switch between policy with/without input augmentations
    if config["augmentation"] == True:
        return DrqSACAETorchPolicy
    else:
        return SACAETorchPolicy
    ################################################################################################


def validate_config(config):
    if config.get("grad_norm_clipping", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning("grad_norm_clipping", "grad_clip")
        config["grad_clip"] = config.pop("grad_norm_clipping")

    # Use same keys as for standard Trainer "model" config.
    for model in ["Q_model", "policy_model"]:
        if config[model].get("hidden_activation", DEPRECATED_VALUE) != \
                DEPRECATED_VALUE:
            deprecation_warning(
                "{}.hidden_activation".format(model),
                "{}.fcnet_activation".format(model),
                error=True)
        if config[model].get("hidden_layer_sizes", DEPRECATED_VALUE) != \
                DEPRECATED_VALUE:
            deprecation_warning(
                "{}.hidden_layer_sizes".format(model),
                "{}.fcnet_hiddens".format(model),
                error=True)


#######################################################################################################
#####################################   Trainer  #####################################################
#######################################################################################################


# SACAETrainer = GenericOffPolicyTrainer.with_updates(
SACAETrainer = build_trainer(
    name="SACAE",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=SACAETorchPolicy,
    get_policy_class=get_policy_class,
)
