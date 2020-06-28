import numpy as np

from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional

from kornia.augmentation import RandomCrop
from models import make_encoder
from utils.utils import NoisyLinear



#######################################################################################################
#####################################   Main models   #####################################################
#######################################################################################################

class DrqRainbowTorchModel(TorchModelV2, nn.Module):
    """ Extension of standard TorchModelV2 to provide dueling-Q functionality.
    reference: https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/distributional_q_tf_model.py
    """

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            *,
            dueling=False,
            q_hiddens=(256, ),
            dueling_activation="relu",
            use_noisy=False,
            sigma0=0.5,
            # TODO(sven): Move `add_layer_norm` into ModelCatalog as
            #  generic option, then error if we use ParameterNoise as
            #  Exploration type and do not have any LayerNorm layers in
            #  the net.
            add_layer_norm=False,
            num_atoms=1,
            v_min=-10.0,
            v_max=10.0,
            # customs
            embed_dim = 256,
            encoder_type="impala",
            augmentation=False,
            aug_num=2,
            max_shift=4,
            **kwargs):
        """Initialize variables of this model.
        Extra model kwargs:
            dueling (bool): Whether to build the advantage(A)/value(V) heads
                for DDQN. If True, Q-values are calculated as:
                Q = (A - mean[A]) + V. If False, raw NN output is interpreted
                as Q-values.
            q_hiddens (List[int]): List of layer-sizes after(!) the
                Advantages(A)/Value(V)-split. Hence, each of the A- and V-
                branches will have this structure of Dense layers. To define
                the NN before this A/V-split, use - as always -
                config["model"]["fcnet_hiddens"].
            dueling_activation (str): The activation to use for all dueling
                layers (A- and V-branch). One of "relu", "tanh", "linear".
            use_noisy (bool): use noisy nets
            sigma0 (float): initial value of noisy nets
            add_layer_norm (bool): Enable layer norm (for param noise).
        """
        nn.Module.__init__(self)
        super(DrqRainbowTorchModel, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        # NOTE: customs 
        self.embed_dim = embed_dim
        h, w, c = obs_space.shape
        shape = (c, h, w)
        # obs embedding 
        self.encoder = make_encoder(encoder_type, shape, out_features=embed_dim)

        # NOTE: value output branches
        self.dueling = dueling
        # ins = num_outputs
        ins = embed_dim

        # Dueling case: Build the shared (advantages and value) fc-network.
        advantage_module = nn.Sequential()
        value_module = None
        # if to use noisy net 
        layer_cls = NoisyLinear if use_noisy else nn.Linear 

        if self.dueling:
            value_module = nn.Sequential()
            for i, n in enumerate(q_hiddens):

                # MLP layers 
                advantage_module.add_module(
                    "dueling_A_{}".format(i), layer_cls(ins, n))
                value_module.add_module(
                    "dueling_V_{}".format(i), layer_cls(ins, n))

                # Add activations if necessary.
                if dueling_activation == "relu":
                    advantage_module.add_module(
                        "dueling_A_act_{}".format(i), nn.ReLU())
                    value_module.add_module(
                        "dueling_V_act_{}".format(i), nn.ReLU())
                elif dueling_activation == "tanh":
                    advantage_module.add_module(
                        "dueling_A_act_{}".format(i), nn.Tanh())
                    value_module.add_module(
                        "dueling_V_act_{}".format(i), nn.Tanh())

                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module(
                        "LayerNorm_A_{}".format(i), nn.LayerNorm(n))
                    value_module.add_module(
                        "LayerNorm_V_{}".format(i), nn.LayerNorm(n))
                ins = n

            # Actual Advantages layer (nodes=num-actions) and 
            # value layer (nodes=1).
            advantage_module.add_module(
                "A", layer_cls(ins, action_space.n * num_atoms))
            value_module.add_module(
                "V", layer_cls(ins, num_atoms))

        # Non-dueling:
        # Q-value layer (use main module's outputs as Q-values).
        else:
            # pass
            # NOTE: manually adding q value (no dueling) branch following embedding
            for i, n in enumerate(q_hiddens):
                advantage_module.add_module(
                    "Q_{}".format(i), layer_cls(ins, n))
                if dueling_activation == "relu":
                    advantage_module.add_module(
                        "Q_act_{}".format(i), nn.ReLU())
                elif dueling_activation == "tanh":
                    advantage_module.add_module(
                        "Q_act_{}".format(i), nn.Tanh())
                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module(
                        "LayerNorm_Q_{}".format(i), nn.LayerNorm(n))
                ins = n

            # Actual Q value layer (nodes=num-actions) and
            # value layer (nodes=1).
            advantage_module.add_module(
                "Q", layer_cls(ins, action_space.n * num_atoms))


        self.advantage_module = advantage_module
        self.value_module = value_module
        # distributional dqn settings
        self.num_atoms = num_atoms
        z = torch.arange(num_atoms).float()
        z = v_min + z * (v_max - v_min) / float(num_atoms - 1)
        self.z = z  # return distribution support 

        # augmentations 
        self.augmentation = augmentation
        self.aug_num = aug_num
        if augmentation:
            obs_shape = obs_space.shape[-2]
            self.trans = nn.Sequential(
                nn.ReplicationPad2d(max_shift),
                RandomCrop((obs_shape, obs_shape))
            )
    
    def get_advantages_or_q_values(self, model_out):
        """ Returns distributional values for Q(s, a) given a state embedding.
        Override this in your custom model to customize the Q output head.
        Arguments:
            model_out (Tensor): embedding from the model layers
        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        # return self.advantage_module(model_out)
        action_scores = self.advantage_module(model_out)
        if self.num_atoms > 1:
            support_logits_per_action = action_scores.reshape(
                -1, self.action_space.n, self.num_atoms)
            support_prob_per_action = F.softmax(support_logits_per_action)

            action_scores = torch.sum(
                self.z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            dist = support_prob_per_action
            return [action_scores, self.z, support_logits_per_action, logits, dist]
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            dist = torch.unsqueeze(torch.ones_like(action_scores), -1)
            return [action_scores, logits, dist]

    def get_state_value(self, model_out):
        """ Returns the state value prediction for the given state embedding.
        """
        return self.value_module(model_out)

    # NOTE: customs 
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """ return action logits/scores # return embedding value
        """
        x, state = self.get_embeddings(input_dict, state, seq_lens)
        # logits = self.get_advantages_or_q_values(x)[0]
        return x, state

    def get_embeddings(self, input_dict, state, seq_lens, permute=True):
        """ encode observations 
        """
        x = input_dict["obs"].float()
        if permute:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.encoder(x)
        return x, state



#######################################################################################################
#####################################   Misc   #####################################################
#######################################################################################################

# Register model in ModelCatalog
ModelCatalog.register_custom_model("drq_rainbow", DrqRainbowTorchModel)



