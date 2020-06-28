import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

from models import make_encoder


#######################################################################################################
#####################################   Helper stuff   #####################################################
#######################################################################################################

class BaselineDQNTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality.
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
            #  customs 
            embed_dim = 256,
            encoder_type="impala",
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
        super(BaselineDQNTorchModel, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        # NOTE: customs 
        self.embed_dim = embed_dim
        h, w, c = obs_space.shape
        shape = (c, h, w)
        # obs embedding 
        self.encoder = make_encoder(encoder_type, shape, out_features=embed_dim)
  
        # NOTE: value output branches 
        self.dueling = dueling
        # ins = num_outputså
        ins = embed_dim

        # Dueling case: Build the shared (advantages and value) fc-network.
        advantage_module = nn.Sequential()
        value_module = None
        if self.dueling:
            value_module = nn.Sequential()
            for i, n in enumerate(q_hiddens):
                advantage_module.add_module("dueling_A_{}".format(i),
                                            nn.Linear(ins, n))
                value_module.add_module("dueling_V_{}".format(i),
                                        nn.Linear(ins, n))
                # Add activations if necessary.
                if dueling_activation == "relu":
                    advantage_module.add_module("dueling_A_act_{}".format(i),
                                                nn.ReLU())
                    value_module.add_module("dueling_V_act_{}".format(i),
                                            nn.ReLU())
                elif dueling_activation == "tanh":
                    advantage_module.add_module("dueling_A_act_{}".format(i),
                                                nn.Tanh())
                    value_module.add_module("dueling_V_act_{}".format(i),
                                            nn.Tanh())

                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module("LayerNorm_A_{}".format(i),
                                                nn.LayerNorm(n))
                    value_module.add_module("LayerNorm_V_{}".format(i),
                                            nn.LayerNorm(n))
                ins = n
            # Actual Advantages layer (nodes=num-actions) and
            # value layer (nodes=1).
            advantage_module.add_module("A", nn.Linear(ins, action_space.n))
            value_module.add_module("V", nn.Linear(ins, 1))
        # Non-dueling:
        # Q-value layer (use main module's outputs as Q-values).
        else:
            # pass (UGLY HACK!!!)
            # NOTE: manually adding q value (no dueling) branch following embedding
            for i, n in enumerate(q_hiddens):
                advantage_module.add_module("Q_{}".format(i),
                                            nn.Linear(ins, n))
                if dueling_activation == "relu":
                    advantage_module.add_module("Q_act_{}".format(i),
                                                nn.ReLU())
                elif dueling_activation == "tanh":
                    advantage_module.add_module("Q_act_{}".format(i),
                                                nn.Tanh())
                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module("LayerNorm_Q_{}".format(i),
                                                nn.LayerNorm(n))
                ins = n
            # Actual Q value layer (nodes=num-actions) and
            # value layer (nodes=1).
            advantage_module.add_module("Q", nn.Linear(ins, action_space.n))

        self.advantage_module = advantage_module
        self.value_module = value_module
        

    def get_advantages_or_q_values(self, model_out):
        """ Returns distributional values for Q(s, a) given a state embedding.
        Override this in your custom model to customize the Q output head.
        Arguments:
            model_out (Tensor): embedding from the model layers
        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        return self.advantage_module(model_out)

    def get_state_value(self, model_out):
        """ Returns the state value prediction for the given state embedding.
        """
        return self.value_module(model_out)

    # NOTE: customs 
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """ return embedding value
        NOTE: only output embedded output to fit the "compute_q_values" func signature
        from https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn_torch_policy.py
        """
        x, state = self.get_embeddings(input_dict, state, seq_lens)
        # logits = self.get_policy_output(x)
        return x, state

    def get_embeddings(self, input_dict, state, seq_lens, permute=True):
        """ encode observations 
        """
        x = input_dict["obs"].float()
        if permute:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.encoder(x)
        return x, state


# Register model in ModelCatalog
ModelCatalog.register_custom_model("baseline_dqn", BaselineDQNTorchModel)