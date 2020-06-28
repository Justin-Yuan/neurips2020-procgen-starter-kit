from gym.spaces import Discrete
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch

torch, nn = try_import_torch()

from models import make_encoder


#######################################################################################################
#####################################   Main models   #####################################################
#######################################################################################################

class BaselinePPOTorchModel(TorchModelV2, nn.Module):
    def __init__(self,
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                # customs 
                embed_dim = 256,
                encoder_type="impala",
                **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space,
                                            num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.n
        self.discrete = True
        self.action_outs = q_outs = self.action_dim
        self.action_ins = None  # No action inputs for the discrete case.
        self.embed_dim = embed_dim

        h, w, c = obs_space.shape
        shape = (c, h, w)
        # obs embedding 
        self.encoder = make_encoder(encoder_type, shape, out_features=embed_dim)
  
        self.logits_fc = nn.Linear(in_features=embed_dim, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=embed_dim, out_features=1)

        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        # x = x / 255.0  # scale to 0-1     # done in encoder already 
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.encoder(x)
        # get output stuff 
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value


# Register model in ModelCatalog
ModelCatalog.register_custom_model("baseline_ppo", BaselinePPOTorchModel)
