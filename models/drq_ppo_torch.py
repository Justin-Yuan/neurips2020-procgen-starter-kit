from gym.spaces import Discrete
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
torch, nn = try_import_torch()
from models.impala_cnn_torch import ResidualBlock, ConvSequence
from ray.rllib.utils.annotations import override
import kornia

class DrqPPOTorchModel(TorchModelV2, nn.Module):
    def __init__(self,
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                embed_dim = 256,
                augmentation=False,
                aug_num=2,
                max_shift=4,
                **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space,
                                            num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.n
        self.discrete = True
        self.action_outs = q_outs = self.action_dim
        self.action_ins = None  # No action inputs for the discrete case.
        self.embed_dim = embed_dim
    

        if augmentation:
            obs_shape = obs_space.shape[-2]
        self.trans = nn.Sequential(
            nn.ReplicationPad2d(max_shift),
            kornia.augmentation.RandomCrop((obs_shape, obs_shape))
        )

        h, w, c = obs_space.shape
        shape = (c, h, w)
  
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value


# Register model in ModelCatalog
ModelCatalog.register_custom_model("drq_ppo", DrqPPOTorchModel)
