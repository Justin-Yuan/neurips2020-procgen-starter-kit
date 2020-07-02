from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

from utils.utils import get_conv_output_shape 


#######################################################################################################
#####################################   Helper classes   #####################################################
#######################################################################################################

def get_activation_func(act_name):
    if act_name == "relu":
        return nn.functional.relu 
    elif act_name == "tanh":
        return nn.functional.tanh 
    else:   # default to identity
        return lambda x: x


class ConvLayers(nn.Module):
    def __init__(self, obs_shape, conv_filters, conv_activation="relu"):
        """ conv stack for encoder 
        Arguments:
            - conv_filters: list of [x, x, x] for (out_channels, kernel, stride)
        """
        super(ConvLayers, self).__init__()
        convs, shape = [], obs_shape
        for out_channels, kernel, stride in conv_filters:
            convs.append(
                nn.Conv2d(shape[0], out_channels, kernel, stride=stride))
            # update output shape 
            conv_params = (kernel, stride, 0, 1, out_channels)
            shape = get_conv_output_shape(shape, *conv_params)

        # register to module 
        self.convs = nn.ModuleList(convs)
        self.output_shape = shape 
        # activations 
        self.act = get_activation_func(conv_activation)
        
    def forward(self, x):
        for conv_layer in self.convs:
            x = self.act(conv_layer(x))
        return x


class FCLayers(nn.Module):
    def __init__(self, in_dim, out_dim, fcnet_hiddens=[], fcnet_activation="relu", last_act="relu"):
        """ fully-connected stack of encoder (after conv stack)
        Arguments:
            - fcnet_hiddens: [x, x, x]
        """ 
        super(FCLayers, self).__init__()
        fcs, ins = [], in_dim
        for dim in fcnet_hiddens:
            fcs.append(nn.Linear(ins, dim))
            ins = dim 
        self.fcs = nn.ModuleList(fcs)
        self.act = get_activation_func(fcnet_activation)
        # final layer 
        self.last_layer = nn.Linear(ins, out_dim)
        self.final_act = get_activation_func(last_act)
        self.output_shape = out_dim 
        
    def forward(self, x):
        for fc_layer in self.fcs:
            x = self.act(fc_layer())
        # last act might be None / identity 
        x = self.final_act(self.last_layer(x))
        return x


#######################################################################################################
#####################################   Encoder   #####################################################
#######################################################################################################


class VisionNetworkEncoder(nn.Module):
    """ embed image inputs using simple CNN 
    """
    DEFAULT_CONV_FILTERS = [
        [16, 3, 3], 
        [16, 3, 1]
    ]

    def __init__(self, 
        obs_shape, 
        out_features=256, 
        conv_filters=None,
        conv_activation="relu", 
        fcnet_hiddens=[256],
        fcnet_activation="relu",
        last_act="relu"
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = self.DEFAULT_CONV_FILTERS
        self.conv_layers = ConvLayers(
            obs_shape, conv_filters, conv_activation
        )

        out_dim = self.conv_layers.output_shape
        fc_in_dim = out_dim[0] * out_dim[1] * out_dim[2]

        self.fc_layers = FCLayers(
            fc_in_dim, out_features, fcnet_hiddens, 
            fcnet_activation, last_act
        )

    def forward(self, x):
        """ assume input is of shape (B,C,H,W)
        """
        x = x / 255.0   # normalize 
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
