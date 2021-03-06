from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from utils.utils import get_conv_output_shape


#######################################################################################################
#####################################   Helper funcs   #####################################################
#######################################################################################################

# # for 84 x 84 inputs
# OUT_DIM = {2: 39, 4: 35, 6: 31}
# # for 64 x 64 inputs
OUT_DIM = {2: 29, 4: 25, 6: 21}


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


#######################################################################################################
#####################################   Encoders   #####################################################
#######################################################################################################

class PixelEncoder(nn.Module):
    """ Convolutional encoder of pixels observations.
    """
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # out_dim = OUT_DIM[num_layers]
        # NOTE: infer it instead 
        out_dim = self.get_pre_fc_shape(obs_shape, num_filters)[-1]
        
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def get_pre_fc_shape(self, obs_shape, num_filters):
        """ automatic shape inference after con layers 
        NOTE: hard-coded for the above specific conv params 
        """
        shape = obs_shape
        # NOTE: params (kernel, stride, padding, dilation), default (3,1,0,1)
        # 1st layer
        conv_params = (3, 2, 0, 1, num_filters)
        shape = get_conv_output_shape(shape, *conv_params)
        # other conv layers 
        conv_params = (3, 1, 0, 1, num_filters)
        for _ in range(self.num_layers-1):
            shape = get_conv_output_shape(shape, *conv_params)
        return shape 

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        hid = self.forward_conv(obs)
        if detach:
            hid = hid.detach()
            
        try:
            h_fc = self.fc(hid)
        except:
            import ipdb; ipdb.set_trace()
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)



class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass



#######################################################################################################
#####################################   Misc   #####################################################
#######################################################################################################

if __name__ == "__main__":

    # test infering OUT_DIM sizes 
    obs_shape = (3,64,64)
    feature_dim = 128
    num_layers = 4
    
    net = PixelEncoder(obs_shape, feature_dim, num_layers=num_layers)    
    print("test done ...")