from models.pixel_encoder import IdentityEncoder, PixelEncoder
from models.impala_cnn_torch import ImpalaEncoder
from models.pixel_decoder import PixelDecoder
from models.vision_network_torch import VisionNetworkEncoder

#######################################################################################################
#####################################   Encoders   #####################################################
#######################################################################################################

_AVAILABLE_ENCODERS = {
    "identity": IdentityEncoder,
    "pixel": PixelEncoder, 
    "impala": ImpalaEncoder,
    "vision": VisionNetworkEncoder,
}

# def make_encoder(
#     encoder_type, obs_shape, feature_dim, num_layers, num_filters, **kwargs
# ):
#     assert encoder_type in _AVAILABLE_ENCODERS
#     return _AVAILABLE_ENCODERS[encoder_type](
#         obs_shape, feature_dim, num_layers, num_filters, **kwargs
#     )

def make_encoder(encoder_type, *args, **kwargs):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](*args, **kwargs)


#######################################################################################################
#####################################   Decoders   #####################################################
#######################################################################################################

_AVAILABLE_DECODERS = {
    "pixel": PixelDecoder
}


# def make_decoder(
#     decoder_type, obs_shape, feature_dim, num_layers, num_filters, **kwargs
# ):
#     assert decoder_type in _AVAILABLE_DECODERS
#     return _AVAILABLE_DECODERS[decoder_type](
#         obs_shape, feature_dim, num_layers, num_filters, **kwargs
#     )

def make_decoder(decoder_type, *args, **kwargs):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](*args, **kwargs)