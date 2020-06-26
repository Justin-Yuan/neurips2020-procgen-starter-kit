from models.pixel_encoder import IdentityEncoder, PixelEncoder



_AVAILABLE_ENCODERS = {
    "identity": IdentityEncoder,
    "pixel": PixelEncoder, 
    "impala": 
}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )