model = dict(
    type="BasePointTransformerSegmentor",
    in_channels=None,
    num_classes=None,
    point_embedding=dict(
        type="MLPEmbedding",
        in_channels=None,  # dummy parameter
        embed_dims=None,
    ),
    backbone=dict(
        type="PointTransformerUnet",
        in_channels=None,  # dummy parameter
        base_channels=32,
        share_channels=8,
        num_stages=5,
        down_strides=(4, 4, 4, 4),
        down_nsamples=(16, 16, 16, 16),
        transformer_nsamples=(8, 16, 16, 16, 16),
        enc_num_layers=(1, 1, 1, 1, 1),
        dec_num_layers=(1, 1, 1, 1),
    ),
    auxiliary_head=dict(
        type="MLPHead",
        in_channels=None,  # dummy parameter
        num_classes=None  # dummy parameter
    )

)

