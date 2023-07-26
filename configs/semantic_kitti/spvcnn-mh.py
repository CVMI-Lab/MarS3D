_base_ = [
    '../_base_/datasets/semantic_kitti_25cls.py',
    '../_base_/schedulers/one-cycle_adamw_epoch50.py',
    '../_base_/tests/segmentation.py',
    '../_base_/default_runtime.py'
]

batch_size = 4
batch_size_val = 2
empty_cache = False
enable_amp = False


model = dict(
    type="SPVCNNMH",
    in_channels=4,
    out_channels=25,
    bs = 2,
    fs = 3,
    channels=(32, 64, 128, 256, 256, 128, 96, 96),
    layers=(2, 2, 2, 2, 2, 2, 2, 2)
)

epochs = 50
start_epoch = 0
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.005)
scheduler = dict(type='OneCycleLR',
                 max_lr=optimizer["lr"],
                 epochs=epochs,
                 steps_per_epoch=1,
                 pct_start=0.04,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=100.0)
