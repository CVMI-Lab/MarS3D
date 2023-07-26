# dataset settings
dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti"
cache_data = False
ignore_label = 255
names = ["car", "bicycle", "motorcycle", "truck", "other-vehicle",
         "person", "bicyclist", "motorcyclist", "road", "parking",
         "sidewalk", "other-ground", "building", "fence", "vegetation",
         "trunk", "terrain", "pole", "traffic-sign"]
learning_map = {
    0: ignore_label,  # "unlabeled"
    1: ignore_label,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: ignore_label,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: ignore_label,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

data = dict(
    num_classes=19,
    ignore_label=ignore_label,
    names=names,
    train=dict(
        type=dataset_type,
        mode='sh',
        split="train",
        data_root=data_root,
        fusion_frame=[0],
        learning_map=learning_map,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.9),
            dict(type="PointClip", point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="Voxelize", voxel_size=0.05, hash_type='fnv', mode='train', return_discrete_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=("coord", "color"))
            # color is placed by remission
        ],
        cache_data=cache_data,
        loop=1,
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        mode='sh',
        split="val",
        fusion_frame=[0],
        data_root=data_root,
        learning_map=learning_map,
        transform=[
            dict(type="PointClip", point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type="Voxelize", voxel_size=0.05, hash_type='fnv', mode='train', return_discrete_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=("coord", "color"))
        ],
        cache_data=cache_data,
        loop=1,
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        mode='sh',
        split="val",
        fusion_frame=[0],
        data_root=data_root,
        learning_map=learning_map,
        transform=[
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=("coord", "color"))
        ],
        cache_data=cache_data,
        loop=1,
        test_mode=True,
        test_cfg=dict()
    ),
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=ignore_label)
]

