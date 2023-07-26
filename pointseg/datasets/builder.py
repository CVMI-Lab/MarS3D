from pointseg.utils.registry import Registry

DATASETS = Registry('datasets')


def build_dataset(cfg):
    """Build test_datasets."""
    return DATASETS.build(cfg)
