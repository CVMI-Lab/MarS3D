from pointseg.utils.registry import Registry

MODELS = Registry('models')
MODULES = Registry('modules')


def build_model(cfg):
    """Build test_datasets."""
    return MODELS.build(cfg)
