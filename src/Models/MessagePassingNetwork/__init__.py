from .VanillaMPN import VanillaMPN
from .VanillaMPN2 import VanillaMPN2
from .VanillaMPNFeatureDrop import VanillaMPNDrop
from .ClassificationMPN import ClassificationMPN


def get_mpn_model(config, **kwargs):
    if config.NAME == "VanillaMPN":
        if config.DROP_FEATURE != "":
            return VanillaMPNDrop(config)
        return VanillaMPN(config)
    elif config.NAME == "ClassificationMPN":
        return ClassificationMPN(config)
    elif config.NAME == "VanillaMPN2":
        return VanillaMPN2(config)
    else:
        raise NotImplementedError
