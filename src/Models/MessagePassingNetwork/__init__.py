from .VanillaMPN import VanillaMPN
from .VanillaMPN2 import VanillaMPN2


def get_mpn_model(config, **kwargs):
    if config.NAME == "VanillaMPN":
        return VanillaMPN(config)
    elif config.NAME == "VanillaMPN2":
        return VanillaMPN2(config)
    else:
        raise NotImplementedError
