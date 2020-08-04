from .VanillaMPN import VanillaMPN
from .VanillaMPN2 import VanillaMPN2
from .VanillaMPNFeatureDrop import VanillaMPNDrop
from .ClassificationMPN import ClassificationMPN
from .ClassificationMPNSimple import ClassificationMPNSimple
from .ClassificationNaive import ClassificationNaive
from .NodeClassificationMPNSimple import NodeClassificationMPNSimple



def get_mpn_model(config, **kwargs):
    if config.NAME == "VanillaMPN":
        if config.DROP_FEATURE != "":
            return VanillaMPNDrop(config)
        return VanillaMPN(config)
    elif config.NAME == "ClassificationMPN":
        return ClassificationMPN(config)
    elif config.NAME == "ClassificationMPNSimple":
        return ClassificationMPNSimple(config)
    elif config.NAME == "VanillaMPN2":
        return VanillaMPN2(config)
    elif config.NAME == "ClassificationNaive":
        return ClassificationNaive(config)
    elif config.NAME == "NodeClassificationMPN":
        return NodeClassificationMPNSimple(config)
    else:
        raise NotImplementedError
