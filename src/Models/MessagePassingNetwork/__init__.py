from .VanillaMPN import VanillaMPN
from .VanillaMPN2 import VanillaMPN2
from .VanillaMPNFeatureDrop import VanillaMPNDrop
from .ClassificationMPN import ClassificationMPN
from .ClassificationMPNSimple import ClassificationMPNSimple
from .ClassificationNaive import ClassificationNaive
from .NodeClassificationMPNSimple import NodeClassificationMPNSimple
from .NodeClassificationMPNSimpleWithRef import NodeClassificationMPNSimpleWithRef
from .JointTypeClassification import JointTypeClassification
from .NodeClassificationMPNTypeBased import NodeClassificationMPNTypeBased
from .NodeClassificationMPNAttention import NodeClassificationMPNAttention
from .NodeClassificationMPNFPConstrained import NodeClassificationMPNFPConstrained
from .NodeClassificationMPNWithBackground import NodeClassificationMPNWithBackground
from .NodeClassificationMPNTypeConstrained import NodeClassificationMPNTypeConstrained
from .NodeClassificationMPNGroupBased import NodeClassificationMPNGroupBased
from .NodeClassificationMPNSelftAttention import NodeClassificationMPNSelfAttention



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
    elif config.NAME == "NodeClassificationMPNWithBackground":
        return NodeClassificationMPNWithBackground(config)
    elif config.NAME == "NodeClassificationMPNTypeBased":
        return NodeClassificationMPNTypeBased(config)
    elif config.NAME == "NodeClassificationMPNAttention":
        return NodeClassificationMPNAttention(config)
    elif config.NAME == "NodeClassificationMPNSelfAttention":
        return NodeClassificationMPNSelfAttention(config)
    elif config.NAME == "NodeClassificationMPNWithRef":
        return NodeClassificationMPNSimpleWithRef(config)
    elif config.NAME == "NodeClassificationMPNFPConstrained":
        return NodeClassificationMPNFPConstrained(config)
    elif config.NAME == "NodeClassificationMPNTypeConstrained":
        return NodeClassificationMPNTypeConstrained(config)
    elif config.NAME == "NodeClassificationMPNGroupBased":
        return NodeClassificationMPNGroupBased(config)
    elif config.NAME == "JointTypeClassification":
        return JointTypeClassification(config)
    else:
        raise NotImplementedError
