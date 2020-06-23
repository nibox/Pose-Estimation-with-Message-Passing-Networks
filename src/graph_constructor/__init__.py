from .ConstructGraphClasses import ClassAgnosticGraphConstructor
from .ConstructGraph import NaiveGraphConstructor


def get_graph_constructor(config, **kwargs):
    if config.NAME == "naive_graph_constructor":
        return NaiveGraphConstructor(config=config, **kwargs)
    elif config.NAME == "class_agnostic_graph_constructor":
        return ClassAgnosticGraphConstructor(config=config, **kwargs)
    else:
        raise NotImplementedError