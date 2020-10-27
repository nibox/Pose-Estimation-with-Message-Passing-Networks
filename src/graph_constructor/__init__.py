from .ConstructGraphClasses import ClassAgnosticGraphConstructor
from .ConstructGraph import NaiveGraphConstructor


def get_graph_constructor(config, **kwargs):
    return NaiveGraphConstructor(config=config, **kwargs)
