from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train

def get_network(name):
    """Get a network by name."""
    if name == "VGGnet_test":
        return VGGnet_test()
    if name == "VGGnet_train":
        return VGGnet_train()
    raise KeyError("Unknown dataset: {}".format(name))
