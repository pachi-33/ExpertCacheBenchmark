from importlib import import_module
from ..api import CachePolicy


def import_symbol(path):
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


def make_policy(config):
    if config.name == "lfu":
        from .lfu import LFU
        cls = LFU
    elif config.name == "strata":
        from .strata import Strata
        cls = Strata
    else:
        cls = import_symbol(config.name)
    policy = cls(**config.params)
    if not isinstance(policy, CachePolicy): raise TypeError("policy must subclass CachePolicy")
    return policy
