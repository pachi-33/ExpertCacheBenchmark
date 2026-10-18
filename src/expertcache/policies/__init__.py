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
    elif config.name == "belady":
        from .belady import Belady
        cls = Belady
    else:
        cls = import_symbol(config.name)
    policy = cls(**config.params)
    if not isinstance(policy, CachePolicy): raise TypeError("policy must subclass CachePolicy")
    return policy


def implementation_identity(instance):
    import inspect
    from ..data import digest_file
    cls = type(instance)
    result = {"class": cls.__module__ + ":" + cls.__qualname__}
    try:
        path = inspect.getsourcefile(cls)
        if path: result["source_sha256"] = digest_file(path)
    except (TypeError, OSError):
        pass
    return result
