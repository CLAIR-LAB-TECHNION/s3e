import gc
import random

import numpy as np
import torch

from importlib import import_module

ENTRYPOINT_MODULE_FUNC_SEP = ":"


def remove_from_gpu_memory(*items):
    # delete all items and invoke garbage collection
    for item in items:
        del item
    gc.collect()

    # clear GPU cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_and_kwargs_to_filename(model_id: str, **kwargs):
    """
    turn model arguments into filenames for caching purposes.
    the filename will be:
        model_id--(k1=v1;k2=v2;...kn=vn)
    """
    fname = model_id.replace("/", "__") + "--("
    fname += ";".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
    fname += ")"
    return fname


def load_from_entrypoint(entrypoint: str):
    """
    Loads an exported value from a given module described as an entrypoint string. The entrypoint string is of the form
    `module_name:function_name`. For example, we can load function `foo` from module `my_module.py` in package
    `my_package` with entrypoint "my_package.my_module:foo".

    >>> load_from_entrypoint("math:pi")
    3.141592653589793
    >>> load_from_entrypoint("math:ceil")(2.3)
    3

    :param entrypoint: A string of the form `path.to.module:class_or_function`.
    :return: The function specified by the entrypoint string.
    """
    # split the module path from the name of the exported value
    module_path, exported_name = entrypoint.split(ENTRYPOINT_MODULE_FUNC_SEP)

    # dynamically import the module
    module = import_module(module_path)

    # get the desired exported value
    exported = getattr(module, exported_name)

    return exported


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
