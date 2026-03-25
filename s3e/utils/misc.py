import gc
import random

import numpy as np
import torch

from importlib import import_module

ENTRYPOINT_MODULE_FUNC_SEP = ":"

from s3e.constants import (
    LLAMA_70B_INSTRUCT,
    LLAVA_7B_OV,
)


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


def squash_predicate(dp_dicts):
    out = []
    dp_dict_keys_sorted = sorted(dp_dicts.keys())
    predicates = sorted(next(iter(dp_dicts.values())))
    for dp_key in dp_dict_keys_sorted:
        dp_arr = []
        for predicate in predicates:
            dp_arr.append(dp_dicts[dp_key][predicate])
        out.append(dp_arr)

    return np.array(out)


def load_se_from_args(se_class, se_kwargs, domain, problem):
    # these modules import heavy packages.
    # import here to avoid waiting when calling with `--help`
    from s3e.semantic_state_estimator import (
        SemanticStateEstimator,
        SemanticEstimatorMultiImageRun
    )

    # load state estimator class
    if se_class is None:
        se_class = SemanticStateEstimator
    elif isinstance(se_class, str):
        se_class = load_from_entrypoint(se_class)

    # handle default state estimator class
    se_kwargs.setdefault("vqa_model_id", LLAVA_7B_OV)
    
    # load state estimator
    se = se_class(domain=domain, problem=problem, **se_kwargs)

    return se