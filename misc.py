import gc
import os

import torch


CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
NL_PREDICATES_CACHE_DIR = os.path.join(CACHE_DIR, 'nl-predicates')


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
    fname = model_id.replace('/', '__') + '--('
    fname += ';'.join([f'{k}={v}' for k, v in kwargs.items()]) if kwargs else ''
    fname += ')'
    return fname