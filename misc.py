import torch
import gc


def remove_from_gpu_memory(*items):
    # delete all items and invoke garbage collection
    for item in items:
        del item
    gc.collect()
    
    # clear GPU cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

