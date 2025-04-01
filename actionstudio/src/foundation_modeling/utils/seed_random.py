from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
import time
import numpy as np
import os


def init_device_seed(seed=None, accelerator=None, use_fsdp=False):

    if accelerator is None and use_fsdp:
        accelerator = Accelerator()
    else:
        accelerator = None

    if seed is None:
        seed = np.int32(time.time())
    if accelerator is not None:
        seed += AcceleratorState().process_index
        set_seed(seed, device_specific=False)  # we move the device dependent seed control out of accelerate
    else:
        seed += int(os.environ["RANK"])
    return seed
