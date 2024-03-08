from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
import time
import numpy as np


def init_device_seed(seed=None):

    accelerator = Accelerator()

    if seed is None:
        seed = np.int32(time.time())
    seed += AcceleratorState().process_index
    set_seed(seed, device_specific=False) # we move the device dependent seed control out of accelerate
    return seed
