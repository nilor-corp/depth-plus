import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from contextlib import nullcontext
from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

def process_depth():
    print("Processing depth")