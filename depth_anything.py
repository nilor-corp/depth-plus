import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from contextlib import nullcontext
from contextlib import nullcontext
from utils import load_torch_file
from depth_anything_v2.dpt import DepthAnythingV2

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

class DepthPlusDepth:
    def load_model(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
        dtype = torch.float16 if "fp16" in model_path else torch.float32
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            #'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified path does not exist: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
        if "vitl" in model_path:
            encoder = "vitl"
        elif "vitb" in model_path:
            encoder = "vitb"
        elif "vits" in model_path:
            encoder = "vits"

        if "hypersim" in model_path:
            max_depth = 20.0
        else:
            max_depth = 80.0

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            if 'metric' in model_path:
                model = DepthAnythingV2(**{**model_configs[encoder], 'is_metric': True, 'max_depth': max_depth})
            else:
                model = DepthAnythingV2(**model_configs[encoder])
        
        state_dict = load_torch_file(model_path)
        if is_accelerate_available:
            for key in state_dict:
                set_module_tensor_to_device(model, key, device=device, dtype=dtype, value=state_dict[key])
        else:
            model.load_state_dict(state_dict)

        model.eval()
        da_model = {
            "model": model,
            "dtype": dtype,
            "is_metric": model.is_metric
        }
        return da_model

    def process_depth(self):
        print("Processing depth")
        self.load_model(r"models\depth_anything_v2_vitl.pth")
        print("Loaded model")