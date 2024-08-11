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
        # custom_config = {
        #     'model_name': model,
        # }
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified path does not exist: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
        # if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
        #     self.current_config = custom_config
        #     download_path = os.path.join(folder_paths.models_dir, "depthanything")
        #     model_path = os.path.join(download_path, model)

            
        #     if not os.path.exists(model_path):
        #         raise FileNotFoundError(f"The specified path does not exist: {model_path}")
                
        #     print(f"Loading model from: {model_path}")

        #     if "vitl" in model:
        #         encoder = "vitl"
        #     elif "vitb" in model:
        #         encoder = "vitb"
        #     elif "vits" in model:
        #         encoder = "vits"

        #     if "hypersim" in model:
        #         max_depth = 20.0
        #     else:
        #         max_depth = 80.0

        #     with (init_empty_weights() if is_accelerate_available else nullcontext()):
        #         if 'metric' in model:
        #             self.model = DepthAnythingV2(**{**model_configs[encoder], 'is_metric': True, 'max_depth': max_depth})
        #         else:
        #             self.model = DepthAnythingV2(**model_configs[encoder])
            
        #     state_dict = load_torch_file(model_path)
        #     if is_accelerate_available:
        #         for key in state_dict:
        #             set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=state_dict[key])
        #     else:
        #         self.model.load_state_dict(state_dict)

        #     self.model.eval()
        #     da_model = {
        #         "model": self.model,
        #         "dtype": dtype,
        #         "is_metric": self.model.is_metric
        #     }
           
        # return (da_model,)

    def process_depth(self):
        print("Processing depth")
        self.load_model("models\depth_anything_v2_vitl.pth")