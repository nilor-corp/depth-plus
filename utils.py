import torch
import safetensors.torch
import numpy as np

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def get_bitsize_from_torch_type(torch_type):
    bit8 = 255
    bit16 = 65535
    bit32 = 4294967295
    print("torch_type: ", torch_type)

    if(torch_type == torch.float8_e4m3fn or torch_type == torch.float8_e5m2 or torch_type == torch.float8_e4m3fnuz or torch_type == torch.float8_e5m2uz):
        return bit8, np.uint8
    elif(torch_type == torch.float16):
        return bit16, np.uint16
    elif(torch_type == torch.float32):
        return bit32, np.uint32
    else:
        raise ValueError("Invalid torch type get_bitsize_from_torch_type")
    