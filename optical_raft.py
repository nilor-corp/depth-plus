from raft.core.raft import RAFT
import argparse
import torch


class DepthPlusOptical:
    def process_optical(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(r"models\raft-things.pth"))
        print("running optical flow")
        

