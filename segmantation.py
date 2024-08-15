import torch
import os
from utils import write_out_video_as_jpeg_sequence, delete_directory

class DepthPlusSegmentation:
    def process_segmentation(self, video_path=None, outdir=None):
        print("Processing segmentation")
        if(video_path is None or video_path == ""):
            video_path=r"test-video\S1_DOLPHINS_A_v1-trim.mp4"
        if(outdir is None or outdir == ""):    
            outdir=r"test-video-output"
        model_path = r"models\sam2_hiera_large.pt"

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'

          # You can provide a file path or a directory full of videos
        if os.path.isfile(video_path):
            if(video_path.endswith(".mp4")):
                print("Video file found at: ", video_path)
                filenames = [video_path]
            else:
                raise ValueError("The file is not an mp4 file")
        else :
            filenames = os.listdir(video_path)
            filenames = [os.path.join(video_path, file) for file in filenames if file.endswith(".mp4")]

        print(f"outdir: {outdir}")
        os.makedirs(outdir, exist_ok=True)

        #iterate through all videos and process one by one
        for k, filename in enumerate(filenames):
            print(f'Progress: {k+1}/{len(filenames)}: {filename}')
            jpg_dir = write_out_video_as_jpeg_sequence(video_path,filename)

            #clean up temp jpgs
            delete_directory(jpg_dir)

            
