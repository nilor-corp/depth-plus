from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
import argparse
import torch
import os
import cv2


class DepthPlusOptical:
    def process_optical(self, video_path=None, outdir=None, mp4=True, png=False, exr=False, is_png_8bit=True):
        print("running optical flow")

        if(video_path is None):
            video_path=r"test-video"
        if(outdir is None):    
            outdir=r"test-video-output"

        parser = argparse.ArgumentParser()
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        args = parser.parse_args()
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(r"models\raft-things.pth"))
        model = model.module
        model.to(device)
        model.eval()

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

        os.makedirs(outdir, exist_ok=True)
        #iterate through all videos and process one by one
        for k, filename in enumerate(filenames):
            print(f'Progress: {k+1}/{len(filenames)}: {filename}')

            raw_video = cv2.VideoCapture(filename)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
            output_width = frame_width

            filename = os.path.basename(filename)
            basename = filename[:filename.rfind('.')]
            type_dir = os.path.join("optical", basename)
            output_dir = os.path.join(outdir, type_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if mp4:
                mp4_output_path_and_name = os.path.join(output_dir, f'{basename}_optical.mp4')
                mp4_out = cv2.VideoWriter(mp4_output_path_and_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (output_width, frame_height))
                print("Writing mp4 to: ", mp4_output_path_and_name)
            if png:
                png_output_path = os.path.join(output_dir, f'{basename}_optical_png')
                if(is_png_8bit):
                    png_output_path = f"{png_output_path}_8bit"
                else:
                    png_output_path = f"{png_output_path}_16bit"
                if not os.path.exists(png_output_path):
                    os.makedirs(png_output_path)
                print("Writing png's to: ", png_output_path)
            if exr:
                exr_output_path = os.path.join(output_dir, f'{basename}_optical_exr_32bit')
                if not os.path.exists(exr_output_path):
                    os.makedirs(exr_output_path)
                print("Writing exr's to: ", exr_output_path)

            frame_count = 0
            prev_frame = None
            while raw_video.isOpened():
                with torch.no_grad():
                    ret, curr_frame = raw_video.read()  
                    if not ret:
                        break
                    if prev_frame is None:
                        prev_frame = curr_frame
                        continue
                    img1 = torch.from_numpy(prev_frame).permute(2, 0, 1).float()
                    frame1 = img1[None].to(device)
                    img2 = torch.from_numpy(curr_frame).permute(2, 0, 1).float()
                    frame2 = img2[None].to(device)
                    padder = InputPadder(frame1.shape)
                    frame1, frame2 = padder.pad(frame1, frame2)
                    flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
                    img = frame1[0].permute(1, 2, 0).cpu().numpy()
                    flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    flo = flow_viz.flow_to_image(flo) 

                if mp4:
                    mp4_frame = (flo[:,:,[2,1,0]] * 255).astype('uint8')
                    mp4_out.write(mp4_frame)
                prev_fame = curr_frame
            raw_video.release()
            if mp4:
                mp4_out.release()
        print("Optical flow processing complete")

