from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
import argparse
import torch
import os
import cv2
from utils import get_bitsize_from_torch_type, make_exr, construct_output_paths
import gradio as gr


class DepthPlusOptical:
    def process_optical(self, progress=gr.Progress(), video_path=None, outdir=None, mp4=True, png=False, exr=False, is_png_8bit=True):
        print("running optical flow")

        if(video_path is None or video_path==""):
            video_path=r"test-video"
        if(outdir is None or outdir==""):    
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

        print(f"outdir: {outdir}")
        os.makedirs(outdir, exist_ok=True)

        #iterate through all videos and process one by one
        mp4s_out = []
        for k, filename in enumerate(filenames):
            progress(k/len(filenames), desc=f"Processing video {k+1}/{len(filenames)}")
            
            paths = construct_output_paths(filename, outdir, "optical", is_png_8bit=is_png_8bit, is_exr_32bit=True)


            raw_video = cv2.VideoCapture(filename)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS)
            output_width = frame_width

            if mp4:
                mp4_output_path = paths['mp4']
                mp4_out = cv2.VideoWriter(mp4_output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (output_width, frame_height))
                print("Writing mp4 to: ", mp4_output_path)
            if png:
                png_output_path = paths['png']
                print("Writing png's to: ", png_output_path)
            if exr:
                exr_output_path = paths['exr']
                print("Writing exr's to: ", exr_output_path)

            frame_count = 0
            total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing {total_frames} frames...")
            
            prev_frame = None
            while raw_video.isOpened():
                progress((frame_count + 1) / total_frames, 
                        desc=f"Processing frame {frame_count + 1}/{total_frames}")
                
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
                        #force 8 bit if mp4
                        bitsize, nptype = get_bitsize_from_torch_type(torch.float8_e4m3fn)
                        mp4_frame = (flo[:,:,[2,1,0]] * bitsize).astype(nptype)
                        mp4_out.write(mp4_frame)
                    if png:
                        if is_png_8bit:
                            bitsize, nptype = get_bitsize_from_torch_type(torch.float8_e4m3fn)
                        else:
                            bitsize, nptype = get_bitsize_from_torch_type(torch.float16)
                        png_frame = ((flo[:,:,[2,1,0]]/255.0) * bitsize).astype(nptype)
                        png_filename = os.path.join(png_output_path, '{:04d}.png'.format(frame_count))
                        success = cv2.imwrite(png_filename, png_frame)
                        if not success:
                            raise ValueError("Error writing png file")
                    if exr:
                        bitsize, nptype = get_bitsize_from_torch_type(torch.float32)
                        exr_frame = ((flo[:,:,[2,1,0]]/255.0) * bitsize).astype(nptype)
                        exr_filename = os.path.join(exr_output_path, '{:04d}.exr'.format(frame_count))
                        success = make_exr(exr_filename, exr_frame)
                        if not success:
                            raise ValueError("Error writing exr file")
                frame_count += 1
                prev_frame = curr_frame

            raw_video.release()

            if mp4:
                mp4s_out.append(mp4_output_path)
                mp4_out.release()
                
            print("Optical flow processing complete")

        return mp4s_out

