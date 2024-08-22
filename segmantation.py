import torch
import os
import json
from utils import write_out_video_as_jpeg_sequence, delete_directory, get_int_sorted_dir_list
from sam2.build_sam import build_sam2_video_predictor
from unittest.mock import patch

#workaround for unnecessary flash_attn requirement
#---------
from transformers.dynamic_module_utils import get_imports
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports
#---------

from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np

class DepthPlusSegmentation:
    def process_segmentation(self, video_path=None, outdir=None):
        print("Processing segmentation")
        if(video_path is None or video_path == ""):
            video_path=r"test-video\S1_DOLPHINS_A_v1-trim.mp4"
            video_path=r"test-video\S5_Abstract_B_v1_15sec-trim.mp4"
        if(outdir is None or outdir == ""):    
            outdir=r"test-video-output\segmentation"
        model_path = r"models\sam2_hiera_large.pt"
        model_config_path = r"sam2_hiera_l.yaml"
        #all_model_config_paths = r"C:\ocean\depth-plus-plus\sam_2\sam2_configs"
        relative_sam_path = r"\sam_2\sam2_configs"
        all_model_config_paths = os.path.abspath(relative_sam_path)

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
            jpg_dir, width, height, frame_rate = write_out_video_as_jpeg_sequence(video_path,filename)

            self.florence_object_detection(jpg_dir)

            predictor = build_sam2_video_predictor(model_config_path, model_path, device)
            video_out = os.path.join(outdir, "segmentation.mp4")

            mp4_out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))


            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                state = predictor.init_state(jpg_dir)
                #iterate through jpg_dir and add new points or boxes
                jpg_list = get_int_sorted_dir_list(jpg_dir, ".jpg")
                obj_list = get_int_sorted_dir_list(jpg_dir, ".txt")
                if(len(jpg_list) != len(obj_list)):
                    print(f"Error: number of jpgs and txts do not match: {len(jpg_list)} != {len(obj_list)}")
                for i, filename in enumerate(jpg_list):
                    with open(f"{jpg_dir}/{obj_list[i]}", "r") as f:
                        content = f.read()
                        parsed_object = json.loads(content)
                    bboxes = parsed_object["<OD>"]["bboxes"]
                    labels = parsed_object["<OD>"]["labels"]
                    points = [( (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 ) for bbox in bboxes]

                    frame_idx = i
                    print(f"labels: {labels}")
                    #print(f"obj_list: {obj_list}")
                    for i, box in enumerate(bboxes):
                        #print area of bounding bx
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        print(f"area: {i} = {area}")
                    for i, label in enumerate(labels):
                        print(f"enumrate labels: {i}, {label}")
                        labels = np.array([i],np.int32)
                        obj_idx = i
                        _, object_ids, masks = predictor.add_new_points_or_box(state, frame_idx, obj_idx, box=bboxes[i])

                #frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 0,1)
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    #write out masks
                    # print(f"frame_idx: {frame_idx}")
                    # print(f"object_ids: {object_ids}")
                    #print(f"masks: {masks}")
                    combined_mask = np.zeros((height,width), dtype=np.uint8)
                    for i, mask in enumerate(masks):
                        # print(f"mask len: {len(mask)}")
                        # print(f"mask[{i}].shape: {mask.shape}")
                        #print(f"masksI {masks[i]}")
                        # if(i == 0):
                        #     print("skipping")
                        #     continue
                        mask_temp = (mask[0] > 0.0).cpu().numpy()
                        combined_mask = np.logical_or(combined_mask, mask_temp)
                        mask_frame_out = (mask_temp * 255).astype(np.uint8)
                        cv2.imwrite(f"{outdir}/mask_{frame_idx}_{object_ids[i]}.png", mask_frame_out)
                        #print(f"mask_{frame_idx}_{object_ids[i]}.png saved to {outdir}")
                    mask_out = (combined_mask * 255).astype(np.uint8)
                    mask_out_bgr = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
                    mp4_out.write(mask_out_bgr)

            mp4_out.release()
            #clean up temp jpgs
            delete_directory(jpg_dir)
    def florence_object_detection(self, jpg_dir):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = r"models\Florence-2-large"
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation='sdpa', trust_remote_code=True, torch_dtype=torch_dtype).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        task_prompt = "<OD>"

        def extract_number(file_name):
            return int(''.join(filter(str.isdigit, file_name)))

        #interate through each image in jpg_dir
        file_list = sorted(
            [f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg")],
            key=extract_number)
        print(file_list)
        for filename in file_list:
            image = cv2.imread(os.path.join(jpg_dir, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            generate_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(width, height))
            
            #write parsed answer to file
            with open(f"{jpg_dir}/{filename}.txt", "w") as f:
                f.write(json.dumps(parsed_answer))
            #open and read the file after the appending:
            with open(f"{jpg_dir}/{filename}.txt", "r") as f:
                content = f.read()
                parsed_object = json.loads(content)
            
            print(f"Parsed object: {parsed_object}")
            print(f"bboxes: {parsed_object[task_prompt]['bboxes']}")
            print(f"labels: {parsed_object[task_prompt]['labels']}")

        pass    

            
