import json
import os
import time
from datetime import datetime
import gradio as gr
from pathlib import Path
from image_downloader import resolve_online_collection
from image_downloader import organise_local_files
from image_downloader import copy_uploaded_files_to_local_dir
import asyncio
import threading

from depth_anything import DepthPlusDepth 
from depthanyvideo import DepthPlusDepthAnyVideo 
from optical_raft import DepthPlusOptical
from segmantation import DepthPlusSegmentation

import signal
import sys
import io
from contextlib import contextmanager
import queue
from threading import Thread

with open("workflow_definitions.json") as f:
    workflow_definitions = json.load(f)

OUT_DIR = os.path.abspath("./output/")
INPUTS_DIR = os.path.abspath("./inputs/")

running = True
output_type = ""
threads = []
previous_content = ""
tick_timer = None
log_queue = queue.Queue()
log_history = []

class StreamToQueue(io.TextIOBase):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def write(self, text):
        if text.strip():  # Only queue non-empty strings
            self.queue.put(text)
            # Print to original stdout/stderr
            print(text, end='\n', file=sys.__stdout__)
            return len(text)

    def flush(self):
        pass

@contextmanager
def redirect_stdout_stderr():
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    stdout_redirector = StreamToQueue(stdout_queue)
    stderr_redirector = StreamToQueue(stderr_queue)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    sys.stdout = stdout_redirector
    sys.stderr = stderr_redirector
    
    def queue_reader(q, prefix=''):
        while True:
            try:
                line = q.get()
                log_queue.put(f"{prefix}{line}")
            except:
                break

    stdout_thread = Thread(target=queue_reader, args=(stdout_queue,), daemon=True)
    stderr_thread = Thread(target=queue_reader, args=(stderr_queue, 'ERROR: '), daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def signal_handler(signum, frame):
    global running, tick_timer, threads
    print("\nShutdown signal received. Cleaning up...")
    running = False
    
    # Deactivate timer
    if tick_timer:
        tick_timer.active = False
    tick_timer = None
    
    # Wait for threads to finish
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=2.0)
    
    sys.exit(0)


def get_logs():
    logs = []
    while True:
        try:
            log = log_queue.get_nowait()
            log_history.append(log)
            logs.append(log)
        except queue.Empty:
            break
    
    # Keep last 1000 lines
    if len(log_history) > 1000:
        del log_history[:len(log_history)-1000]
    
    return "\n".join(log_history)


#region Content Getters
def get_all_images(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []
    
    files = os.listdir(folder)
    image_files = [
        f for f in files if f.lower().endswith(("png", "jpg", "jpeg", "gif"))
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return image_files

def get_latest_image(folder):
    image_files = get_all_images(folder)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

def get_latest_image_with_prefix(folder, prefix):
    image_files = get_all_images(folder)
    image_files = [
        f for f in image_files if f.lower().startswith(prefix)
    ]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


def get_all_videos(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []
    
    video_files = []
    now = time.time()
    time_threshold = 1.0  # Time in seconds

    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(("mp4", "mov")):
                full_path = os.path.join(root, f)
                mtime = os.path.getmtime(full_path)

                # Exclude files modified within the last second, to prevent returning files that are actively being written to (and as such are unplayable)
                if now - mtime > time_threshold:
                    video_files.append(full_path)
    
    video_files.sort(key=lambda x: os.path.getmtime(x))
    return video_files

def get_latest_video(folder):
    video_files = get_all_videos(folder)
    if not video_files:
        print(f"No video files found in {folder}")
        return None
    latest_video = video_files[-1]
    print(f"Found latest video: {latest_video}")
    return latest_video

def get_latest_video_with_prefix(folder, prefix):
    video_files = get_all_videos(folder)
    video_files = [
        f for f in video_files if f.lower().startswith(prefix)
    ]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_video = os.path.join(folder, video_files[-1]) if video_files else None
    return latest_video
#endregion

#region Info Checkers
def check_for_new_content():
    global latest_content, previous_content
    #print(f"Checking for new content in: {OUT_DIR}\n")

    latest_content = ""
    if output_type == "image":
        latest_content = get_latest_image(OUT_DIR)
    else:
        latest_content = get_latest_video(OUT_DIR)

    if latest_content != previous_content:
        print(f"New content found: {latest_content}")
        previous_content = latest_content

    #output_filepath_component = gr.Markdown(f"{latest_content}")

    return gr.update(value=latest_content)

async def wait_for_new_content(previous_content, output_directory):
    while True:
        latest_content = ""
        if output_type == "video":
            latest_content = get_latest_video(output_directory)
        elif output_type == "image":
            latest_content = get_latest_image(output_directory)
        if latest_content != previous_content:
            print(f"New content created: {latest_content}")
            return latest_content
        await asyncio.sleep(1)
#endregion


def run_depth_plus_wrapper(raw_components, component_info_dict, progress=gr.Progress(track_tqdm=True)):
    for component in raw_components:
        print(f"Component: {component.label}")

    def wrapper(*args):
        # match the component to the arg
        for component, arg in zip(raw_components, args):
            # access the component_info_dict using component.elem_id and add a value field = arg
            component_info_dict[component.elem_id]["value"] = arg

        return run_depth_plus(progress, **component_info_dict, )

    return wrapper

#TODO pass in all the other necessary params from UI, like what metric, bit, etc
def run_depth_plus(progress, **kwargs):
    print("\nDepth+ Triggered")
    #print(f"kwargs: {kwargs}")
    
    run_depth_anything = kwargs["depthanything"]["value"]
    run_depth_anyvideo = kwargs["depthanyvideo"]["value"]
    run_optical = kwargs["flow"]["value"]
    run_segmentation = kwargs["segmentation"]["value"]
    
    if not run_depth_anything and not run_depth_anyvideo and not run_optical and not run_segmentation:
        print("No processing selected, aborting")
        pass

    # Find Inputs
    in_dir = kwargs["input-dir"]["value"]
    out_dir = kwargs["output-dir"]["value"]
    depth_type = kwargs["depth-type"]["value"]
    png = kwargs["png"]["value"]
    png_bit_depth = kwargs["png-bit-depth"]["value"]
    mp4 = kwargs["mp4"]["value"]
    exr = kwargs["exr"]["value"]
    seg_prompt = kwargs["segmentation-prompt"]["value"]
    seg_filter = kwargs["segmentation-filter"]["value"]
    filter_threshold = kwargs["filter-threshold"]["value"]

    # Prepare inputs
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)
    depth_type = depth_type.lower()
    png_bit_depth = png_bit_depth.lower()

    # Print inputs
    print(f"Directory: {in_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"DepthAnythingV2 Enabled: {run_depth_anything}")
    print(f"DepthAnyVideo Enabled: {run_depth_anyvideo}")
    print(f"Depth Type: {depth_type}")
    print(f"Optical Enabled: {run_optical}")
    print(f"Segmentation Enabled: {run_segmentation}")
    print(f"PNG: {png}")
    print(f"MP4: {mp4}")
    print(f"EXR: {exr}")
    print(f"PNG Bit Depth: {png_bit_depth}")
    print(f"Segmentation Prompt: {seg_prompt}")
    print(f"Segmentation Filter: {seg_filter}")
    print(f"Filter Threshold: {filter_threshold}")

    # Set the output type flags based on the selected output type
    metric = False
    if run_depth_anything and "metric" in depth_type:
        metric = True

    if "8" in png_bit_depth:
        is_png_8bit = True
    else:
        is_png_8bit = False
    
    print(f"Running Depth+ with DepthAnythingV2: {run_depth_anything}, DepthAnyVideo: {run_depth_anyvideo}, Optical: {run_optical}, Segmentation: {run_segmentation}")

    depth = None
    optical = None
    segmentation = None

    depth_mp4_paths = []
    optical_mp4_paths = []
    segmentation_mp4_paths = []

    if run_depth_anything:
        depth = DepthPlusDepth()
        depth_mp4_paths = depth.process_depth(progress=progress, video_path=in_dir, outdir=out_dir, metric=metric, mp4=mp4, png=png, exr=exr, is_png_8bit=is_png_8bit)
    if run_depth_anyvideo:
        depth = DepthPlusDepthAnyVideo()
        depth_mp4_paths = depth.process_depth(progress=progress, video_path=in_dir, outdir=out_dir)
    if run_optical:
        optical = DepthPlusOptical()
        optical_mp4_paths = optical.process_optical(progress=progress, video_path=in_dir, outdir=out_dir, mp4=mp4, png=png, exr=exr, is_png_8bit=is_png_8bit)
    if run_segmentation:
        segmentation = DepthPlusSegmentation()
        segmentation_mp4_paths = segmentation.process_segmentation(
            progress=progress,
            video_path=in_dir,
            outdir=out_dir,
            segmentation_prompt=seg_prompt,
            mp4=mp4, png=png, exr=exr,
            is_png_8bit=is_png_8bit,
            seg_filter=seg_filter,
            filter_threshold=filter_threshold)
        pass

    print("Depth+ processing complete")
    
    # Return the first valid video file path with additional logging
    if run_depth_anyvideo and depth_mp4_paths:
        path = depth_mp4_paths[0] if os.path.isfile(depth_mp4_paths[0]) else None
        print(f"Returning DepthAnyVideo output: {path}")
        return path
    elif run_depth_anything and depth_mp4_paths:
        path = depth_mp4_paths[0] if os.path.isfile(depth_mp4_paths[0]) else None
        print(f"Returning DepthAnything output: {path}")
        return path
    elif run_optical and optical_mp4_paths:
        path = optical_mp4_paths[0] if os.path.isfile(optical_mp4_paths[0]) else None
        print(f"Returning Optical output: {path}")
        return path
    elif run_segmentation and segmentation_mp4_paths:
        path = segmentation_mp4_paths[0] if os.path.isfile(segmentation_mp4_paths[0]) else None
        print(f"Returning Segmentation output: {path}")
        return path
    
    print("No valid output paths found")
    return None

def update_gif(workflow_name):
    workflow_json = workflow_definitions[workflow_name]["name"]
    gif_path = Path(f"gifs/{workflow_json}.gif")
    if gif_path.exists():
        return str(gif_path)
    else:
        return None

def select_dynamic_input_option(selected_option, choices):
    print(f"Selected option: {selected_option}")
    # print(f"Choices: {choices}")
    updaters = [gr.update(visible=False) for _ in choices]
    
    # Make the corresponding input visible based on the selected option
    if selected_option in choices:
        #print(f"Reveal option input: {selected_option}")
        selected_index = choices.index(selected_option)
        updaters[selected_index] = gr.update(visible=True)

    return updaters

def process_dynamic_input(selected_option, possible_options, input_type, *option_values):
    print("\nProcessing dynamic input")
    print(f"Selected Option: {selected_option}")
    print(f"Possible Options: {possible_options}")
    print(f"Option Values: {option_values}")

    # Get the selected option
    selected_index = possible_options.index(selected_option)
    selected_value = option_values[selected_index]
    print(f"Selected Value: {selected_value}")

    # process the selected value based on the selected option
    if selected_option == "filepath":
        return selected_value
    elif selected_option == "nilor collection":
        return resolve_online_collection(selected_value, None, False)
    elif selected_option == "upload":
        return copy_uploaded_files_to_local_dir(selected_value, input_type, None, False)
    else:
        return None

def create_dynamic_input(input_type, choices, tooltips, text_label, identifier):
    gr.Markdown(f"##### {input_type.capitalize()} Input", elem_classes="group-label")    
    with gr.Group():            
        selected_option = gr.Radio(choices, label=text_label, value=choices[0])
        if input_type == "images":
            possible_inputs = [
                gr.Textbox(label=choices[0], show_label=False, visible=True, info=tooltips[0]),
                gr.Textbox(label=choices[1], show_label=False, visible=False, info=tooltips[1]),
                gr.Gallery(label=choices[2], show_label=False, visible=False)
            ]
        elif input_type == "video":
            possible_inputs = [
                gr.Textbox(label=choices[0], show_label=False, visible=True, info=tooltips[0]),
                gr.File(label=choices[1], show_label=False, visible=False, file_count="single", type="filepath", file_types=["video"])
            ]

        output = gr.Markdown(elem_id=identifier, elem_classes="group-label")

    # modify visibility of inputs based on selected_option
    selected_option.change(select_dynamic_input_option, inputs=[selected_option, gr.State(choices)], outputs=possible_inputs)

    for input_box in possible_inputs:
        if isinstance(input_box, gr.Textbox):
            input_box.submit(process_dynamic_input, inputs=[selected_option, gr.State(choices), gr.State(input_type)] + possible_inputs, outputs=output)
        elif isinstance(input_box, gr.Gallery) or isinstance(input_box, gr.File):
            input_box.upload(process_dynamic_input, inputs=[selected_option, gr.State(choices), gr.State(input_type)] + possible_inputs, outputs=output)
    return selected_option, possible_inputs, output

# Ensure all elements in self.inputs are valid Gradio components
def filter_valid_components(components):
    valid_components = []
    for component in components:
        if hasattr(component, '_id'):
            valid_components.append(component)
    return valid_components

def toggle_group(checkbox_value):
    # If checkbox is selected, the group of inputs will be visible
    if checkbox_value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def make_visible():
    return gr.update(visible=True)

def watch_input(component, default_value, elem_id):
    resetter_visibility = False
    if component == default_value:
        # Return HTML to reset background color when value matches default
        html = ""
        resetter_visibility = False
    else:
        # Return HTML to change the background color when does NOT match default
        html = f"<style>#{elem_id}  {{ background: var(--background-fill-primary); }}</style>"
        resetter_visibility = True

    return gr.update(value=html, visible=resetter_visibility), gr.update(visible=resetter_visibility)

def reset_input(default_value):
    return default_value

def process_input(input_context, input_key):
    input_details = input_context.get(input_key, None)
    input_type = input_details.get("type", None)
    input_label = input_details.get("label", None)
    input_node_id = input_details.get("node-id", None)
    input_value = input_details.get("value", None)
    input_interactive = input_details.get("interactive", True)
    input_minimum = input_details.get("minimum", None)
    input_maximum = input_details.get("maximum", None)
    input_step = input_details.get("step", 1)
    input_choices = input_details.get("choices", None)
    input_info = input_details.get("info", None)

    # Define a mapping of input types to Gradio components
    component_map = {
        "path": gr.Textbox,
        "string": gr.Textbox,
        "text": gr.Textbox,
        "images": None, # special case for radio selection handled below
        "video": None, # special case for video selection handled below
        "bool": gr.Checkbox,
        "float": gr.Number,
        "int": gr.Number,
        "slider": gr.Slider,
        "radio": gr.Radio, # True radios collect their options from the workflow_definitions.json
        "group": None,
        "toggle-group": gr.Checkbox
    }
    
    component = None
    reset_button = None
    components = []
    components_dict = {}

    with gr.Group():
        if input_type in component_map:
            # Use the mapping to find Gradio component based on input_type
            component_constructor = component_map.get(input_type)
            
            if input_type == "group":
                gr.Markdown(f"##### {input_label}", elem_classes="group-label")    
                
                with gr.Group():
                    # Group of inputs
                    with gr.Group():
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(sub_context, group_input_key)

                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)
            elif input_type == "toggle-group":
                with gr.Group():
                    with gr.Row(equal_height=True):
                        # Checkbox component which enables Group
                        component = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100, info=input_info)
                        
                        # Compact Reset button with reduced width, initially hidden
                        reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)

                    # Group of inputs (initially hidden)
                    with gr.Group(visible=component.value) as input_group:
                        sub_context = input_context[input_key]["inputs"]
                        for group_input_key in sub_context:
                            [sub_components, sub_dict_values] = process_input(sub_context, group_input_key)

                            components.extend(sub_components)
                            components_dict.update(sub_dict_values)

                # Update the group visibility based on the checkbox
                component.change(fn=toggle_group, inputs=component, outputs=input_group, queue=False, show_progress="hidden")
            elif input_type == "images":
                selected_option, inputs, component = create_dynamic_input(
                    input_type,
                    choices=["filepath", "nilor collection", "upload"], 
                    tooltips=["Enter the path of the directory of images and press Enter to submit", "Enter the name of the Nilor Collection and press Enter to resolve"],
                    text_label="Select Input Type", 
                    identifier=input_key
                )
            elif input_type == "video":
                selected_option, inputs, component = create_dynamic_input(
                    input_type,
                    choices=["filepath", "upload"], 
                    tooltips=["Enter the path of the directory of video and press Enter to submit"],
                    text_label="Select Input Type", 
                    identifier=input_key
                )
            elif input_type == "float" or input_type == "int" or input_type == "slider":
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, minimum=input_minimum, maximum=input_maximum, step=input_step, interactive=input_interactive, scale=100, info=input_info)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)
            elif input_type == "radio":
                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(label=input_label, elem_id=input_key, choices=input_choices, value=input_value, scale=100, info=input_info)
                    
                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)
            else:
                if input_type == "path" and input_value is not None:
                    input_value = os.path.abspath(input_value)

                with gr.Row(equal_height=True):
                    # Use the mapping to create components based on input_type
                    component = component_constructor(label=input_label, elem_id=input_key, value=input_value, interactive=input_interactive, scale=100, info=input_info)

                    # Compact Reset button with reduced width, initially hidden
                    reset_button = gr.Button("↺", visible=False, elem_id="reset-button", scale=1, variant="secondary", min_width=5)

            if component is not None:
                components.append(component)
                components_dict[input_key] = input_details
                
                if reset_button is not None:
                    # Trigger the reset check when the value of the input changes
                    html_output = gr.HTML(visible=False)
                    component.change(fn=watch_input, inputs=[component, gr.State(input_value), gr.State(input_key)], outputs=[html_output, reset_button], queue=False, show_progress="hidden")

                    # Trigger the reset function when the button is clicked
                    reset_button.click(fn=reset_input, inputs=[gr.State(input_value)], outputs=component, queue=False, show_progress="hidden")
        else:
            print(f"Whoa! Unsupported input type: {input_type}")

    return [components, components_dict]

def create_tab_interface(workflow_name):
    gr.Markdown("### Workflow Parameters")

    key_context = workflow_definitions[workflow_name]["inputs"]

    components = []
    component_data_dict = {}

    print(f"\nWORKFLOW: {workflow_name}")

    interactive_inputs = []
    noninteractive_inputs = []

    interactive_components = []
    noninteractive_components = []

    for input_key in key_context:
        input_details = key_context[input_key]
        input_interactive = input_details.get("interactive", True)

        if input_interactive:
            interactive_inputs.append(input_key)
        else:
            noninteractive_inputs.append(input_key)

    for input_key in interactive_inputs:
        [sub_components, sub_dict_values] = process_input(key_context, input_key)
        interactive_components.extend(sub_components)
        component_data_dict.update(sub_dict_values)

    if noninteractive_inputs:
        with gr.Accordion("Constants", open=False):
            gr.Markdown("You can edit these constants in `workflow_definitions.json` if you know what you're doing.")
        
            for input_key in noninteractive_inputs:
                [sub_components, sub_dict_values] = process_input(key_context, input_key)
                noninteractive_components.extend(sub_components)
                component_data_dict.update(sub_dict_values)

    components.extend(interactive_components)
    components.extend(noninteractive_components)
    
    return components, component_data_dict

def load_demo():
    global tick_timer
    print("Loading the demo...")
    tick_timer = gr.Timer(value=1.0)

def unload_demo():
    global tick_timer
    print("Unloading the demo...")
    if tick_timer:
        tick_timer.active = False

def setup_signal_handlers():
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, signal_handler)    # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)   # Termination request
            print("Signal handlers set up successfully")
        except ValueError as e:
            print(f"Could not set up signal handlers: {e}")

custom_css = """
.group-label {
    padding: .25rem;
}

#workflow-info {
    background-image: linear-gradient(120deg, var(--neutral-800) 0%, var(--neutral-900) 70%, var(--primary-800) 100%);
}

#run-button {
    background-color: var(--primary-600);
}

html {
    overflow-y: scroll;
}

.logs textarea {
    font-family: monospace;
    font-size: 12px;
    background-color: var(--neutral-950);
    color: var(--neutral-100);
}
"""

with gr.Blocks(title="Depth+", theme=gr.themes.Citrus(font=gr.themes.GoogleFont("DM Sans"), primary_hue="yellow", secondary_hue="amber"), css=custom_css) as demo:
    demo.load(fn=load_demo)
    demo.unload(fn=unload_demo)

    with gr.Row():
        with gr.Column(scale=5):
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem(label="About"):
                    with gr.Row():
                        gr.Markdown(
                            "Depth+ is a tool by Nilor Corp for extracting depth, optical flow, and segmentation masks from a video.\n\n"
                            "Select a workflow from the tabs above and fill in the parameters.\n\n"
                            "Click 'Run Depth+' to start the workflow.",
                            line_breaks=True
                        )
                for workflow_name in workflow_definitions.keys():
                    workflow_filename = workflow_definitions[workflow_name]["filename"]

                    # make a tab for each workflow
                    with gr.TabItem(label=workflow_definitions[workflow_name]["name"]):
                        with gr.Row():
                            # main input construction
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row(equal_height=True):
                                        run_button = gr.Button("Run Depth+", variant="primary", scale=3, elem_id="run-button")
                                    with gr.Accordion("Workflow Info", open=False, elem_id="workflow-info"):
                                        info = gr.Markdown(workflow_definitions[workflow_name].get("description", ""))

                                # also make a dictionary with the components' data
                                components, component_dict = create_tab_interface(workflow_name)

                            output_type = workflow_definitions[workflow_name]["outputs"].get("type", "")
                        
        with gr.Column(scale=4):
            # TODO: is it possible to preview only an output that was produced by this workflow tab? otherwise this should probably exist outside of the workflow tab
            gr.Markdown("### Output Preview")
            with gr.Group():
                if output_type == "image":
                    output_player = gr.Image(show_label=False, interactive=False)
                else:
                    # populate the Output Preview with the latest video in the output directory 
                    latest_content = get_latest_video(OUT_DIR)
                    if (latest_content is not None):
                        output_player = gr.Video(value=latest_content, show_label=False, autoplay=True, loop=True, interactive=False)
                    else:
                        output_player = gr.Video(show_label=False, autoplay=True, loop=True, interactive=False)
                #output_filepath_component = gr.Markdown("N/A")
                
            # Modified logs section
            gr.Markdown("### Console Output")
            with gr.Accordion("View Logs", open=False):
                log_output = gr.TextArea(
                    value="", 
                    label="Logs",
                    interactive=False,
                    autoscroll=True,
                    lines=20,
                    elem_classes="logs"
                )
                
                # Create timer and connect it to log updates
                timer = gr.Timer(1)  # Updates every 1 second
                timer.tick(fn=get_logs, outputs=[log_output])

        if (components is not None) and (component_dict is not None):
            run_button.click(
                fn=run_depth_plus_wrapper(components, component_dict),
                inputs=components,
                outputs=[output_player],
                trigger_mode="multiple"
                #show_progress="full"
            )



if __name__ == "__main__":
    setup_signal_handlers()
    with redirect_stdout_stderr():
        demo.launch(
            allowed_paths=[
                OUT_DIR,
                INPUTS_DIR
            ], favicon_path="favicon.png"
        )