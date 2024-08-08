import requests
from PIL import Image
from io import BytesIO
import os
import json
from datetime import datetime
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

load_dotenv()

with open("config.json") as f:
    config = json.load(f)


COMFY_ROOT = os.path.normpath(config["COMFY_ROOT"])
IN_DIR = os.path.join(COMFY_ROOT, "input", "WorkFlower")


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c.isspace())


def resize_image(image, base_width):
    w_percent = base_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.LANCZOS)
    return image


def download_image(i, image_url, directory, max_images, num_images):
    if max_images is not None and i >= max_images:
        print(f"Reached the limit of {max_images} images. Stopping download.")
        return

    print(f"Downloading image {i+1} of {num_images}")
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image = resize_image(image, 768)
    image.save(os.path.join(directory, f"image_{i}.png"))
    print(f"Saved image {i+1} to {directory}")


def resolve_online_collection(collection_name, max_images=None, shuffle=False):
    try:
        apiKey = os.environ["NILOR_API_KEY"]
        url = (
            f"{os.environ['NILOR_API_URI']}/collection/api-get-collection-image-urls-by-name"
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {apiKey}",
        }
        data = {"collectionName": collection_name}

        print(f"Requesting image URLs for collection: {collection_name}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        image_urls = response.json()["imageUrls"]
        print(f"Received {len(image_urls)} image URLs")

        num_images = len(image_urls)

        if max_images is not None:
            num_images = min(max_images, num_images)
            if shuffle:
                image_urls = random.sample(image_urls, min(max_images, len(image_urls)))
                print(f"Randomly selected {len(image_urls)} image URLs")
            else:
                image_urls = image_urls[:max_images]
                print(f"Selected first {len(image_urls)} image URLs")

        sanitized_name = sanitize_name(collection_name)
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for images: {directory}")

        # print(image_urls)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    download_image, i, url, directory, max_images, num_images
                )
                for i, url in enumerate(image_urls)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

        print(f"Finished downloading images for collection: {collection_name}")
        return directory

    except Exception as e:
        print(f"Failed to resolve online collection: {e}")
        return None


def process_files(files, destination, input_type, singular=False, reorganising=False, source=None):
    print(f"\nProcessing files: {'Reorganising' if reorganising else 'Organising'}")
    if not singular:
        if not reorganising:
            # get each filename of the uploaded items,
            filenames = [file[0] for file in files]
        else:
            filenames = files
        
        # copy files to the directory
        for i, file in enumerate(filenames):
            print(f"Copying file {i+1} of {len(filenames)}")
            if source:
                src = os.path.join(source, os.path.basename(file))
            else:
                src = os.path.join(file)
            src = os.path.abspath(src)  # Convert to absolute path
            print(f"Source: {src}")
            ext = os.path.splitext(file)[1]
            print(f"Extension: {ext}")
            index_as_str = format(i, "04") # zero pad
            new_filename = f"{input_type}_{index_as_str}{ext}"
            print(new_filename)
            dst = os.path.join(destination, f"{new_filename}")
            shutil.copy(src, dst)
            print(f"Copied file {i+1} to {destination}")
    else:
        print("We are using a File input")
        print(f"{files}")
        file = files if isinstance(files, str) else files[0]

        print(f"File: {file}")
        src = os.path.join(file)
        src = os.path.abspath(src)  # Convert to absolute path
        print(f"Source: {src}")
        ext = os.path.splitext(file)[1]

        old_filename = os.path.basename(file)
        # new_filename = f"{input_type}{ext}" # could rename the file to the input type
        new_filename = old_filename

        print(new_filename)
        dst = os.path.join(destination, f"{new_filename}")
        shutil.copy(src, dst)
        print(f"Copied file to {destination}")
        # when a video is uploaded, we need to append the filename to the destination
        if input_type == "video":
            return os.path.join(destination, new_filename)
        else:
            return destination



def organise_local_files(dir, input_type, max_images=None, shuffle=False, reorganising=False):
    try:
        print("\nOrganising local files")
        if not os.path.exists(dir):
            print(f"Directory does not exist: {dir}")
            return None

        print(f"Sorting files in directory: {dir}")
        files = sorted(os.listdir(dir))

        print(f"Sorted files: {files}")

        if shuffle:
            random.shuffle(files)
            print(f"Shuffled files in directory {dir}")

        sanitized_name = sanitize_name(os.path.basename(dir))
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, sanitized_name, current_datetime)
        os.makedirs(directory, exist_ok=True)

        print(f"Created directory for {input_type}: {directory}")
        if input_type == "images":
            process_files(files, directory, input_type, singular=False, reorganising=reorganising, source=dir)
        else: # means we are using a File input with exactly one input
            process_files(files, directory, input_type, singular=True, reorganising=reorganising, source=dir)

        print(f"Finished copying {input_type} from directory: {dir}")
        return directory

    except Exception as e:
        print(f"Failed to reorganise local files: {e}")
        return None
    



def copy_uploaded_files_to_local_dir(files, input_type, max_files=None, shuffle=False):
    """
    Args:
        files: list of uploaded files
    """
    try:
        print("\nCopying uploaded files to local directory")

        # copy the files to a directory in the comfyui input folder
        # create a new directory
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = os.path.join(IN_DIR, "uploaded", current_datetime)
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory for {input_type}: {directory}")

        if input_type == "images":
            process_files(files, directory, input_type, singular=False, reorganising=False)
            return organise_local_files(directory, input_type, max_files, shuffle, reorganising=True)
        else: # means we are using a File input with exactly one input
            return process_files(files, directory, input_type, singular=True, reorganising=False)

        # perform reorganisation on local dir
        # and supply the directory path 


    except Exception as e:
        print(f"Failed to copy local files: {e}")
        return None

