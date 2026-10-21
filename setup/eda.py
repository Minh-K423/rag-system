import os
import json
import numpy as np
from qdrant_client.models import PointStruct

import pathlib
import uuid 


def data_availability(npy_path, json_path):
    print("="*50)
    print(f" EXPLORING DATA ")
    print("="*50)

    if os.path.exists(npy_path):
        npy_existence = True
        features = np.load(npy_path)
        print(f"\n[+] NPY file: {npy_path}")
        print(f"    -> Data type: {features.dtype}")
        print(f"    -> Matrix Shape: {features.shape}")
    else:
        npy_existence = False
        print(f"\n[-] Not found: {npy_path}")

    if os.path.exists(json_path):
        json_existence = True
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n[+] JSON file: {json_path}")
        print(f"    -> Available fields (Keys): {list(metadata.keys())}")
        
        if 'description' in metadata:
            print(f"    -> Description Preview: {metadata['description'][:100]}...")
    else:
        json_existence = False
        print(f"\n[-] Not found: {json_path}")

    if (npy_existence and json_existence):
        return True
    else:
        return False


def generate_point(npy_path, json_path):
    video_name = os.path.splitext(os.path.basename(npy_path))[0]
    metadata = {}

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    title = metadata.get("title", "No available key")
    description = metadata.get("description", "No available key")

    points_to_upload = []

    video_features = np.load(npy_path)
    num_frames = video_features.shape[0]

    for frame_index in range(num_frames):
        frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{frame_index}"))

        vector = video_features[frame_index].tolist() #Cause Qdrant do not support numpy array, we need to convert it to list

        payload = {
            "video_name": video_name,
            "frame_index": frame_index+1, # Store frame index starting from 1, because the first frame is 1, not 0 despite Qdrant using 0-based indexing
            "title": title,
            "description": description
        }

        point = PointStruct(id=frame_id, vector=vector, payload=payload)

        points_to_upload.append(point)

    return points_to_upload


def check_metadata(json_dir_path):

    json_dir = pathlib.Path(json_dir_path)
    
    for json_file in json_dir.glob("*.json"):

        if not json_file.is_file():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        title = metadata.get("title", "No available key")
        description = metadata.get("description", "No available key")

        if "quân" in title.lower() or "quân" in description.lower():
            print(f"Found 'quân': {json_file.stem} Title: {title}, Description: {description}")



def main():
    npy_dir_path = os.getenv("LINUX_NPY_DIR")
    json_dir_path = os.getenv("LINUX_JSON_DIR")

    if not os.path.exists(npy_dir_path) or not os.path.exists(json_dir_path):
        print("[-] One or both of the specified directories do not exist.")
        return
    
    ''' 
    npy_dir = pathlib.Path(npy_dir_path)
    json_dir = pathlib.Path(json_dir_path)

    for npy_file in npy_dir.glob("*.npy"):

        if not npy_file.is_file():
            continue

        json_file = json_dir / (npy_file.stem + ".json")
        if not json_file.is_file():
            print(f"[-] Corresponding JSON file not found for: {npy_file.name}")
            continue

        if data_availability(npy_file, json_file):
            points = generate_point(npy_file, json_file)
            print(f"\n[+] Generated {len(points)} points for video: {npy_file.stem}")
        else:
            print(f"[-] Data not available for video: {npy_file.stem}")
    '''
    check_metadata(json_dir_path)

        
if __name__ == "__main__":
    main()
