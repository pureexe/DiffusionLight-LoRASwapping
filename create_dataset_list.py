import os
import json
from tqdm.auto import tqdm

INPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images"

def main():
    scenes = sorted(os.listdir(INPUT_DIR))
    index_list = []
    for scene in tqdm(scenes):
        scene_path = os.path.join(INPUT_DIR, scene)
        if not os.path.isdir(scene_path):
            continue
        images = sorted(os.listdir(scene_path))
        for image in images:
            if not image.endswith('.jpg'):
                continue
            index_list.append( f"{scene}/{image}")
    
    with open("dataset_list.json", 'w') as f:
        json.dump(index_list, f, indent=4)

if __name__ == "__main__":
    main()