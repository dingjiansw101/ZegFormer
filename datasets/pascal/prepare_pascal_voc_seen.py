import copy
import json
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os
# This code is for pascal voc, only has 10582 images for training
from shutil import copyfile
categories = [
              {"name": "aeroplane", "id": 1, "trainId": 0},
              {"name": "bicycle", "id": 2, "trainId": 1},
              {"name": "bird", "id": 3, "trainId": 2},
              {"name": "boat", "id": 4, "trainId": 3},
              {"name": "bottle", "id": 5, "trainId": 4},
              {"name": "bus", "id": 6, "trainId": 5},
              {"name": "car", "id": 7, "trainId": 6},
              {"name": "cat", "id": 8, "trainId": 7},
              {"name": "chair", "id": 9, "trainId": 8},
              {"name": "cow", "id": 10, "trainId": 9},
              {"name": "diningtable", "id": 11, "trainId": 10},
              {"name": "dog", "id": 12, "trainId": 11},
              {"name": "horse", "id": 13, "trainId": 12},
              {"name": "motorbike", "id": 14, "trainId": 13},
              {"name": "person", "id": 15, "trainId": 14},
              {"name": "potted plant", "id": 16, "trainId": 15},
              {"name": "sheep", "id": 17, "trainId": 16},
              {"name": "sofa", "id": 18, "trainId": 17},
              {"name": "train", "id": 19, "trainId": 18},
              {"name": "tvmonitor", "id": 20, "trainId": 19}]

categories_seen = copy.deepcopy(categories[:15])

categories_unseen = copy.deepcopy(categories[15:])
for index, item in enumerate(categories_unseen):
    item["trainId"] = index

if __name__ == '__main__':
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCZERO"

    id_map = {}
    for cat in categories_seen:
        id_map[cat["id"]] = cat["trainId"]

    # read train_list from the npy file of spnet
    train_list = np.load(r'datasets/voc12/split/train_list.npy')
    train_list_basename = [os.path.splitext(os.path.basename(item))[0] for item in train_list]

    val_list = np.load(r'datasets/voc12/split/test_list.npy')
    val_list_basename = [os.path.splitext(os.path.basename(item))[0] for item in val_list]

    for name in ["val", "train"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / f"{name}_seen"
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            basename = os.path.splitext(file.name)[0]
            if name == "train":
                if basename not in train_list_basename:
                    continue

            if name == "val":
                if basename not in val_list_basename:
                    continue

            output_file = output_dir / file.name
            # convert(file, output_file)
            lab = np.asarray(Image.open(file))
            assert lab.dtype == np.uint8
            # img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1

            output = np.zeros_like(lab, dtype=np.uint8) + 255
            for obj_id in np.unique(lab):
                if obj_id in id_map:
                    output[lab == obj_id] = id_map[obj_id]

            Image.fromarray(output).save(output_file)