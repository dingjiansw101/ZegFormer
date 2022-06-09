#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os

from multiprocessing import Pool

COCO_CATEGORIES_Seen = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 0, 'name': 'person', 'trainId': 0},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 1, 'name': 'bicycle', 'trainId': 1},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 2, 'name': 'car', 'trainId': 2},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 3, 'name': 'motorcycle', 'trainId': 3},
    {'color': [106, 0, 228], 'isthing': 1, 'id': 4, 'name': 'airplane', 'trainId': 4},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 5, 'name': 'bus', 'trainId': 5},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 6, 'name': 'train', 'trainId': 6},
    {'color': [0, 0, 70], 'isthing': 1, 'id': 7, 'name': 'truck', 'trainId': 7},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 8, 'name': 'boat', 'trainId': 8},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 9, 'name': 'traffic light', 'trainId': 9},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 10, 'name': 'fire hydrant', 'trainId': 10},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'stop sign', 'trainId': 11},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'parking meter', 'trainId': 12},
    {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'bench', 'trainId': 13},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'bird', 'trainId': 14},
    {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'cat', 'trainId': 15},
    {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'dog', 'trainId': 16},
    {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'horse', 'trainId': 17},
    {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'sheep', 'trainId': 18},
    {'color': [110, 76, 0], 'isthing': 1, 'id': 21, 'name': 'elephant', 'trainId': 19},
    {'color': [174, 57, 255], 'isthing': 1, 'id': 22, 'name': 'bear', 'trainId': 20},
    {'color': [199, 100, 0], 'isthing': 1, 'id': 23, 'name': 'zebra', 'trainId': 21},
    {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'backpack', 'trainId': 22},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'umbrella', 'trainId': 23},
    {'color': [209, 0, 151], 'isthing': 1, 'id': 30, 'name': 'handbag', 'trainId': 24},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 31, 'name': 'tie', 'trainId': 25},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 34, 'name': 'skis', 'trainId': 26},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 35, 'name': 'snowboard', 'trainId': 27},
    {'color': [78, 180, 255], 'isthing': 1, 'id': 36, 'name': 'sports ball', 'trainId': 28},
    {'color': [0, 228, 0], 'isthing': 1, 'id': 37, 'name': 'kite', 'trainId': 29},
    {'color': [174, 255, 243], 'isthing': 1, 'id': 38, 'name': 'baseball bat', 'trainId': 30},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 39, 'name': 'baseball glove', 'trainId': 31},
    {'color': [145, 148, 174], 'isthing': 1, 'id': 41, 'name': 'surfboard', 'trainId': 32},
    {'color': [255, 208, 186], 'isthing': 1, 'id': 42, 'name': 'tennis racket', 'trainId': 33},
    {'color': [197, 226, 255], 'isthing': 1, 'id': 43, 'name': 'bottle', 'trainId': 34},
    {'color': [171, 134, 1], 'isthing': 1, 'id': 45, 'name': 'wine glass', 'trainId': 35},
    {'color': [109, 63, 54], 'isthing': 1, 'id': 46, 'name': 'cup', 'trainId': 36},
    {'color': [207, 138, 255], 'isthing': 1, 'id': 47, 'name': 'fork', 'trainId': 37},
    {'color': [151, 0, 95], 'isthing': 1, 'id': 48, 'name': 'knife', 'trainId': 38},
    {'color': [9, 80, 61], 'isthing': 1, 'id': 49, 'name': 'spoon', 'trainId': 39},
    {'color': [84, 105, 51], 'isthing': 1, 'id': 50, 'name': 'bowl', 'trainId': 40},
    {'color': [74, 65, 105], 'isthing': 1, 'id': 51, 'name': 'banana', 'trainId': 41},
    {'color': [166, 196, 102], 'isthing': 1, 'id': 52, 'name': 'apple', 'trainId': 42},
    {'color': [208, 195, 210], 'isthing': 1, 'id': 53, 'name': 'sandwich', 'trainId': 43},
    {'color': [255, 109, 65], 'isthing': 1, 'id': 54, 'name': 'orange', 'trainId': 44},
    {'color': [0, 143, 149], 'isthing': 1, 'id': 55, 'name': 'broccoli', 'trainId': 45},
    {'color': [209, 99, 106], 'isthing': 1, 'id': 57, 'name': 'hot dog', 'trainId': 46},
    {'color': [5, 121, 0], 'isthing': 1, 'id': 58, 'name': 'pizza', 'trainId': 47},
    {'color': [227, 255, 205], 'isthing': 1, 'id': 59, 'name': 'donut', 'trainId': 48},
    {'color': [147, 186, 208], 'isthing': 1, 'id': 60, 'name': 'cake', 'trainId': 49},
    {'color': [153, 69, 1], 'isthing': 1, 'id': 61, 'name': 'chair', 'trainId': 50},
    {'color': [3, 95, 161], 'isthing': 1, 'id': 62, 'name': 'couch', 'trainId': 51},
    {'color': [163, 255, 0], 'isthing': 1, 'id': 63, 'name': 'potted plant', 'trainId': 52},
    {'color': [119, 0, 170], 'isthing': 1, 'id': 64, 'name': 'bed', 'trainId': 53},
    {'color': [0, 182, 199], 'isthing': 1, 'id': 66, 'name': 'dining table', 'trainId': 54},
    {'color': [0, 165, 120], 'isthing': 1, 'id': 69, 'name': 'toilet', 'trainId': 55},
    {'color': [183, 130, 88], 'isthing': 1, 'id': 71, 'name': 'tv', 'trainId': 56},
    {'color': [95, 32, 0], 'isthing': 1, 'id': 72, 'name': 'laptop', 'trainId': 57},
    {'color': [130, 114, 135], 'isthing': 1, 'id': 73, 'name': 'mouse', 'trainId': 58},
    {'color': [110, 129, 133], 'isthing': 1, 'id': 74, 'name': 'remote', 'trainId': 59},
    {'color': [166, 74, 118], 'isthing': 1, 'id': 75, 'name': 'keyboard', 'trainId': 60},
    {'color': [219, 142, 185], 'isthing': 1, 'id': 76, 'name': 'cell phone', 'trainId': 61},
    {'color': [79, 210, 114], 'isthing': 1, 'id': 77, 'name': 'microwave', 'trainId': 62},
    {'color': [178, 90, 62], 'isthing': 1, 'id': 78, 'name': 'oven', 'trainId': 63},
    {'color': [65, 70, 15], 'isthing': 1, 'id': 79, 'name': 'toaster', 'trainId': 64},
    {'color': [127, 167, 115], 'isthing': 1, 'id': 80, 'name': 'sink', 'trainId': 65},
    {'color': [59, 105, 106], 'isthing': 1, 'id': 81, 'name': 'refrigerator', 'trainId': 66},
    {'color': [142, 108, 45], 'isthing': 1, 'id': 83, 'name': 'book', 'trainId': 67},
    {'color': [196, 172, 0], 'isthing': 1, 'id': 84, 'name': 'clock', 'trainId': 68},
    {'color': [95, 54, 80], 'isthing': 1, 'id': 85, 'name': 'vase', 'trainId': 69},
    {'color': [201, 57, 1], 'isthing': 1, 'id': 87, 'name': 'teddy bear', 'trainId': 70},
    {'color': [246, 0, 122], 'isthing': 1, 'id': 88, 'name': 'hair drier', 'trainId': 71},
    {'color': [191, 162, 208], 'isthing': 1, 'id': 89, 'name': 'toothbrush', 'trainId': 72},
    {'id': 91, 'name': 'banner', 'supercategory': 'textile', 'trainId': 73},
    {'id': 92, 'name': 'blanket', 'supercategory': 'textile', 'trainId': 74},
    {'id': 93, 'name': 'branch', 'supercategory': 'plant', 'trainId': 75},
    {'id': 94, 'name': 'bridge', 'supercategory': 'building', 'trainId': 76},
    {'id': 95, 'name': 'building-other', 'supercategory': 'building', 'trainId': 77},
    {'id': 96, 'name': 'bush', 'supercategory': 'plant', 'trainId': 78},
    {'id': 97, 'name': 'cabinet', 'supercategory': 'furniture-stuff', 'trainId': 79},
    {'id': 98, 'name': 'cage', 'supercategory': 'structural', 'trainId': 80},
    {'id': 100, 'name': 'carpet', 'supercategory': 'floor', 'trainId': 81},
    {'id': 101, 'name': 'ceiling-other', 'supercategory': 'ceiling', 'trainId': 82},
    {'id': 102, 'name': 'ceiling-tile', 'supercategory': 'ceiling', 'trainId': 83},
    {'id': 103, 'name': 'cloth', 'supercategory': 'textile', 'trainId': 84},
    {'id': 104, 'name': 'clothes', 'supercategory': 'textile', 'trainId': 85},
    {'id': 106, 'name': 'counter', 'supercategory': 'furniture-stuff', 'trainId': 86},
    {'id': 107, 'name': 'cupboard', 'supercategory': 'furniture-stuff', 'trainId': 87},
    {'id': 108, 'name': 'curtain', 'supercategory': 'textile', 'trainId': 88},
    {'id': 109, 'name': 'desk-stuff', 'supercategory': 'furniture-stuff', 'trainId': 89},
    {'id': 110, 'name': 'dirt', 'supercategory': 'ground', 'trainId': 90},
    {'id': 111, 'name': 'door-stuff', 'supercategory': 'furniture-stuff', 'trainId': 91},
    {'id': 112, 'name': 'fence', 'supercategory': 'structural', 'trainId': 92},
    {'id': 113, 'name': 'floor-marble', 'supercategory': 'floor', 'trainId': 93},
    {'id': 114, 'name': 'floor-other', 'supercategory': 'floor', 'trainId': 94},
    {'id': 115, 'name': 'floor-stone', 'supercategory': 'floor', 'trainId': 95},
    {'id': 116, 'name': 'floor-tile', 'supercategory': 'floor', 'trainId': 96},
    {'id': 117, 'name': 'floor-wood', 'supercategory': 'floor', 'trainId': 97},
    {'id': 118, 'name': 'flower', 'supercategory': 'plant', 'trainId': 98},
    {'id': 119, 'name': 'fog', 'supercategory': 'water', 'trainId': 99},
    {'id': 120, 'name': 'food-other', 'supercategory': 'food-stuff', 'trainId': 100},
    {'id': 121, 'name': 'fruit', 'supercategory': 'food-stuff', 'trainId': 101},
    {'id': 122, 'name': 'furniture-other', 'supercategory': 'furniture-stuff', 'trainId': 102},
    {'id': 124, 'name': 'gravel', 'supercategory': 'ground', 'trainId': 103},
    {'id': 125, 'name': 'ground-other', 'supercategory': 'ground', 'trainId': 104},
    {'id': 126, 'name': 'hill', 'supercategory': 'solid', 'trainId': 105},
    {'id': 127, 'name': 'house', 'supercategory': 'building', 'trainId': 106},
    {'id': 128, 'name': 'leaves', 'supercategory': 'plant', 'trainId': 107},
    {'id': 129, 'name': 'light', 'supercategory': 'furniture-stuff', 'trainId': 108},
    {'id': 130, 'name': 'mat', 'supercategory': 'textile', 'trainId': 109},
    {'id': 131, 'name': 'metal', 'supercategory': 'raw-material', 'trainId': 110},
    {'id': 132, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff', 'trainId': 111},
    {'id': 133, 'name': 'moss', 'supercategory': 'plant', 'trainId': 112},
    {'id': 134, 'name': 'mountain', 'supercategory': 'solid', 'trainId': 113},
    {'id': 135, 'name': 'mud', 'supercategory': 'ground', 'trainId': 114},
    {'id': 136, 'name': 'napkin', 'supercategory': 'textile', 'trainId': 115},
    {'id': 137, 'name': 'net', 'supercategory': 'structural', 'trainId': 116},
    {'id': 138, 'name': 'paper', 'supercategory': 'raw-material', 'trainId': 117},
    {'id': 139, 'name': 'pavement', 'supercategory': 'ground', 'trainId': 118},
    {'id': 140, 'name': 'pillow', 'supercategory': 'textile', 'trainId': 119},
    {'id': 141, 'name': 'plant-other', 'supercategory': 'plant', 'trainId': 120},
    {'id': 142, 'name': 'plastic', 'supercategory': 'raw-material', 'trainId': 121},
    {'id': 143, 'name': 'platform', 'supercategory': 'ground', 'trainId': 122},
    {'id': 145, 'name': 'railing', 'supercategory': 'structural', 'trainId': 123},
    {'id': 146, 'name': 'railroad', 'supercategory': 'ground', 'trainId': 124},
    {'id': 149, 'name': 'rock', 'supercategory': 'solid', 'trainId': 125},
    {'id': 150, 'name': 'roof', 'supercategory': 'building', 'trainId': 126},
    {'id': 151, 'name': 'rug', 'supercategory': 'textile', 'trainId': 127},
    {'id': 152, 'name': 'salad', 'supercategory': 'food-stuff', 'trainId': 128},
    {'id': 153, 'name': 'sand', 'supercategory': 'ground', 'trainId': 129},
    {'id': 154, 'name': 'sea', 'supercategory': 'water', 'trainId': 130},
    {'id': 155, 'name': 'shelf', 'supercategory': 'furniture-stuff', 'trainId': 131},
    {'id': 156, 'name': 'sky-other', 'supercategory': 'sky', 'trainId': 132},
    {'id': 157, 'name': 'skyscraper', 'supercategory': 'building', 'trainId': 133},
    {'id': 158, 'name': 'snow', 'supercategory': 'ground', 'trainId': 134},
    {'id': 159, 'name': 'solid-other', 'supercategory': 'solid', 'trainId': 135},
    {'id': 160, 'name': 'stairs', 'supercategory': 'furniture-stuff', 'trainId': 136},
    {'id': 161, 'name': 'stone', 'supercategory': 'solid', 'trainId': 137},
    {'id': 162, 'name': 'straw', 'supercategory': 'plant', 'trainId': 138},
    {'id': 163, 'name': 'structural-other', 'supercategory': 'structural', 'trainId': 139},
    {'id': 164, 'name': 'table', 'supercategory': 'furniture-stuff', 'trainId': 140},
    {'id': 165, 'name': 'tent', 'supercategory': 'building', 'trainId': 141},
    {'id': 166, 'name': 'textile-other', 'supercategory': 'textile', 'trainId': 142},
    {'id': 167, 'name': 'towel', 'supercategory': 'textile', 'trainId': 143},
    {'id': 169, 'name': 'vegetable', 'supercategory': 'food-stuff', 'trainId': 144},
    {'id': 170, 'name': 'wall-brick', 'supercategory': 'wall', 'trainId': 145},
    {'id': 172, 'name': 'wall-other', 'supercategory': 'wall', 'trainId': 146},
    {'id': 173, 'name': 'wall-panel', 'supercategory': 'wall', 'trainId': 147},
    {'id': 174, 'name': 'wall-stone', 'supercategory': 'wall', 'trainId': 148},
    {'id': 175, 'name': 'wall-tile', 'supercategory': 'wall', 'trainId': 149},
    {'id': 176, 'name': 'wall-wood', 'supercategory': 'wall', 'trainId': 150},
    {'id': 177, 'name': 'water-other', 'supercategory': 'water', 'trainId': 151},
    {'id': 178, 'name': 'waterdrops', 'supercategory': 'water', 'trainId': 152},
    {'id': 179, 'name': 'window-blind', 'supercategory': 'window', 'trainId': 153},
    {'id': 180, 'name': 'window-other', 'supercategory': 'window', 'trainId': 154},
    {'id': 181, 'name': 'wood', 'supercategory': 'solid', 'trainId': 155}]

def worker(file_tuple):
    file, output_file = file_tuple
    lab = np.asarray(Image.open(file))
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_map:
            output[lab == obj_id] = id_map[obj_id]

    Image.fromarray(output).save(output_file)

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "coco" / "coco_stuff"

    id_map = {}
    for cat in COCO_CATEGORIES_Seen:
        id_map[cat["id"]] = cat["trainId"]

    pool = Pool(32)

    for name in ["val2017_seen", "train2017"]:

        if name == "val2017_seen":
            annotation_dir = dataset_dir / "annotations" / "val2017"
        else:
            annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        file_list = []
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            file_list.append((file, output_file))

        pool.map(worker, file_list)
        print('done {}'.format(name))