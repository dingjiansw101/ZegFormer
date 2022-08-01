import os
# import cv2
from PIL import Image
import tqdm
import numpy as np
from pathlib import Path

all_ids_set = set()
# annotation_dir = Path(r'/home/tx1/code/MaskFormer/datasets/coco/coco_stuff/annotations/train2017')
annotation_dir = Path(r'/home/dj/code/MaskFormer/datasets/VOCZERO/annotations_detectron2/train_seen')

average_num_classes = 0
count = 0
for file in tqdm.tqdm(list(annotation_dir.iterdir())):
    img = np.asarray(Image.open(file))
    assert img.dtype == np.uint8
    ids = np.unique(img)
    cur_num_classes = len(ids)
    average_num_classes = average_num_classes + cur_num_classes
    count = count + 1
    # import pdb; pdb.set_trace()
    # all_ids_set.add(tuple(ids.tolist()))
    all_ids_set = all_ids_set.union(set(ids.tolist()))
    # import pdb; pdb.set_trace()
    # if len(all_ids_set) == 182:
    #     import pdb; pdb.set_trace()
    # if (len(all_ids_set) > 0) and (max(all_ids_set) == 255):
    #     import pdb; pdb.set_trace()
    # if (len(all_ids_set) > 0) and (min(all_ids_set) == 0):
    #     import pdb; pdb.set_trace()
    # try:
    #     print('min id: ', min(all_ids_set))
    #     print('max id: ', max(all_ids_set))
    # except:
    #     pass
    print('len all ids set: ', len(all_ids_set))
average_num_classes = average_num_classes / count
print('average num classes: ', average_num_classes)
print('min id: ', min(all_ids_set))
print('max id: ', max(all_ids_set))
# import pdb; pdb.set_trace()