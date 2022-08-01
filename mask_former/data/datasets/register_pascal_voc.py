import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy

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

def _get_pascal_voc_seen_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in categories_seen]
    assert len(stuff_ids) == 15, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in categories_seen]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_pascal_voc_seen(root):
    root = os.path.join(root, "VOCZERO")
    meta = _get_pascal_voc_seen_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2/train_seen"),
        ("test", "images/val", "annotations_detectron2/val_seen"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"pascal_voc_{name}_seen_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

def register_pascal_voc_seenv2(root):
    root = os.path.join(root, "VOCZERO")
    meta = _get_pascal_voc_seen_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train_seen", "annotations_detectron2_v2/train_seen"),
        ("test", "images_detectron2/val_seen", "annotations_detectron2_v2/val_seen"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"pascal_voc_{name}_seen_sem_segv2"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        # import ipdb; ipdb.set_trace()
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


def _get_pascal_voc_val_unseen_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in categories_unseen]
    assert len(stuff_ids) == 5, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in categories_unseen]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_coco_stuff_val_unseen(root):
    root = os.path.join(root, "VOCZERO")
    meta = _get_pascal_voc_val_unseen_meta()

    name = 'val_unseen'
    image_dirname = "images/val"
    sem_seg_dirname = "annotations_detectron2/val_unseen"
    image_dir = os.path.join(root, image_dirname)
    gt_dir = os.path.join(root, sem_seg_dirname)
    name = f"pascal_voc_{name}_sem_seg"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
    )
    val_extra_classes = [k["name"] for k in categories_unseen]
    MetadataCatalog.get(name).set(
        val_extra_classes=val_extra_classes,
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )

def _get_pascal_voc_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in categories]
    assert len(stuff_ids) == 20, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in categories]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_voc_stuff_val_all(root):
    root = os.path.join(root, "VOCZERO")
    meta = _get_pascal_voc_stuff_meta()
    name = 'val_all'
    image_dirname = "images/val"
    sem_seg_dirname = "annotations_detectron2/val_all"
    image_dir = os.path.join(root, image_dirname)
    gt_dir = os.path.join(root, sem_seg_dirname)
    name = f"pascal_voc_{name}_sem_seg"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
    )

    val_extra_classes = [k["name"] for k in categories_unseen]
    MetadataCatalog.get(name).set(
        val_extra_classes=val_extra_classes,
        image_root=image_dir,
        sem_seg_root=gt_dir,
        # evaluator_type="sem_seg",
        evaluator_type="sem_seg_gzero",
        ignore_label=255,
        **meta,
    )
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_voc_seen(_root)
register_coco_stuff_val_unseen(_root)
register_voc_stuff_val_all(_root)
register_pascal_voc_seenv2(_root)
