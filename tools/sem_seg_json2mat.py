import numpy as np
import scipy
from scipy.io import savemat
import json
from pycocotools import mask as maskUtils
from collections import defaultdict
import os
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
from PIL import Image

# def json_to_mat(filename, outfile):
#     fin = open(filename, encoding='UTF-8')
#     s = json.load(fin)
#     data = dict()
#     for k, v in s.items():
#         data[k] = v
#     savemat(outfile, data)
#     fin.close()

def sem_seg_json_to_mat(filename, outdir=None, dataset_name="coco_2017_val_all_stuff_sem_seg"):
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    assert dataset_name in ["coco_2017_val_all_stuff_sem_seg"]
    with open(filename, 'r') as f_in:
        predictions = json.load(f_in)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    imgToAnns = defaultdict(list)
    for pred in predictions:
        image_id = os.path.basename(pred["file_name"]).split(".")[0]
        imgToAnns[image_id].append(
            {"category_id": pred["category_id"], "segmentation": pred["segmentation"]}
        )
    image_ids = list(imgToAnns.keys())

    for image_id in tqdm(image_ids):
        if dataset_name == "coco_2017_val_all_stuff_sem_seg":
            gt_dir = os.path.join(_root, "coco/coco_stuff", "annotations_detectron2", "val2017_all")
            segm_gt = read_image(os.path.join(gt_dir, image_id + ".png")).copy().astype(np.int64)
            # import ipdb; ipdb.set_trace()
        # get predictions
        segm_dt = np.zeros_like(segm_gt)
        anns = imgToAnns[image_id]
        # import ipdb; ipdb.set_trace()
        for ann in anns:
            # map back category_id
            category_id = ann["category_id"]
            mask = maskUtils.decode(ann["segmentation"])
            # TODO: keep it in imind, that ther id here just represent a partition, and not the real category_id
            segm_dt[mask > 0] = category_id + 1
        # import ipdb;
        # ipdb.set_trace()
        Image.fromarray(segm_dt.astype(np.uint16)).save(os.path.join(outdir, image_id + '.tif'))

if __name__ == '__main__':
    # sem_seg_json_to_mat(r'/home/dj/code/MaskFormer/work_dirs/'
    #                     r'maskformer_R50_bs32_60k_zeroshot_gzss_eval_clipcls_vit16_coco-stuff'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'/home/dj/code/MaskFormer/work_dirs/'
    #                     r'maskformer_R50_bs32_60k_zeroshot_gzss_eval_clipcls_vit16_coco-stuff'
    #                     r'/inference/pngs')
    # sem_seg_json_to_mat(r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff'
    #                     r'/inference/images')
    # sem_seg_json_to_mat(r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_gzss_eval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_gzss_eval'
    #                     r'/inference/images')
    # sem_seg_json_to_mat(r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_vit16_coco-stuff_gzss_eval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_vit16_coco-stuff_gzss_eval'
    #                     r'/inference/images')

    # sem_seg_json_to_mat(r'work_dirs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16_zeroshot_coco-stuff_gzss_eval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16_zeroshot_coco-stuff_gzss_eval'
    #                     r'/inference/images')

    # sem_seg_json_to_mat(r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff_groupeval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff_groupeval'
    #                     r'/inference/images')

    # sem_seg_json_to_mat(r'work_dirs/maskformer_R50_bs32_60k_zeroshot_wordvec_coco-stuff_gzss_eval_group_eval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/maskformer_R50_bs32_60k_zeroshot_wordvec_coco-stuff_gzss_eval_group_eval'
    #                     r'/inference/images')

    # sem_seg_json_to_mat(r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_wordvec_coco-stuff_gzss_eval_group_eval'
    #                     r'/inference/sem_seg_predictions.json',
    #                     r'work_dirs/per_pixel_baseline_R50_bs32_60k_zeroshot_wordvec_coco-stuff_gzss_eval_group_eval'
    #                     r'/inference/images')

    sem_seg_json_to_mat(r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff_group_eval_tx1'
                        r'/inference/sem_seg_predictions.json',
                        r'work_dirs/maskformer_R50_bs32_60k_zeroshot_vit16_gzss_eval_coco-stuff_group_eval_tx1'
                        r'/inference/images')
