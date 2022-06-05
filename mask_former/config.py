# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    # TODO: maybe the no object weight need to be adjusted
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # for mask pool
    cfg.MODEL.MASK_FORMER.NUM_CLS_CONV = 0
    cfg.MODEL.MASK_FORMER.FUSION_WAY = 'add'
    cfg.MODEL.MASK_FORMER.SEGMENTS_MASK_THRESHOLD = 0.5
    # MASK_POOL_FROM in ["x", "mask_features"]
    cfg.MODEL.MASK_FORMER.MASK_POOL_FROM = "mask_features"
    cfg.MODEL.MASK_FORMER.CLS_DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.CLS_PRE_NORM = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.MASK_FORMER.GZERO_CALIBRATE = -1.0
    cfg.MODEL.MASK_FORMER.ENSEMBLING = False
    cfg.MODEL.MASK_FORMER.ENSEMBLING_ALL_CLS = False
    cfg.MODEL.MASK_FORMER.GZERO_CALIBRATE_BEFORE = -1.0

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    # gzero calibrate
    cfg.MODEL.SEM_SEG_HEAD.GZERO_CALIBRATE = -1.0

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES = "datasets/coco/coco_stuff/split/seen_indexes.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES = "datasets/coco/coco_stuff/split/unseen_indexes.json"


    # cfg.MODEL.MASK_FORMER.TEST.CLIP_CLASSIFICATION = False
    cfg.MODEL.SEM_SEG_HEAD.CLIP_CLASSIFICATION = False
    cfg.MODEL.SEM_SEG_HEAD.DENSECLIP = False
    cfg.MODEL.SEM_SEG_HEAD.MASKATTENTIONPOOL = False
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/32"
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_IMG = "ViT-B/32"

    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"

    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask
    cfg.MODEL.CLIP_CLS_STYLE = "cropmask"

    # cfg.MODEL.MASK_FORMER.NUM_FEATURE_LEVELS = 4
    # cfg.MODEL.MASK_FORMER.DEC_N_POINTS = 4
    # cfg.MODEL.MASK_FORMER.ENC_N_POINTS = 4
    # cfg.MODEL.MASK_FORMER.TWO_STAGE = False
    cfg.MODEL.MASK_FORMER.CLUSTER_QUERIES = False
    cfg.MODEL.MASK_FORMER.USE_SEMANTIC_QUERY = False
    cfg.MODEL.MASK_FORMER.GRAD_SEMANTIC_QUERY = False
    cfg.MODEL.MASK_FORMER.SEMANTIC_QUERY_MULTIPLIER = 0.1
    cfg.MODEL.MASK_FORMER.INIT_QUERY_BY_SEMANTIC = False
    cfg.MODEL.MASK_FORMER.INIT_QUERY_BY_SEMANTIC_AS_ZERO = False
    cfg.MODEL.SEM_SEG_HEAD.WORDVEC = False
    cfg.MODEL.SEM_SEG_HEAD.TEMPERATURE = 0.01
