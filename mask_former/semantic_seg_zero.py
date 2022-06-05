# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Tuple
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling import SemanticSegmentor
from detectron2.modeling.backbone import Backbone
from detectron2.data import MetadataCatalog
# from ..backbone import build_backbone
# from ..postprocessing import sem_seg_postprocess
# from .build import META_ARCH_REGISTRY

# __all__ = ["SemanticSegmentor", "SEM_SEG_HEADS_REGISTRY", "SemSegFPNHead", "build_sem_seg_head"]
#
#
# SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
# SEM_SEG_HEADS_REGISTRY.__doc__ = """
# Registry for semantic segmentation heads, which make semantic segmentation predictions
# from feature maps.
# """


@META_ARCH_REGISTRY.register()
class SemanticSegmentorGzero(SemanticSegmentor):
    """
    Main class for semantic segmentation architectures.
    """
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            gzero_calibrate: float,
            metadata,
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.gzero_calibrate = gzero_calibrate
        self.metadata = metadata
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "gzero_calibrate": cfg.MODEL.SEM_SEG_HEAD.GZERO_CALIBRATE,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0])
        }
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)
        # Note: results are logits, instead of prob
        if self.training:
            return losses
        # import pdb; pdb.set_trace()
        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            # softmax
            # shape of result: [171, H, W]
            r = F.softmax(result, dim=0)
            # import pdb; pdb.set_trace()
            # r = sem_seg_postprocess(result, image_size, height, width)
            r = sem_seg_postprocess(r, image_size, height, width)
            # import ipdb; ipdb.set_trace()

            # gzero calibrate
            if self.gzero_calibrate > 0:
                # seen_classnames = self.sem_seg_head.class_texts
                # num_seen_classnames = len(seen_classnames)
                # r[:num_seen_classnames, :, :] = r[:num_seen_classnames, :, :] - self.gzero_calibrate
                val_extra_classes = self.metadata.val_extra_classes
                seen_indexes = []
                for cls in self.metadata.stuff_classes:
                    if cls not in val_extra_classes:
                        seen_indexes.append(self.metadata.stuff_classes.index(cls))
                r[seen_indexes, :, :] = r[seen_indexes, :, :] - self.gzero_calibrate
            processed_results.append({"sem_seg": r})
        # logits are enough for per-pixel semantic segmentation inference. so the initial code do not transform it to prob

        # import pdb; pdb.set_trace()
        return processed_results