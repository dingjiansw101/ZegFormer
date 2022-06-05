# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

# from ..transformer.transformer_predictor import TransformerPredictor
from ..transformer.transformer_zeroshot_predictor import TransformerZeroshotPredictor
from .pixel_decoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class ZegFormerHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        # feature_strides = [v.stride for k, v in input_shape]
        # feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # import pdb; pdb.set_trace()
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            # TODO: check if other places related to TransformerPredictor need modification
            # "transformer_predictor": TransformerPredictor(
            "transformer_predictor": TransformerZeroshotPredictor(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, images_tensor=None, ori_sizes=None):

        return self.layers(features, images_tensor, ori_sizes)

    def layers(self, features, images_tensor=None, ori_sizes=None):
        mask_features, transformer_encoder_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features, images_tensor, ori_sizes)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features, images_tensor, ori_sizes)
        return predictions
