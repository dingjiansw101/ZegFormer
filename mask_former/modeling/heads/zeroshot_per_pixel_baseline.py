# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .pixel_decoder import build_pixel_decoder
from mask_former.third_party import clip
from mask_former.third_party import imagenet_templates

import torch
import numpy as np


@SEM_SEG_HEADS_REGISTRY.register()
class ZeroshotPerPixelBaselineHead(nn.Module):

    # TODO: finish prompt ensembling
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        train_class_indexes_json: str,
        test_class_indexes_json: str,
        wordvec: bool,
        temperature: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.wordvec = wordvec
        self.temperature = temperature
        ####################################################################################
        #zeroshot head
        import json
        # use class_texts in train_forward, and self.test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        self.clip_pretrained = clip_pretrained
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text = clip.tokenize(self.class_texts).to(device)
        self.text_test = clip.tokenize(self.test_class_texts).to(device)
        self.prompt_ensemble_type = prompt_ensemble_type

        if self.wordvec:
            import pickle
            with open(train_class_indexes_json, 'r') as f_in:
                train_class_indexes = json.load(f_in)
            with open(test_class_indexes_json, 'r') as f_in:
                test_class_indexes = json.load(f_in)
            class_emb = np.concatenate([pickle.load(open('datasets/coco/coco_stuff/word_vectors/fasttext.pkl', "rb")),
                                        pickle.load(open('datasets/coco/coco_stuff/word_vectors/word2vec.pkl', "rb"))], axis=1)
            text_features = torch.from_numpy(class_emb[np.asarray(train_class_indexes)]).to(device)
            text_features_test = torch.from_numpy(class_emb[np.asarray(test_class_indexes)]).to(device)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features_test = text_features_test / text_features_test.norm(dim=-1, keepdim=True)
            self.text_features = self.text_features.float()
            self.text_features_test = self.text_features_test.float()
        else:
            with torch.no_grad():
                clip_model, clip_preprocess = clip.load(self.clip_pretrained, device=device, jit=False)
                # shape of text_features: [num_classes, 512]
                assert "A photo of" not in self.class_texts[0]
                if self.prompt_ensemble_type == "imagenet_select":
                    prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
                elif self.prompt_ensemble_type == "imagenet":
                    prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
                elif self.prompt_ensemble_type == "single":
                    prompt_templates = ['A photo of a {} in the scene',]
                else:
                    raise NotImplementedError

                prompt_templates_clip = imagenet_templates.IMAGENET_TEMPLATES_SELECT_CLIP

                def zeroshot_classifier(classnames, templates, clip_modelp):
                    with torch.no_grad():
                        zeroshot_weights = []
                        for classname in classnames:
                            if ', ' in classname:
                                classname_splits = classname.split(', ')
                                texts = []
                                for template in templates:
                                    for cls_split in classname_splits:
                                        texts.append(template.format(cls_split))
                            else:
                                texts = [template.format(classname) for template in templates]  # format with class
                            texts = clip.tokenize(texts).cuda()  # tokenize, shape: [48, 77]
                            class_embeddings = clip_modelp.encode_text(texts)  # embed with text encoder
                            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                            class_embedding = class_embeddings.mean(dim=0)
                            class_embedding /= class_embedding.norm()
                            zeroshot_weights.append(class_embedding)
                        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                        # shape of zeroshot_weights: [512, 156]
                    return zeroshot_weights
                ## train features
                # shape of text_features: [156, 512]
                self.text_features = zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0).float()
                self.text_features_test = zeroshot_classifier(self.test_class_texts, prompt_templates, clip_model).permute(1, 0).float()

                self.text_features_clip = zeroshot_classifier(self.class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()
                self.text_features_test_clip = zeroshot_classifier(self.test_class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()

        if self.wordvec:
            self.projection_layer = nn.Conv2d(self.pixel_decoder.mask_dim, 600, kernel_size=1, stride=1)
        else:
            self.projection_layer = nn.Conv2d(self.pixel_decoder.mask_dim, 512, kernel_size=1, stride=1)
        weight_init.c2_msra_fill(self.projection_layer)
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/temperature)]).float())
        self.logit_scale.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "train_class_indexes_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES,
            "test_class_indexes_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "prompt_ensemble_type": cfg.MODEL.PROMPT_ENSEMBLE_TYPE,
            "wordvec": cfg.MODEL.SEM_SEG_HEAD.WORDVEC,
            "temperature": cfg.MODEL.SEM_SEG_HEAD.TEMPERATURE,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        features, _ = self.pixel_decoder.forward_features(features)
        features = self.projection_layer(features) # shape of x: [16, 512, 128, 128]
        N, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(N * H * W, C)
        features = features / features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.training:
            features = logit_scale * features @ self.text_features.clone().detach().t()
            numclass, text_dim = self.text_features.shape
        else:
            features = logit_scale * features @ self.text_features_test.clone().detach().t()
            numclass, text_dim = self.text_features_test.shape
        features = features.reshape(N, H, W, numclass).permute(0, 3, 1, 2)

        return features

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses