# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from mask_former.third_party import clip
from mask_former.third_party import imagenet_templates

import numpy as np

class TransformerZeroshotPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        train_class_json: str,
        test_class_json: str,
        train_class_indexes_json: str,
        test_class_indexes_json: str,
        clip_classification: bool,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        wordvec: bool,
        temperature: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        self.wordvec = wordvec
        ####################################################################################
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)

        self.text = clip.tokenize(self.class_texts).to(device)
        self.text_test = clip.tokenize(self.test_class_texts).to(device)

        import math
        if self.wordvec:
            self.bg_feature = nn.Parameter(torch.Tensor(1, 600))
        else:
            self.bg_feature = nn.Parameter(torch.Tensor(1, 512))
        nn.init.kaiming_uniform_(
            self.bg_feature, a=math.sqrt(5))
        self.bg_feature.requires_grad = True
        self.prompt_ensemble_type = prompt_ensemble_type
        if self.wordvec:
            self.projection_layer = nn.Linear(hidden_dim, 600)
        else:
            self.projection_layer = nn.Linear(hidden_dim, 512)

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
                    return zeroshot_weights
                ## train features
                # shape of text_features: [156, 512]
                self.text_features = zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0).float()
                self.text_features_test = zeroshot_classifier(self.test_class_texts, prompt_templates, clip_model).permute(1, 0).float()

                self.text_features_clip = zeroshot_classifier(self.class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()
                self.text_features_test_clip = zeroshot_classifier(self.test_class_texts, prompt_templates_clip, clip_model).permute(1, 0).float()

        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/temperature)]).float())
        self.logit_scale.requires_grad = False
        self.clip_classification = clip_classification
        if self.clip_classification:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess
        ####################################################################################
        self.mask_classification = mask_classification
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["train_class_indexes_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES
        ret["test_class_indexes_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES

        ret["clip_classification"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_CLASSIFICATION
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE
        ret["wordvec"] = cfg.MODEL.SEM_SEG_HEAD.WORDVEC
        ret["temperature"] = cfg.MODEL.SEM_SEG_HEAD.TEMPERATURE

        return ret

    def forward(self, x, mask_features, images_tensor=None, ori_sizes=None):
        assert images_tensor == None

        pos = self.pe_layer(x)
        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            x_cls = self.projection_layer(hs)
            # TODO: check if it is l2 norm
            x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            if self.training:
                cls_score = logit_scale * x_cls @ self.text_features.clone().detach().t()
            else:
                cls_score = logit_scale * x_cls @ self.text_features_test.clone().detach().t()

            bg_score = logit_scale * x_cls @ self.bg_feature.t()
            outputs_class = torch.cat((cls_score, bg_score), -1)
            out = {"pred_logits": outputs_class[-1]}

        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
