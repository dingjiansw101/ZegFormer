# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, transforms
from mask_former.third_party import imagenet_templates

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        metadata_val_all,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        gzero_calibrate: float,
        clip_classification: bool,
        ensembling: bool,
        ensembling_all_cls: bool,
        train_class_json: str,
        test_class_json: str,
        clip_cls_style: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        # import ipdb; ipdb.set_trace()
        self.sem_seg_head = sem_seg_head

        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.panoptic_on = panoptic_on
        self.clip_classification = clip_classification
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        self.metadata_val_all = metadata_val_all
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        if self.clip_classification:
            self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        self.gzero_calibrate = gzero_calibrate
        self.ensembling = ensembling
        self.ensembling_all_cls = ensembling_all_cls
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.clip_cls_style = clip_cls_style
        assert clip_cls_style in ["crop", "mask", "cropmask"]

        if hasattr(self.metadata, "val_extra_classes"):
            val_extra_classes = self.metadata.val_extra_classes
        else:
            val_extra_classes = []
        seen_indexes = []
        unseen_indexes = []
        for cls in self.metadata.stuff_classes:
            if cls not in val_extra_classes:
                seen_indexes.append(self.metadata.stuff_classes.index(cls))
            else:
                unseen_indexes.append(self.metadata.stuff_classes.index(cls))
        self.seen_indexes = seen_indexes
        self.unseen_indexes = unseen_indexes

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        metadata_val_all = MetadataCatalog.get(cfg.DATASETS.VAL_ALL[0])

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": metadata,
            "metadata_val_all": metadata_val_all,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "gzero_calibrate": cfg.MODEL.MASK_FORMER.GZERO_CALIBRATE,
            "clip_classification": cfg.MODEL.SEM_SEG_HEAD.CLIP_CLASSIFICATION,
            "ensembling": cfg.MODEL.MASK_FORMER.ENSEMBLING,
            "ensembling_all_cls": cfg.MODEL.MASK_FORMER.ENSEMBLING_ALL_CLS,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "clip_cls_style": cfg.MODEL.CLIP_CLS_STYLE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.clip_classification:
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # note, after from_tensors, the images are padded, so the shape of images and batched_inputs[0]["image"] are different
        # TODO: check the add_mask operation
        # add_mask = True
        add_mask = False
        if add_mask:
            if self.training:
                ori_sizes = [x["ori_size"] for x in batched_inputs]
            else:
                ori_sizes = [(x.shape[1], x.shape[2]) for x in images]
        else:
            ori_sizes = None

        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        outputs = self.sem_seg_head(features, None, ori_sizes)
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if self.clip_classification:
                ##########################
                mask_pred_results_224 = F.interpolate(mask_pred_results,
                    size=(224, 224), mode="bilinear", align_corners=False,)
                images_tensor = F.interpolate(clip_images.tensor,
                                              size=(224, 224), mode='bilinear', align_corners=False,)
                mask_pred_results_224 = mask_pred_results_224.sigmoid() > 0.5

                mask_pred_results_224 = mask_pred_results_224.unsqueeze(2)

                masked_image_tensors = (images_tensor.unsqueeze(1) * mask_pred_results_224)
                cropp_masked_image = True
                # vis_cropped_image = True
                if cropp_masked_image:
                    # import ipdb; ipdb.set_trace()
                    mask_pred_results_ori = mask_pred_results
                    mask_pred_results_ori = mask_pred_results_ori.sigmoid() > 0.5
                    mask_pred_results_ori = mask_pred_results_ori.unsqueeze(2)
                    masked_image_tensors_ori = (clip_images.tensor.unsqueeze(1) * mask_pred_results_ori)
                    # TODO: repeat the clip_images.tensor to get the non-masked images for later crop.
                    ori_bs, ori_num_queries, ori_c, ori_h, ori_w = masked_image_tensors_ori.shape
                    # if vis_cropped_image:
                    clip_images_repeat = clip_images.tensor.unsqueeze(1).repeat(1, ori_num_queries, 1, 1, 1)
                    clip_images_repeat = clip_images_repeat.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)

                    masked_image_tensors_ori = masked_image_tensors_ori.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)
                    import torchvision
                    import numpy as np
                    # binary_mask_preds: [1, 100, 512, 704]
                    binary_mask_preds = mask_pred_results.sigmoid() > 0.5
                    binary_bs, binary_num_queries, binary_H, binary_W = binary_mask_preds.shape
                    # assert binary_bs == 1
                    binary_mask_preds = binary_mask_preds.reshape(binary_bs * binary_num_queries,
                                                                  binary_H, binary_W)
                    sum_y = torch.sum(binary_mask_preds, dim=1)
                    cumsum_x = torch.cumsum(sum_y, dim=1).float()
                    xmaxs = torch.argmax(cumsum_x, dim=1, keepdim=True) # shape: [100, 1]
                    cumsum_x[cumsum_x==0] = np.inf
                    xmins = torch.argmin(cumsum_x, dim=1, keepdim=True)
                    sum_x = torch.sum(binary_mask_preds, dim=2)
                    cumsum_y = torch.cumsum(sum_x, dim=1).float()
                    ymaxs = torch.argmax(cumsum_y, dim=1, keepdim=True)
                    cumsum_y[cumsum_y==0] = np.inf
                    ymins = torch.argmin(cumsum_y, dim=1, keepdim=True)
                    areas = (ymaxs - ymins) * (xmaxs - xmins)
                    ymaxs[areas == 0] = images.tensor.shape[-2]
                    ymins[areas == 0] = 0
                    xmaxs[areas == 0] = images.tensor.shape[-1]
                    xmins[areas == 0] = 0
                    boxes = torch.cat((xmins, ymins, xmaxs, ymaxs), 1)  # [binary_bs * binary_num_queries, 4]
                    # boxes = boxes.reshape(binary_bs, binary_num_queries, 4)
                    # TODO: crop images by boxes in the original image size
                    # boxes_list = [boxes[i].reshape(1, -1) for i in range(boxes.shape[0])]
                    boxes_list = []
                    for i in range(boxes.shape[0]):
                        boxes_list.append(boxes[i].reshape(1, -1).float())
                    box_masked_images = torchvision.ops.roi_align(masked_image_tensors_ori, boxes_list, 224, aligned=True)
                    box_masked_images = box_masked_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

                    # if vis_cropped_image:
                        # import ipdb; ipdb.set_trace()
                    box_images = torchvision.ops.roi_align(clip_images_repeat, boxes_list, 224, aligned=True)
                    box_images = box_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

                count = 0
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.clip_classification:
                    import numpy as np
                    masked_image_tensor = masked_image_tensors[count]
                    # if cropp_masked_image:
                    box_masked_image_tensor = box_masked_images[count]
                    # if vis_cropped_image:
                    box_image_tensor = box_images[count]
                    # boxs = boxes_list[count]
                    count = count + 1

                    with torch.no_grad():
                        if self.clip_cls_style == "cropmask":
                            clip_input_images = box_masked_image_tensor
                        elif self.clip_cls_style == "mask":
                            clip_input_images = masked_image_tensor
                        elif self.clip_cls_style == "crop":
                            clip_input_images = box_image_tensor
                        else:
                            raise NotImplementedError

                        image_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_input_images)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        logit_scale = self.sem_seg_head.predictor.clip_model.logit_scale.exp()
                        logits_per_image = logit_scale.half() * image_features @ self.sem_seg_head.predictor.text_features_test_clip.t().half()
                        logits_per_image = logits_per_image.float()
                        logits_per_image = torch.cat((logits_per_image, mask_cls_result[:, -1].unsqueeze(1)), 1)
                        assert not (self.ensembling and self.ensembling_all_cls)
                        if self.ensembling:
                            # note that in this branch, average the seen score of clip
                            # seen_indexes, unseen_indexes = self.seen_unseen_indexes()
                            lambda_balance = 2 / 3.
                            mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
                            # shape of logits_per_image: [100, 171]
                            logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
                            # remove the influence of clip on seen classes
                            logits_per_image[:, self.seen_indexes] = logits_per_image[:, self.seen_indexes].mean(dim=1, keepdim=True)

                            mask_cls_result[:, self.seen_indexes] = torch.pow(mask_cls_result[:, self.seen_indexes], lambda_balance) \
                                                               * torch.pow(logits_per_image[:, self.seen_indexes], 1 - lambda_balance)
                            mask_cls_result[:, self.unseen_indexes] = torch.pow(mask_cls_result[:, self.unseen_indexes], 1 - lambda_balance) \
                                                               * torch.pow(logits_per_image[:, self.unseen_indexes], lambda_balance)
                        elif self.ensembling_all_cls:
                            lambda_balance = 2 / 3.
                            mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
                            logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
                            mask_cls_result = torch.pow(mask_cls_result, 1 - lambda_balance) \
                                                               * torch.pow(logits_per_image, lambda_balance)
                        else:
                            mask_cls_result = logits_per_image

                    ######################################################################################
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )

                # semantic segmentation inference
                if (self.clip_classification and self.ensembling) or (self.clip_classification and self.ensembling_all_cls):
                    r = self.semantic_inference2(mask_cls_result, mask_pred_result)
                else:
                    r = self.semantic_inference(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, height, width)
                #############################################################################
                # gzero calibrate
                if self.gzero_calibrate > 0:
                    r[self.seen_indexes, :, :] = r[self.seen_indexes, :, :] - self.gzero_calibrate
                ###########################################################################
                processed_results.append({"sem_seg": r})

            return processed_results

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            # import ipdb; ipdb.set_trace()
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg

    def semantic_inference2(self, mask_cls, mask_pred):
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg