# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import torch
import numpy as np
import torchvision
from .utils import convert_to_numpy


class GDINOAnnotator:
    def __init__(self, cfg, device=None):
        try:
            from groundingdino.util.inference import Model, load_model, load_image, predict
        except:
            import warnings
            warnings.warn("please pip install groundingdino package, or you can refer to models/VACE-Annotators/gdino/groundingdino-0.1.0-cp310-cp310-linux_x86_64.whl")

        grounding_dino_config_path = cfg['CONFIG_PATH']
        grounding_dino_checkpoint_path = cfg['PRETRAINED_MODEL']
        grounding_dino_tokenizer_path = cfg['TOKENIZER_PATH']  # TODO
        self.box_threshold = cfg.get('BOX_THRESHOLD', 0.25)
        self.text_threshold = cfg.get('TEXT_THRESHOLD', 0.2)
        self.iou_threshold = cfg.get('IOU_THRESHOLD', 0.5)
        self.use_nms = cfg.get('USE_NMS', True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        self.model = Model(model_config_path=grounding_dino_config_path,
                           model_checkpoint_path=grounding_dino_checkpoint_path,
                           device=self.device)

    def forward(self, image, classes=None, caption=None):
        image_bgr = convert_to_numpy(image)[..., ::-1]  # bgr

        if classes is not None:
            classes = [classes] if isinstance(classes, str) else classes
            detections = self.model.predict_with_classes(
                image=image_bgr,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
        elif caption is not None:
            detections, phrases = self.model.predict_with_caption(
                image=image_bgr,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
        else:
            raise NotImplementedError()

        if self.use_nms:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.iou_threshold
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx] if detections.class_id is not None else None

        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id
        class_names = [classes[_id] for _id in class_ids] if classes is not None else phrases

        ret_data = {
            "boxes": boxes.tolist() if boxes is not None else None,
            "confidences": confidences.tolist() if confidences is not None else None,
            "class_ids": class_ids.tolist() if class_ids is not None else None,
            "class_names": class_names if class_names is not None else None,
        }
        return ret_data


class GDINORAMAnnotator:
    def __init__(self, cfg, device=None):
        from .ram import RAMAnnotator
        from .gdino import GDINOAnnotator
        self.ram_model = RAMAnnotator(cfg['RAM'], device=device)
        self.gdino_model = GDINOAnnotator(cfg['GDINO'], device=device)

    def forward(self, image):
        ram_res = self.ram_model.forward(image)
        classes = ram_res['tag_e'] if isinstance(ram_res, dict) else ram_res
        gdino_res = self.gdino_model.forward(image, classes=classes)
        return gdino_res

