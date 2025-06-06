# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from scipy import ndimage

from .utils import convert_to_numpy


class SAMImageAnnotator:
    def __init__(self, cfg, device=None):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide
        except:
            import warnings
            warnings.warn("please pip install sam package, or you can refer to models/VACE-Annotators/sam/segment_anything-1.0-py3-none-any.whl")
        self.task_type = cfg.get('TASK_TYPE', 'input_box')
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.transform = ResizeLongestSide(1024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        seg_model = sam_model_registry[cfg.get('MODEL_NAME', 'vit_b')](checkpoint=cfg['PRETRAINED_MODEL']).eval().to(self.device)
        self.predictor = SamPredictor(seg_model)

    def forward(self,
                image,
                input_box=None,
                mask=None,
                task_type=None,
                return_mask=None):
        task_type = task_type if task_type is not None else self.task_type
        return_mask = return_mask if return_mask is not None else self.return_mask
        mask = convert_to_numpy(mask) if mask is not None else None

        if task_type == 'mask_point':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)   # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {
                'point_coords': point_coords,
                'point_labels': point_labels
            }
        elif task_type == 'mask_box':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {'box': bbox}
        elif task_type == 'input_box':
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            sample = {'box': input_box}
        elif task_type == 'mask':
            sample = {'mask_input': mask[None, :, :]}
        else:
            raise NotImplementedError

        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            multimask_output=False,
            **sample
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        if return_mask:
            return masks[0]
        else:
            ret_data = {
                "masks": masks,
                "scores": scores,
                "logits": logits
            }
            return ret_data