# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch

from .utils import convert_to_numpy


class FaceAnnotator:
    def __init__(self, cfg, device=None):
        from insightface.app import FaceAnalysis
        self.return_raw = cfg.get('RETURN_RAW', True)
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.return_dict = cfg.get('RETURN_DICT', False)
        self.multi_face = cfg.get('MULTI_FACE', True)
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        self.device_id = self.device.index if self.device.type == 'cuda' else None
        ctx_id = self.device_id if self.device_id is not None else 0
        self.model = FaceAnalysis(name=cfg.MODEL_NAME, root=pretrained_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def forward(self, image=None, return_mask=None, return_dict=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_dict = return_dict if return_dict is not None else self.return_dict
        image = convert_to_numpy(image)
        # [dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])]
        faces = self.model.get(image)
        if self.return_raw:
            return faces
        else:
            crop_face_list, mask_list = [], []
            if len(faces) > 0:
                if not self.multi_face:
                    faces = faces[:1]
                for face in faces:
                    x_min, y_min, x_max, y_max = face['bbox'].tolist()
                    crop_face = image[int(y_min): int(y_max) + 1, int(x_min): int(x_max) + 1]
                    crop_face_list.append(crop_face)
                    mask = np.zeros_like(image[:, :, 0])
                    mask[int(y_min): int(y_max) + 1, int(x_min): int(x_max) + 1] = 255
                    mask_list.append(mask)
                if not self.multi_face:
                    crop_face_list = crop_face_list[0]
                    mask_list = mask_list[0]
                if return_mask:
                    if return_dict:
                        return {'image': crop_face_list, 'mask': mask_list}
                    else:
                        return crop_face_list, mask_list
                else:
                    return crop_face_list
            else:
                return None
