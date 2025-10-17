# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np
from .utils import convert_to_numpy


class GrayAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, image):
        image = convert_to_numpy(image)
        gray_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_map[..., None].repeat(3, axis=2)


class GrayVideoAnnotator(GrayAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames
