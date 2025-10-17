# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from einops import rearrange

from .utils import convert_to_numpy, resize_image, resize_image_ori

class DepthAnnotator:
    def __init__(self, cfg, device=None):
        from .midas.api import MiDaSInference
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device) 
        self.model = MiDaSInference(model_type='dpt_hybrid', model_path=pretrained_model).to(self.device)
        self.a = cfg.get('A', np.pi * 2.0)
        self.bg_th = cfg.get('BG_TH', 0.1)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        image_depth = image
        h, w, c = image.shape
        image_depth, k = resize_image(image_depth,
                                      1024 if min(h, w) > 1024 else min(h, w))
        image_depth = torch.from_numpy(image_depth).float().to(self.device)
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model(image_depth)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = depth_image[..., None].repeat(3, 2)

        depth_image = resize_image_ori(h, w, depth_image, k)
        return depth_image


class DepthVideoAnnotator(DepthAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames


class DepthV2Annotator:
    def __init__(self, cfg, device=None):
        from .depth_anything_v2.dpt import DepthAnythingV2
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                pretrained_model,
                map_location=self.device
            )
        )
        self.model.eval()

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        depth = self.model.infer_image(image)

        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return depth_image


class DepthV2VideoAnnotator(DepthV2Annotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames
