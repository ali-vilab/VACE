# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import torch
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from .utils import convert_to_pil

class RAMAnnotator:
    def __init__(self, cfg, device=None):
        try:
            from ram.models import ram_plus, ram, tag2text
            from ram import inference_ram
        except:
            import warnings
            warnings.warn("please pip install ram package, or you can refer to models/VACE-Annotators/ram/ram-0.0.1-py3-none-any.whl")

        delete_tag_index = []
        image_size = cfg.get('IMAGE_SIZE', 384)
        ram_tokenizer_path = cfg['TOKENIZER_PATH']
        ram_checkpoint_path = cfg['PRETRAINED_MODEL']
        ram_type = cfg.get('RAM_TYPE', 'swin_l')
        self.return_lang = cfg.get('RETURN_LANG', ['en'])  # ['en', 'zh']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else (torch.device(device) if isinstance(device, str) else device)
        self.model = ram_plus(pretrained=ram_checkpoint_path, image_size=image_size, vit=ram_type,
                              text_encoder_type=ram_tokenizer_path, delete_tag_index=delete_tag_index).eval().to(self.device)
        self.ram_transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inference_ram = inference_ram

    def forward(self, image):
        image = convert_to_pil(image)
        image_ann_trans = self.ram_transform(image).unsqueeze(0).to(self.device)
        tags_e, tags_c = self.inference_ram(image_ann_trans, self.model)
        tags_e_list = [tag.strip() for tag in tags_e.strip().split("|")]
        tags_c_list = [tag.strip() for tag in tags_c.strip().split("|")]
        if len(self.return_lang) == 1 and 'en' in self.return_lang:
            return tags_e_list
        elif len(self.return_lang) == 1 and 'zh' in self.return_lang:
            return tags_c_list
        else:
            return {
                "tags_e": tags_e_list,
                "tags_c": tags_c_list
            }
