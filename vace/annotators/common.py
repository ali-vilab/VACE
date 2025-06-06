# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

class PlainImageAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, image):
        return image

class PlainVideoAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, frames):
        return frames

class PlainMaskAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, mask):
        return mask

class PlainMaskAugInvertAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, mask):
        return 255 - mask

class PlainMaskAugAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, mask):
        return mask

class PlainMaskVideoAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, mask):
        return mask

class PlainMaskAugVideoAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, masks):
        return masks

class PlainMaskAugInvertVideoAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, masks):
        return [255 - mask for mask in masks]

class ExpandMaskVideoAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, mask, expand_num):
        return [mask] * expand_num

class PlainPromptAnnotator:
    def __init__(self, cfg, device=None):
        pass
    def forward(self, prompt):
        return prompt