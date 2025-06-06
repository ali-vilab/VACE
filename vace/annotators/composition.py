# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

class CompositionAnnotator:
    def __init__(self, cfg, device=None):
        self.process_types = ["repaint", "extension", "control"]
        self.process_map = {
            "repaint": "repaint",
            "extension": "extension",
            "control": "control",
            "inpainting": "repaint",
            "outpainting": "repaint",
            "frameref": "extension",
            "clipref": "extension",
            "depth": "control",
            "flow": "control",
            "gray": "control",
            "pose": "control",
            "scribble": "control",
            "layout": "control"
        }

    def forward(self, process_type_1, process_type_2, frames_1, frames_2, masks_1, masks_2):
        total_frames = min(len(frames_1), len(frames_2), len(masks_1), len(masks_2))
        combine_type = (self.process_map[process_type_1], self.process_map[process_type_2])
        if combine_type in [("extension", "repaint"), ("extension", "control"), ("extension", "extension")]:
            output_video = [frames_2[i] * masks_1[i] + frames_1[i] * (1 - masks_1[i]) for i in range(total_frames)]
            output_mask = [masks_1[i] * masks_2[i] * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "extension"), ("control", "extension"), ("repaint", "repaint")]:
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "control"), ("control", "repaint")]:
            if combine_type in [("control", "repaint")]:
                frames_1, frames_2, masks_1, masks_2 = frames_2, frames_1, masks_2, masks_1
            output_video = [frames_1[i] * (1 - masks_1[i]) + frames_2[i] * masks_1[i] for i in range(total_frames)]
            output_mask = [masks_1[i] * 255 for i in range(total_frames)]
        elif combine_type in [("control", "control")]:  # apply masks_2
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        else:
            raise Exception("Unknown combine type")
        return output_video, output_mask


class ReferenceAnythingAnnotator:
    def __init__(self, cfg, device=None):
        from .subject import SubjectAnnotator
        self.sbjref_ins = SubjectAnnotator(cfg['SUBJECT'] if 'SUBJECT' in cfg else cfg)
        self.key_map = {
            "image": "images",
            "mask": "masks"
        }
    def forward(self, images, mode=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        for image in images:
            ret_one_data = self.sbjref_ins.forward(image=image, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
            if isinstance(ret_one_data, dict):
                for key, val in ret_one_data.items():
                    if key in self.key_map:
                        new_key = self.key_map[key]
                    else:
                        continue
                    if new_key in ret_data:
                        ret_data[new_key].append(val)
                    else:
                        ret_data[new_key] = [val]
            else:
                if 'images' in ret_data:
                    ret_data['images'].append(ret_data)
                else:
                    ret_data['images'] = [ret_data]
        return ret_data


class AnimateAnythingAnnotator:
    def __init__(self, cfg, device=None):
        from .pose import PoseBodyFaceVideoAnnotator
        self.pose_ins = PoseBodyFaceVideoAnnotator(cfg['POSE'])
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, frames=None, images=None, mode=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        ret_pose_data = self.pose_ins.forward(frames=frames)
        ret_data.update({"frames": ret_pose_data})

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class SwapAnythingAnnotator:
    def __init__(self, cfg, device=None):
        from .inpainting import InpaintingVideoAnnotator
        self.inp_ins = InpaintingVideoAnnotator(cfg['INPAINTING'])
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, video=None, frames=None, images=None, mode=None, mask=None, bbox=None, label=None, caption=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        mode = mode.split(',') if ',' in mode else [mode, mode]

        ret_inp_data = self.inp_ins.forward(video=video, frames=frames, mode=mode[0], mask=mask, bbox=bbox, label=label, caption=caption, mask_cfg=mask_cfg)
        ret_data.update(ret_inp_data)

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class ExpandAnythingAnnotator:
    def __init__(self, cfg, device=None):
        from .outpainting import OutpaintingAnnotator
        from .frameref import FrameRefExpandAnnotator
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])
        self.frameref_ins = FrameRefExpandAnnotator(cfg['FRAMEREF'])
        self.outpainting_ins = OutpaintingAnnotator(cfg['OUTPAINTING'])

    def forward(self, images=None, mode=None, return_mask=None, mask_cfg=None, direction=None, expand_ratio=None, expand_num=None):
        ret_data = {}
        expand_image, reference_image= images[0], images[1:]
        mode = mode.split(',') if ',' in mode else ['firstframe', mode]

        outpainting_data = self.outpainting_ins.forward(expand_image,expand_ratio=expand_ratio, direction=direction)
        outpainting_image, outpainting_mask = outpainting_data['image'], outpainting_data['mask']

        frameref_data = self.frameref_ins.forward(outpainting_image,  mode=mode[0], expand_num=expand_num)
        frames, masks = frameref_data['frames'], frameref_data['masks']
        masks[0] = outpainting_mask
        ret_data.update({"frames": frames, "masks": masks})

        ret_ref_data = self.ref_ins.forward(images=reference_image, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class MoveAnythingAnnotator:
    def __init__(self, cfg, device=None):
        from .layout import LayoutBboxAnnotator
        self.layout_bbox_ins = LayoutBboxAnnotator(cfg['LAYOUTBBOX'])

    def forward(self, image=None, bbox=None, label=None, expand_num=None):
        frame_size = image.shape[:2]   # [H, W]
        ret_layout_data = self.layout_bbox_ins.forward(bbox, frame_size=frame_size, num_frames=expand_num, label=label)

        out_frames = [image] + ret_layout_data
        out_mask = [np.zeros(frame_size, dtype=np.uint8)] + [np.ones(frame_size, dtype=np.uint8) * 255] * len(ret_layout_data)

        ret_data = {
            "frames": out_frames,
            "masks": out_mask
        }
        return ret_data