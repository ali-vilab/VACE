# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import sys
import json
import os
import argparse
import datetime
import copy
import random

import cv2
import imageio
import numpy as np
import gradio as gr
import tempfile
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3]))
from vace.annotators.utils import single_rle_to_mask, read_video_frames, save_one_video, read_video_one_frame, read_video_last_frame
from vace.configs import VACE_IMAGE_PREPROCCESS_CONFIGS, VACE_IMAGE_MASK_PREPROCCESS_CONFIGS, VACE_IMAGE_MASKAUG_PREPROCCESS_CONFIGS, VACE_VIDEO_PREPROCCESS_CONFIGS, VACE_VIDEO_MASK_PREPROCCESS_CONFIGS, VACE_VIDEO_MASKAUG_PREPROCCESS_CONFIGS, VACE_COMPOSITION_PREPROCCESS_CONFIGS
import vace.annotators as annotators


def tid_maker():
    return '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())

def dict_to_markdown_table(d):
    markdown = "| Key | Value |\n"
    markdown += "| --- | ----- |\n"
    for key, value in d.items():
        markdown += f"| {key} | {value} |\n"
    return markdown


class VACEImageTag():
    def __init__(self, cfg, device=None):
        self.save_dir = os.path.join(cfg.save_dir, 'image')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.image_anno_processor = {}
        self.load_image_anno_list = ["image_plain", "image_depth", "image_gray", "image_pose", "image_scribble", "image_outpainting"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_IMAGE_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_image_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.image_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                    "anno_ins": anno_ins}

        self.mask_anno_processor = {}
        self.load_mask_anno_list = ["image_mask_plain", "image_mask_seg", "image_mask_draw", "image_mask_face"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_IMAGE_MASK_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_mask_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.mask_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                   "anno_ins": anno_ins}

        self.maskaug_anno_processor = {}
        self.load_maskaug_anno_list = ["image_maskaug_plain", "image_maskaug_invert", "image_maskaug", "image_maskaug_region_random", "image_maskaug_region_crop"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_IMAGE_MASKAUG_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_maskaug_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.maskaug_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                      "anno_ins": anno_ins}

        self.seg_type = ['maskpointtrack', 'maskbboxtrack', 'masktrack', 'salientmasktrack', 'salientbboxtrack', 'label', 'caption']
        self.seg_draw_type = ['maskpoint', 'maskbbox', 'mask']

    def create_ui_image(self, *args, **kwargs):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_image = gr.ImageMask(
                        label="input_process_image",
                        layers=False,
                        type='pil',
                        format='png',
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_image = gr.Image(
                        label="output_process_image",
                        value=None,
                        type='pil',
                        image_mode='RGB',
                        format='png',
                        interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_masked_image = gr.Image(
                        label="output_process_masked_image",
                        value=None,
                        type='pil',
                        image_mode='RGB',
                        format='png',
                        interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_mask = gr.Image(
                        label="output_process_mask",
                        value=None,
                        type='pil',
                        image_mode='L',
                        format='png',
                        interactive=False)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.image_process_type = gr.Dropdown(
                        label='Image Annotator',
                        choices=list(self.image_anno_processor.keys()),
                        value=list(self.image_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row(visible=False) as self.outpainting_setting:
                    self.outpainting_direction = gr.Dropdown(
                        multiselect=True,
                        label='Outpainting Direction',
                        choices=['left', 'right', 'up', 'down'],
                        value=['left', 'right', 'up', 'down'],
                        interactive=True)
                    self.outpainting_ratio = gr.Slider(
                        label='Outpainting Ratio',
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=0.3,
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.mask_process_type = gr.Dropdown(
                        label='Mask Annotator',
                        choices=list(self.mask_anno_processor.keys()),
                        value=list(self.mask_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row():
                    self.mask_opacity = gr.Slider(
                        label='Mask Opacity',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=1.0,
                        interactive=True)
                    self.mask_gray = gr.Checkbox(
                        label='Mask Gray',
                        value=True,
                        interactive=True)
                with gr.Row(visible=False) as self.segment_setting:
                    self.mask_type = gr.Dropdown(
                        label='Segment Type',
                        choices=self.seg_type,
                        value='maskpointtrack',
                        interactive=True)
                    self.mask_segtag = gr.Textbox(
                        label='Mask Seg Tag',
                        value='',
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.mask_aug_process_type = gr.Dropdown(
                        label='Mask Aug Annotator',
                        choices=list(self.maskaug_anno_processor.keys()),
                        value=list(self.maskaug_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row(visible=False) as self.maskaug_setting:
                    self.mask_aug_type = gr.Dropdown(
                        label='Mask Aug Type',
                        choices=['random', 'original', 'original_expand', 'hull', 'hull_expand', 'bbox', 'bbox_expand'],
                        value='original',
                        interactive=True)
                    self.mask_expand_ratio = gr.Slider(
                        label='Mask Expand Ratio',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.3,
                        interactive=True)
                    self.mask_expand_iters = gr.Slider(
                        label='Mask Expand Iters',
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.process_button = gr.Button(
                        value='[1]Sample Process',
                        elem_classes='type_row',
                        elem_id='process_button',
                        visible=True)
                with gr.Row():
                    self.save_button = gr.Button(
                        value='[2]Sample Save',
                        elem_classes='type_row',
                        elem_id='save_button',
                        visible=True)
                with gr.Row():
                    self.save_log = gr.Markdown()


    def change_process_type(self, image_process_type, mask_process_type, mask_aug_process_type):
        outpainting_setting_visible = False
        segment_setting = False
        maskaug_setting = False
        segment_choices = self.seg_type
        if image_process_type == "image_outpainting":
            outpainting_setting_visible = True
        if mask_process_type in ["image_mask_seg", "image_mask_draw"]:
            segment_setting = True
            if mask_process_type in ["image_mask_draw"]:
                segment_choices = self.seg_draw_type
        if mask_aug_process_type in ["image_maskaug", "image_maskaug_region_random", "image_maskaug_region_crop"]:
            maskaug_setting = True
        return gr.update(visible=outpainting_setting_visible), gr.update(visible=segment_setting), gr.update(choices=segment_choices, value=segment_choices[0]), gr.update(visible=maskaug_setting)

    def process_image_data(self, input_process_image, image_process_type,  outpainting_direction, outpainting_ratio, mask_process_type, mask_type, mask_segtag, mask_opacity, mask_gray, mask_aug_process_type, mask_aug_type, mask_expand_ratio, mask_expand_iters):
        image = np.array(input_process_image['background'].convert('RGB'))
        mask = np.array(input_process_image['layers'][0].split()[-1].convert('L'))
        image_shape = image.shape

        if image_process_type in ['image_outpainting']:
            ret_data = self.image_anno_processor[image_process_type]['anno_ins'].forward(image, direction=outpainting_direction, expand_ratio=outpainting_ratio)
            image, mask = ret_data['image'], ret_data['mask']
        else:
            image = self.image_anno_processor[image_process_type]['anno_ins'].forward(image)
            if image.shape != image_shape:
                image = cv2.resize(image, image_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        if mask_process_type in ["image_mask_seg"]:
            mask = mask[..., None]
            mask = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(image, mask=mask, label=mask_segtag, caption=mask_segtag, mode=mask_type)['mask']
        elif mask_process_type in ['image_mask_draw']:
            ret_data = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(mask=mask, mode=mask_type)
            mask = ret_data['mask'] if isinstance(ret_data, dict) and 'mask' in ret_data else ret_data
        elif mask_process_type in ['image_mask_face']:
            ret_data = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(image=image)
            mask = ret_data['mask'] if isinstance(ret_data, dict) and 'mask' in ret_data else ret_data
        else:
            ret_data = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(mask=mask)
            mask = ret_data['mask'] if isinstance(ret_data, dict) and 'mask' in ret_data else ret_data

        mask_cfg = {
            'mode': mask_aug_type,
            'kwargs': {
                'expand_ratio': mask_expand_ratio,
                'expand_iters': mask_expand_iters
            }
        }
        if mask_aug_process_type == 'image_maskaug':
            mask = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(np.array(mask), mask_cfg)
        elif mask_aug_process_type in ["image_maskaug_region_random", "image_maskaug_region_crop"]:
            image = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(np.array(image), np.array(mask), mask_cfg=mask_cfg)
        else:
            ret_data = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(mask=mask)
            mask = ret_data['mask'] if isinstance(ret_data, dict) and 'mask' in ret_data else ret_data

        if mask_opacity > 0:
            if mask.shape[:2] != image.shape[:2]:
                raise gr.Error(f"Mask shape {mask.shape[:2]} should be the same as image shape {image.shape[:2]} or set mask_opacity to 0.")
            if mask_gray:
                masked_image = image.copy()
                masked_image[mask == 255] = 127.5
            else:
                mask_weight = mask / 255 * mask_opacity
                masked_image = np.clip(image * (1 - mask_weight[:, :, None]), 0, 255).astype(np.uint8)
        else:
            masked_image = image
        return image, masked_image, mask

    def save_image_data(self, input_image, image, masked_image, mask):
        save_data = {
            "input_image": input_image['background'].convert('RGB') if isinstance(input_image, dict) else input_image,
            "input_image_mask": input_image['layers'][0].split()[-1].convert('L') if isinstance(input_image, dict) else None,
            "output_image": image,
            "output_masked_image": masked_image,
            "output_image_mask": mask
        }
        save_info = {}
        tid = tid_maker()
        for name, image in save_data.items():
            if image is None: continue
            save_image_dir = os.path.join(self.save_dir, tid[:8])
            if not os.path.exists(save_image_dir): os.makedirs(save_image_dir)
            save_image_path = os.path.join(save_image_dir, tid + '-' + name + '.png')
            save_info[name] = save_image_path
            image.save(save_image_path)
            gr.Info(f'Save {name} to {save_image_path}', duration=15)
        save_txt_path = os.path.join(self.save_dir, tid[:8], tid + '.txt')
        save_info['save_info'] = save_txt_path
        with open(save_txt_path, 'w') as f:
            f.write(json.dumps(save_info, ensure_ascii=False))
        return dict_to_markdown_table(save_info)


    def set_callbacks_image(self, **kwargs):
        inputs = [self.input_process_image, self.image_process_type,  self.outpainting_direction, self.outpainting_ratio, self.mask_process_type, self.mask_type, self.mask_segtag, self.mask_opacity, self.mask_gray, self.mask_aug_process_type, self.mask_aug_type, self.mask_expand_ratio, self.mask_expand_iters]
        outputs = [self.output_process_image, self.output_process_masked_image, self.output_process_mask]
        self.process_button.click(self.process_image_data,
                                  inputs=inputs,
                                  outputs=outputs)
        self.save_button.click(self.save_image_data,
                               inputs=[self.input_process_image, self.output_process_image, self.output_process_masked_image, self.output_process_mask],
                               outputs=[self.save_log])
        process_inputs = [self.image_process_type, self.mask_process_type, self.mask_aug_process_type]
        process_outputs = [self.outpainting_setting, self.segment_setting, self.mask_type, self.maskaug_setting]
        self.image_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)
        self.mask_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)
        self.mask_aug_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)


class VACEVideoTag():
    def __init__(self, cfg, device=None):
        self.save_dir = os.path.join(cfg.save_dir, 'video')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.video_anno_processor = {}
        self.load_video_anno_list = ["plain", "depth", "depthv2", "flow", "gray", "pose", "pose_body", "scribble", "outpainting", "outpainting_inner", "framerefext"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_VIDEO_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_video_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.video_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                    "anno_ins": anno_ins}

        self.mask_anno_processor = {}
        self.load_mask_anno_list = ["mask_expand", "mask_seg"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_VIDEO_MASK_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_mask_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.mask_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                   "anno_ins": anno_ins}

        self.maskaug_anno_processor = {}
        self.load_maskaug_anno_list = ["maskaug_plain", "maskaug_invert", "maskaug", "maskaug_layout"]
        for anno_name, anno_cfg in copy.deepcopy(VACE_VIDEO_MASKAUG_PREPROCCESS_CONFIGS).items():
            if anno_name not in self.load_maskaug_anno_list: continue
            class_name = anno_cfg.pop("NAME")
            input_params = anno_cfg.pop("INPUTS")
            output_params = anno_cfg.pop("OUTPUTS")
            anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
            self.maskaug_anno_processor[anno_name] = {"inputs": input_params, "outputs": output_params,
                                                      "anno_ins": anno_ins}


    def create_ui_video(self, *args, **kwargs):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                self.input_process_video = gr.Video(
                    label="input_process_video",
                    sources=['upload'],
                    interactive=True)
                self.input_process_first_image_show = gr.Image(
                    label="input_process_first_image_show",
                    format='png',
                    interactive=False)
                self.input_process_last_image_show = gr.Image(
                    label="input_process_last_image_show",
                    format='png',
                    interactive=False)
            with gr.Column(scale=2):
                self.input_process_image = gr.ImageMask(
                    label="input_process_image",
                    layers=False,
                    type='pil',
                    format='png',
                    interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_video = gr.Video(
                        label="output_process_video",
                        value=None,
                        interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_masked_video = gr.Video(
                        label="output_process_masked_video",
                        value=None,
                        interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_video_mask = gr.Video(
                        label="output_process_video_mask",
                        value=None,
                        interactive=False)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.video_process_type = gr.Dropdown(
                        label='Video Annotator',
                        choices=list(self.video_anno_processor.keys()),
                        value=list(self.video_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row(visible=False) as self.outpainting_setting:
                    self.outpainting_direction = gr.Dropdown(
                        multiselect=True,
                        label='Outpainting Direction',
                        choices=['left', 'right', 'up', 'down'],
                        value=['left', 'right', 'up', 'down'],
                        interactive=True)
                    self.outpainting_ratio = gr.Slider(
                        label='Outpainting Ratio',
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=0.3,
                        interactive=True)
                with gr.Row(visible=False) as self.frame_reference_setting:
                    self.frame_reference_mode = gr.Dropdown(
                        label='Frame Reference Mode',
                        choices=['first', 'last', 'firstlast', 'random'],
                        value='first',
                        interactive=True)
                    self.frame_reference_num = gr.Textbox(
                        label='Frame Reference Num',
                        value='1',
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.mask_process_type = gr.Dropdown(
                        label='Mask Annotator',
                        choices=list(self.mask_anno_processor.keys()),
                        value=list(self.mask_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row():
                    self.mask_opacity = gr.Slider(
                        label='Mask Opacity',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=1.0,
                        interactive=True)
                    self.mask_gray = gr.Checkbox(
                        label='Mask Gray',
                        value=True,
                        interactive=True)
                with gr.Row(visible=False) as self.segment_setting:
                    self.mask_type = gr.Dropdown(
                        label='Segment Type',
                        choices=['maskpointtrack', 'maskbboxtrack', 'masktrack', 'salientmasktrack', 'salientbboxtrack',
                                 'label', 'caption'],
                        value='maskpointtrack',
                        interactive=True)
                    self.mask_segtag = gr.Textbox(
                        label='Mask Seg Tag',
                        value='',
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.mask_aug_process_type = gr.Dropdown(
                        label='Mask Aug Annotator',
                        choices=list(self.maskaug_anno_processor.keys()),
                        value=list(self.maskaug_anno_processor.keys())[0],
                        interactive=True)
                with gr.Row(visible=False) as self.maskaug_setting:
                    self.mask_aug_type = gr.Dropdown(
                        label='Mask Aug Type',
                        choices=['random', 'original', 'original_expand', 'hull', 'hull_expand', 'bbox', 'bbox_expand'],
                        value='original',
                        interactive=True)
                    self.mask_expand_ratio = gr.Slider(
                        label='Mask Expand Ratio',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.3,
                        interactive=True)
                    self.mask_expand_iters = gr.Slider(
                        label='Mask Expand Iters',
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        interactive=True)
                    self.mask_layout_label = gr.Textbox(
                        label='Mask Layout Label',
                        value='',
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.process_button = gr.Button(
                        value='[1]Sample Process',
                        elem_classes='type_row',
                        elem_id='process_button',
                        visible=True)
                with gr.Row():
                    self.save_button = gr.Button(
                        value='[2]Sample Save',
                        elem_classes='type_row',
                        elem_id='save_button',
                        visible=True)
                with gr.Row():
                    self.save_log = gr.Markdown()

    def process_video_data(self, input_process_video, input_process_image, video_process_type, outpainting_direction, outpainting_ratio, frame_reference_mode, frame_reference_num, mask_process_type, mask_type, mask_segtag, mask_opacity, mask_gray, mask_aug_process_type, mask_aug_type, mask_expand_ratio, mask_expand_iters, mask_layout_label):
        video_frames, fps, width, height, total_frames = read_video_frames(input_process_video, use_type='cv2', info=True)

        # image = np.array(input_process_image['background'].convert('RGB'))
        mask = input_process_image['layers'][0].split()[-1].convert('L')
        if mask.height != height and mask.width != width:
            mask = mask.resize((width, height))

        if mask_process_type in ['mask_seg']:
            mask_data = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(video=input_process_video, mask=mask, label=mask_segtag, caption=mask_segtag, mode=mask_type, return_frame=False)
            mask_frames = mask_data['masks']
        elif mask_process_type in ['mask_expand']:
            mask_frames = self.mask_anno_processor[mask_process_type]['anno_ins'].forward(mask=np.array(mask), expand_num=total_frames)
        else:
            raise NotImplementedError

        output_video = []
        if video_process_type in ['framerefext']:
            output_data = self.video_anno_processor[video_process_type]['anno_ins'].forward(video_frames, ref_cfg={'mode': frame_reference_mode}, ref_num=frame_reference_num)
            output_video, mask_frames = output_data['frames'], output_data['masks']
        elif video_process_type in ['outpainting', 'outpainting_inner']:
            # ratio = ((16 / 9 * height) / width - 1) / 2
            output_data = self.video_anno_processor[video_process_type]['anno_ins'].forward(video_frames, direction=outpainting_direction, expand_ratio=outpainting_ratio)
            output_video, mask_frames = output_data['frames'], output_data['masks']
        else:
            output_video = self.video_anno_processor[video_process_type]['anno_ins'].forward(video_frames)


        mask_cfg = {
            'mode': mask_aug_type,
            'kwargs': {
                'expand_ratio': mask_expand_ratio,
                'expand_iters': mask_expand_iters
            }
        }
        # print(mask_cfg)
        if mask_aug_process_type == 'maskaug_layout':
            output_video = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(mask_frames, mask_cfg=mask_cfg, label=mask_layout_label)
            mask_aug_frames = [ np.ones_like(submask) * 255 for submask in mask_frames ]
        elif mask_aug_process_type == 'maskaug':
            mask_aug_frames = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(mask_frames, mask_cfg=mask_cfg)
        else:
            mask_aug_frames = self.maskaug_anno_processor[mask_aug_process_type]['anno_ins'].forward(mask_frames)

        with (tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_video_path, \
                tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as masked_video_path, \
                tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as mask_video_path):
            output_video_writer = imageio.get_writer(output_video_path.name, codec='libx264', fps=fps, quality=8, macro_block_size=None)
            masked_video_writer = imageio.get_writer(masked_video_path.name, codec='libx264', fps=fps, quality=8, macro_block_size=None)
            mask_video_writer = imageio.get_writer(mask_video_path.name, codec='libx264', fps=fps, quality=8, macro_block_size=None)
            for i in range(total_frames):
                output_frame = output_video[i] if len(output_video) > 0 else video_frames[i]
                frame = output_video[i] if len(output_video) > 0 else video_frames[i]
                mask = mask_aug_frames[i]
                if mask_gray:
                    masked_image = frame.copy()
                    masked_image[mask == 255] = 127.5
                else:
                    mask_weight = mask / 255 * mask_opacity
                    masked_image = np.clip(frame * (1 - mask_weight[:, :, None]), 0, 255).astype(np.uint8)
                output_video_writer.append_data(output_frame)
                masked_video_writer.append_data(masked_image)
                mask_video_writer.append_data(mask)
            output_video_writer.close()
            masked_video_writer.close()
            mask_video_writer.close()

            return output_video_path.name, masked_video_path.name, mask_video_path.name

    def save_video_data(self, input_video_path, input_image, video_path, masked_video_path, mask_path):

        save_image_data = {
            "input_image": input_image['background'].convert('RGB') if isinstance(input_image, dict) else input_image,
            "input_image_mask": input_image['layers'][0].split()[-1].convert('L') if isinstance(input_image, dict) else None
        }
        save_video_data = {
            "input_video": input_video_path,
            "output_video": video_path,
            "output_masked_video": masked_video_path,
            "output_video_mask": mask_path
        }
        save_info = {}
        tid = tid_maker()
        for name, image in save_image_data.items():
            if image is None: continue
            save_image_dir = os.path.join(self.save_dir, tid[:8])
            if not os.path.exists(save_image_dir): os.makedirs(save_image_dir)
            save_image_path = os.path.join(save_image_dir, tid + '-' + name + '.png')
            save_info[name] = save_image_path
            image.save(save_image_path)
            gr.Info(f'Save {name} to {save_image_path}', duration=15)
        for name, ori_video_path in save_video_data.items():
            if ori_video_path is None: continue
            save_video_dir = os.path.join(self.save_dir, tid[:8])
            if not os.path.exists(save_video_dir): os.makedirs(save_video_dir)
            save_video_path = os.path.join(save_video_dir, tid + '-' + name + os.path.splitext(ori_video_path)[-1])
            save_info[name] = save_video_path
            shutil.copy(ori_video_path, save_video_path)
            gr.Info(f'Save {name} to {save_video_path}', duration=15)

        save_txt_path = os.path.join(self.save_dir, tid[:8], tid + '.txt')
        save_info['save_info'] = save_txt_path
        with open(save_txt_path, 'w') as f:
            f.write(json.dumps(save_info, ensure_ascii=False))
        return dict_to_markdown_table(save_info)


    def change_process_type(self, video_process_type, mask_process_type, mask_aug_process_type):
        frame_reference_setting_visible = False
        outpainting_setting_visible = False
        segment_setting = False
        maskaug_setting = False
        if video_process_type in ["framerefext"]:
            frame_reference_setting_visible = True
        elif video_process_type in ["outpainting", "outpainting_inner"]:
            outpainting_setting_visible = True
        if mask_process_type in ["mask_seg"]:
            segment_setting = True
        if mask_aug_process_type in ["maskaug", "maskaug_layout"]:
            maskaug_setting = True
        return gr.update(visible=frame_reference_setting_visible), gr.update(visible=outpainting_setting_visible), gr.update(visible=segment_setting), gr.update(visible=maskaug_setting)


    def set_callbacks_video(self, **kwargs):
        inputs = [self.input_process_video, self.input_process_image, self.video_process_type, self.outpainting_direction, self.outpainting_ratio, self.frame_reference_mode, self.frame_reference_num, self.mask_process_type, self.mask_type, self.mask_segtag, self.mask_opacity, self.mask_gray, self.mask_aug_process_type, self.mask_aug_type, self.mask_expand_ratio, self.mask_expand_iters, self.mask_layout_label]
        outputs = [self.output_process_video, self.output_process_masked_video, self.output_process_video_mask]
        self.process_button.click(self.process_video_data, inputs=inputs, outputs=outputs)
        self.input_process_video.change(read_video_one_frame, inputs=[self.input_process_video], outputs=[self.input_process_first_image_show])
        self.input_process_video.change(read_video_last_frame, inputs=[self.input_process_video], outputs=[self.input_process_last_image_show])
        self.save_button.click(self.save_video_data,
                               inputs=[self.input_process_video, self.input_process_image, self.output_process_video, self.output_process_masked_video, self.output_process_video_mask],
                               outputs=[self.save_log])
        process_inputs = [self.video_process_type, self.mask_process_type, self.mask_aug_process_type]
        process_outputs = [self.frame_reference_setting, self.outpainting_setting, self.segment_setting, self.maskaug_setting]
        self.video_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)
        self.mask_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)
        self.mask_aug_process_type.change(self.change_process_type, inputs=process_inputs, outputs=process_outputs)



class VACETagComposition():
    def __init__(self, cfg, device=None):
        self.save_dir = os.path.join(cfg.save_dir, 'composition')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        anno_name = 'composition'
        anno_cfg = copy.deepcopy(VACE_COMPOSITION_PREPROCCESS_CONFIGS[anno_name])
        class_name = anno_cfg.pop("NAME")
        input_params = anno_cfg.pop("INPUTS")
        output_params = anno_cfg.pop("OUTPUTS")
        anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
        self.comp_anno_processor = {"inputs": input_params, "outputs": output_params,
                                    "anno_ins": anno_ins}
        self.process_types = ["repaint", "extension", "control"]

    def create_ui_composition(self, *args, **kwargs):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                self.input_process_video_1 = gr.Video(
                    label="input_process_video_1",
                    sources=['upload'],
                    interactive=True)
            with gr.Column(scale=1):
                self.input_process_video_2 = gr.Video(
                    label="input_process_video_1",
                    sources=['upload'],
                    interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_video_mask_1 = gr.Video(
                        label="input_process_video_mask_1",
                        sources=['upload'],
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_video_mask_2 = gr.Video(
                        label="input_process_video_mask_2",
                        sources=['upload'],
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_type_1 = gr.Dropdown(
                        label='input_process_type_1',
                        choices=list(self.process_types),
                        value=list(self.process_types)[0],
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_type_2 = gr.Dropdown(
                        label='input_process_type_2',
                        choices=list(self.process_types),
                        value=list(self.process_types)[0],
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.process_button = gr.Button(
                        value='[1]Sample Process',
                        elem_classes='type_row',
                        elem_id='process_button',
                        visible=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                self.output_process_video = gr.Video(
                    label="output_process_video",
                    sources=['upload'],
                    interactive=False)
            with gr.Column(scale=1):
                self.output_process_mask = gr.Video(
                    label="output_process_mask",
                    sources=['upload'],
                    interactive=False)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.save_button = gr.Button(
                        value='[2]Sample Save',
                        elem_classes='type_row',
                        elem_id='save_button',
                        visible=True)
                with gr.Row():
                    self.save_log = gr.Markdown()

    def process_composition_data(self, input_process_video_1, input_process_video_2, input_process_video_mask_1, input_process_video_mask_2, input_process_type_1, input_process_type_2):
        # "repaint", "extension", "control"
        # ('repaint', 'repaint') / ('repaint', 'extension') / ('repaint', 'control')
        # ('extension', 'extension') / ('extension', 'repaint') / ('extension', 'control')
        # ('control', 'control') / ('control', 'repaint') / ('control', 'extension')

        video_frames_1, video_fps_1, video_width_1, video_height_1, video_total_frames_1 = read_video_frames(input_process_video_1, use_type='cv2', info=True)
        video_frames_2, video_fps_2, video_width_2, video_height_2, video_total_frames_2 = read_video_frames(input_process_video_2, use_type='cv2', info=True)
        mask_frames_1, mask_fps_1, mask_width_1, mask_height_1, mask_total_frames_1 = read_video_frames(input_process_video_mask_1, use_type='cv2', info=True)
        mask_frames_2, mask_fps_2, mask_width_2, mask_height_2, mask_total_frames_2 = read_video_frames(input_process_video_mask_2, use_type='cv2', info=True)
        mask_frames_1 = [np.where(mask > 127, 1, 0).astype(np.uint8) for mask in mask_frames_1]
        mask_frames_2 = [np.where(mask > 127, 1, 0).astype(np.uint8) for mask in mask_frames_2]

        assert video_width_1 == video_width_2 == mask_width_1 == mask_width_2
        assert video_height_1 == video_height_2 == mask_height_1 == mask_height_2
        assert video_fps_1 == video_fps_2

        output_video, output_mask = self.comp_anno_processor['anno_ins'].forward(input_process_type_1, input_process_type_2, video_frames_1, video_frames_2, mask_frames_1, mask_frames_2)

        fps = video_fps_1
        total_frames = len(output_video)
        if output_video is not None and output_mask is not None:
            with (tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_video_path, \
                    tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as mask_video_path):
                output_video_writer = imageio.get_writer(output_video_path.name, codec='libx264', fps=fps, quality=8, macro_block_size=None)
                mask_video_writer = imageio.get_writer(mask_video_path.name, codec='libx264', fps=fps, quality=8, macro_block_size=None)
                for i in range(total_frames):
                    output_video_writer.append_data(output_video[i])
                    mask_video_writer.append_data(output_mask[i])
                output_video_writer.close()
                mask_video_writer.close()

                return output_video_path.name, mask_video_path.name
        else:
            return None, None

    def save_composition_data(self, video_path, mask_path):
        save_video_data = {
            "output_video": video_path,
            "output_video_mask": mask_path
        }
        save_info = {}
        tid = tid_maker()
        for name, ori_video_path in save_video_data.items():
            if ori_video_path is None: continue
            save_video_dir = os.path.join(self.save_dir, tid[:8])
            if not os.path.exists(save_video_dir): os.makedirs(save_video_dir)
            save_video_path = os.path.join(save_video_dir, tid + '-' + name + os.path.splitext(ori_video_path)[-1])
            save_info[name] = save_video_path
            shutil.copy(ori_video_path, save_video_path)
            gr.Info(f'Save {name} to {save_video_path}', duration=15)
        save_txt_path = os.path.join(self.save_dir, tid[:8], tid + '.txt')
        save_info['save_info'] = save_txt_path
        with open(save_txt_path, 'w') as f:
            f.write(json.dumps(save_info, ensure_ascii=False))
        return dict_to_markdown_table(save_info)

    def set_callbacks_composition(self, **kwargs):
        inputs = [self.input_process_video_1, self.input_process_video_2, self.input_process_video_mask_1, self.input_process_video_mask_2, self.input_process_type_1, self.input_process_type_2]
        outputs = [self.output_process_video, self.output_process_mask]
        self.process_button.click(self.process_composition_data,
                                  inputs=inputs,
                                  outputs=outputs)
        self.save_button.click(self.save_composition_data,
                               inputs=[self.output_process_video, self.output_process_mask],
                               outputs=[self.save_log])


class VACEVideoTool():
    def __init__(self, cfg, device=None):
        self.save_dir = os.path.join(cfg.save_dir, 'video_tool')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.process_types = ["expand_frame", "expand_blank_clip", "expand_clip_blank", "expand_ff_clip_blank_lf", "concat_clip", "concat_ff_clip_lf", "blank_mask"]

    def create_ui_video_tool(self, *args, **kwargs):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_image_1 = gr.Image(
                        label="input_process_image_1",
                        type='pil',
                        format='png',
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_image_2 = gr.Image(
                        label="input_process_image_2",
                        type='pil',
                        format='png',
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                self.input_process_video_1 = gr.Video(
                    label="input_process_video_1",
                    sources=['upload'],
                    interactive=True)
            with gr.Column(scale=1):
                self.input_process_video_2 = gr.Video(
                    label="input_process_video_2",
                    sources=['upload'],
                    interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_video_mask_1 = gr.Video(
                        label="input_process_video_mask_1",
                        sources=['upload'],
                        interactive=True)
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_video_mask_2 = gr.Video(
                        label="input_process_video_mask_2",
                        sources=['upload'],
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.input_process_type = gr.Dropdown(
                        label='input_process_type',
                        choices=list(self.process_types),
                        value=list(self.process_types)[0],
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_height = gr.Textbox(
                        label='resolutions_height',
                        value=720,
                        interactive=True)
                    self.output_width = gr.Textbox(
                        label='resolutions_width',
                        value=1280,
                        interactive=True)
                    self.frame_rate = gr.Textbox(
                        label='frame_rate',
                        value=16,
                        interactive=True)
                    self.num_frames = gr.Textbox(
                        label='num_frames',
                        value=81,
                        interactive=True)
                    self.mask_gray = gr.Checkbox(
                        label='Mask Gray',
                        value=False,
                        interactive=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.process_button = gr.Button(
                        value='[1]Sample Process',
                        elem_classes='type_row',
                        elem_id='process_button',
                        visible=True)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.output_process_image = gr.Image(
                        label="output_process_image",
                        value=None,
                        type='pil',
                        image_mode='RGB',
                        format='png',
                        interactive=False)
            with gr.Column(scale=1):
                self.output_process_video = gr.Video(
                    label="output_process_video",
                    sources=['upload'],
                    interactive=False)
            with gr.Column(scale=1):
                self.output_process_mask = gr.Video(
                    label="output_process_mask",
                    sources=['upload'],
                    interactive=False)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                with gr.Row():
                    self.save_button = gr.Button(
                        value='[2]Sample Save',
                        elem_classes='type_row',
                        elem_id='save_button',
                        visible=True)
                with gr.Row():
                    self.save_log = gr.Markdown()

    def process_tool_data(self, input_process_image_1, input_process_image_2, input_process_video_1, input_process_video_2, input_process_video_mask_1, input_process_video_mask_2, input_process_type, output_height, output_width, frame_rate, num_frames):
        output_height, output_width, frame_rate, num_frames = int(output_height), int(output_width), int(frame_rate), int(num_frames)
        output_video, output_mask = None, None
        if input_process_type == 'expand_frame':
            assert input_process_image_1 or input_process_image_2
            output_video = [np.ones((output_height, output_width, 3), dtype=np.uint8) * 127.5] * num_frames
            output_mask = [np.ones((output_height, output_width), dtype=np.uint8) * 255] * num_frames
            if input_process_image_1 is not None:
                output_video[0] = np.array(input_process_image_1.resize((output_width, output_height)))
                output_mask[0] = np.zeros((output_height, output_width))
            if input_process_image_2 is not None:
                output_video[-1] = np.array(input_process_image_2.resize((output_width, output_height)))
                output_mask[-1] = np.zeros((output_height, output_width))
        elif input_process_type == 'expand_blank_clip':
            video_frames, fps, width, height, total_frames = read_video_frames(input_process_video_1, use_type='cv2', info=True)
            frame_rate = fps
            output_video = [np.ones((height, width, 3), dtype=np.uint8) * 127.5] * num_frames + video_frames
            output_mask = [np.ones((height, width), dtype=np.uint8) * 255] * num_frames + [np.zeros((height, width), dtype=np.uint8)] * total_frames
        elif input_process_type == 'expand_clip_blank':
            video_frames, fps, width, height, total_frames = read_video_frames(input_process_video_1, use_type='cv2', info=True)
            frame_rate = fps
            output_video = video_frames + [np.ones((height, width, 3), dtype=np.uint8) * 127.5] * num_frames
            output_mask = [np.zeros((height, width), dtype=np.uint8)] * total_frames + [np.ones((height, width), dtype=np.uint8) * 255] * num_frames
        elif input_process_type == 'expand_ff_clip_blank_lf':
            video_frames, fps, width, height, total_frames = read_video_frames(input_process_video_1, use_type='cv2', info=True)
            frame_rate = fps
            if input_process_image_1 is not None:
                output_video = [np.ones((height, width, 3), dtype=np.uint8) * 127.5] * num_frames + video_frames
                output_mask = [np.ones((height, width), dtype=np.uint8) * 255] * num_frames + [np.zeros((height, width), dtype=np.uint8)] * total_frames
                output_video[0] = np.array(input_process_image_1.resize((width, height)))
                output_mask[0] = np.zeros((height, width))
            if input_process_image_2 is not None:
                output_video = video_frames + [np.ones((height, width, 3), dtype=np.uint8) * 127.5] * num_frames
                output_mask = [np.zeros((height, width), dtype=np.uint8)] * total_frames + [np.ones((height, width), dtype=np.uint8) * 255] * num_frames
                output_video[-1] = np.array(input_process_image_2.resize((width, height)))
                output_mask[-1] = np.zeros((height, width))
        elif input_process_type == 'concat_clip':
            video_frames_1, fps_1, width_1, height_1, total_frames_1 = read_video_frames(input_process_video_1, use_type='cv2', info=True)
            video_frames_2, fps_2, width_2, height_2, total_frames_2 = read_video_frames(input_process_video_2, use_type='cv2', info=True)
            if width_1 != width_2 or height_1 != height_2:
                video_frames_2 = [np.array(frame.resize((width_1, height_1))) for frame in video_frames_2]
            frame_rate = fps_1
            output_video = video_frames_1 + video_frames_2
            output_mask = [np.ones((height_1, width_1), dtype=np.uint8) * 255] * len(output_video)
        elif input_process_type == 'concat_ff_clip_lf':
            video_frames_1, fps_1, width_1, height_1, total_frames_1 = read_video_frames(input_process_video_1, use_type='cv2', info=True)
            video_masks_1 = [np.ones((height_1, width_1), dtype=np.uint8) * 255] * total_frames_1
            frame_rate = fps_1
            if input_process_image_1 is not None:
                video_frames_1 = [np.array(input_process_image_1.resize((width_1, height_1)))] + video_frames_1
                video_masks_1 = [np.zeros((height_1, width_1))] + video_masks_1
            if input_process_image_2 is not None:
                video_frames_1 = video_frames_1 + [np.array(input_process_image_2.resize((width_1, height_1)))]
                video_masks_1 = video_masks_1 + [np.zeros((height_1, width_1))]
            output_video = video_frames_1
            output_mask = video_masks_1
        elif input_process_type == 'blank_mask':
            output_mask = [np.ones((output_height, output_width), dtype=np.uint8) * 255] * num_frames
        else:
            raise NotImplementedError
        output_image_path = None

        if output_video is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_path:
                flag = save_one_video(videos=output_video, file_path=output_path.name, fps=frame_rate)
                output_video_path = output_path.name if flag else None
        else:
            output_video_path = None

        if output_mask is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_path:
                flag = save_one_video(videos=output_mask, file_path=output_path.name, fps=frame_rate)
                output_mask_path = output_path.name if flag else None
        else:
            output_mask_path = None
        return output_image_path, output_video_path, output_mask_path


    def save_tool_data(self, image_path, video_path, mask_path):
        save_video_data = {
            "output_video": video_path,
            "output_video_mask": mask_path
        }
        save_info = {}
        tid = tid_maker()
        for name, ori_video_path in save_video_data.items():
            if ori_video_path is None: continue
            save_video_path = os.path.join(self.save_dir, tid[:8], tid + '-' + name + os.path.splitext(ori_video_path)[-1])
            save_info[name] = save_video_path
            shutil.copy(ori_video_path, save_video_path)
            gr.Info(f'Save {name} to {save_video_path}', duration=15)
        save_txt_path = os.path.join(self.save_dir, tid[:8], tid + '.txt')
        save_info['save_info'] = save_txt_path
        with open(save_txt_path, 'w') as f:
            f.write(json.dumps(save_info, ensure_ascii=False))
        return dict_to_markdown_table(save_info)

    def set_callbacks_video_tool(self, **kwargs):
        inputs = [self.input_process_image_1, self.input_process_image_2, self.input_process_video_1, self.input_process_video_2, self.input_process_video_mask_1, self.input_process_video_mask_2, self.input_process_type, self.output_height, self.output_width, self.frame_rate, self.num_frames]
        outputs = [self.output_process_image, self.output_process_video, self.output_process_mask]
        self.process_button.click(self.process_tool_data,
                                  inputs=inputs,
                                  outputs=outputs)
        self.save_button.click(self.save_tool_data,
                               inputs=[self.output_process_image, self.output_process_video, self.output_process_mask],
                               outputs=[self.save_log])


class VACETag():

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.current_index = 0
        self.loaded_data = {}

        self.vace_video_tag = VACEVideoTag(cfg)
        self.vace_image_tag = VACEImageTag(cfg)
        self.vace_tag_composition = VACETagComposition(cfg)
        self.vace_video_tool = VACEVideoTool(cfg)


    def create_ui(self, *args, **kwargs):
        gr.Markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 15px;">
                        <a href="https://ali-vilab.github.io/VACE-Page/" style="text-decoration: none; color: inherit;">VACE Preprocessor</a>
                    </div>
                    """)
        with gr.Tabs(elem_id='VACE Tag') as vace_tab:
            with gr.TabItem('VACE Video Tag', id=1, elem_id='video_tab'):
                self.vace_video_tag.create_ui_video(*args, **kwargs)
            with gr.TabItem('VACE Image Tag', id=2, elem_id='image_tab'):
                self.vace_image_tag.create_ui_image(*args, **kwargs)
            with gr.TabItem('VACE Composition Tag', id=3, elem_id='composition_tab'):
                self.vace_tag_composition.create_ui_composition(*args, **kwargs)
            with gr.TabItem('VACE Video Tool', id=4, elem_id='video_tool_tab'):
                self.vace_video_tool.create_ui_video_tool(*args, **kwargs)


    def set_callbacks(self, **kwargs):
        self.vace_video_tag.set_callbacks_video(**kwargs)
        self.vace_image_tag.set_callbacks_image(**kwargs)
        self.vace_tag_composition.set_callbacks_composition(**kwargs)
        self.vace_video_tool.set_callbacks_video_tool(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for VACE-Preprocessor:\n')
    parser.add_argument('--server_port', dest='server_port', help='', default=7860)
    parser.add_argument('--server_name', dest='server_name', help='', default='0.0.0.0')
    parser.add_argument('--root_path', dest='root_path', help='', default=None)
    parser.add_argument('--save_dir', dest='save_dir', help='', default='cache')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    vace_tag = VACETag(args)
    with gr.Blocks() as demo:
        vace_tag.create_ui()
        vace_tag.set_callbacks()
        demo.queue(status_update_rate=1).launch(server_name=args.server_name,
                                                server_port=int(args.server_port),
                                                show_api=False, show_error=True,
                                                debug=True)