# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import copy
import time
import inspect
import argparse
import importlib

from configs import VACE_PREPROCCESS_CONFIGS
import annotators
from annotators.utils import read_image, read_mask, read_video_frames, save_one_video, save_one_image


def parse_bboxes(s):
    bboxes = []
    for bbox_str in s.split():
        coords = list(map(float, bbox_str.split(',')))
        if len(coords) != 4:
            raise ValueError(f"The bounding box requires 4 values, but the input is {len(coords)}.")
        bboxes.append(coords)
    return bboxes

def validate_args(args):
    assert args.task in VACE_PREPROCCESS_CONFIGS, f"Unsupport task: [{args.task}]"
    assert args.video is not None or args.image is not None or args.bbox is not None, "Please specify the video or image or bbox."
    return args

def get_parser():
    parser = argparse.ArgumentParser(
        description="Data processing carried out by VACE"
    )
    parser.add_argument(
        "--task",
        type=str,
        default='',
        choices=list(VACE_PREPROCCESS_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="The path of the videos to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The path of the images to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="The specific mode of the task, such as firstframe, mask, bboxtrack, label...")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="The path of the mask images to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--bbox",
        type=parse_bboxes,
        default=None,
        help="Enter the bounding box, with each four numbers separated by commas (x1, y1, x2, y2), and each pair separated by a space."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Enter the label to be processed, separated by commas if there are multiple."
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Enter the caption to be processed."
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="The direction of outpainting includes any combination of left, right, up, down, with multiple combinations separated by commas.")
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=None,
        help="The outpainting's outward expansion ratio.")
    parser.add_argument(
        "--expand_num",
        type=int,
        default=None,
        help="The number of frames extended by the extension task.")
    parser.add_argument(
        "--maskaug_mode",
        type=str,
        default=None,
        help="The mode of mask augmentation, such as original, original_expand, hull, hull_expand, bbox, bbox_expand.")
    parser.add_argument(
        "--maskaug_ratio",
        type=float,
        default=None,
        help="The ratio of mask augmentation.")
    parser.add_argument(
        "--pre_save_dir",
        type=str,
        default=None,
        help="The path to save the processed data.")
    parser.add_argument(
        "--save_fps",
        type=int,
        default=16,
        help="The fps to save the processed data.")
    return parser


def preproccess():
    pass

def proccess():
    pass

def postproccess():
    pass

def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    task_name = args.task
    video_path = args.video
    image_path = args.image
    mask_path = args.mask
    bbox = args.bbox
    caption = args.caption
    label = args.label
    save_fps = args.save_fps

    # init class
    task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)[task_name]
    class_name = task_cfg.pop("NAME")
    input_params = task_cfg.pop("INPUTS")
    output_params = task_cfg.pop("OUTPUTS")

    # input data
    fps = None
    input_data = copy.deepcopy(input_params)
    if 'video' in input_params:
        assert video_path is not None, "Please set video or check configs"
        frames, fps, width, height, num_frames = read_video_frames(video_path.split(",")[0], use_type='cv2',  info=True)
        assert frames is not None, "Video read error"
        input_data['frames'] = frames
        input_data['video'] = video_path
    if 'frames' in input_params:
        assert video_path is not None, "Please set video or check configs"
        frames, fps, width, height, num_frames = read_video_frames(video_path.split(",")[0], use_type='cv2', info=True)
        assert frames is not None, "Video read error"
        input_data['frames'] = frames
    if 'frames_2' in input_params:
        # assert video_path is not None and len(video_path.split(",")[1]) >= 2, "Please set two videos or check configs"
        if  len(video_path.split(",")) >= 2:
            frames, fps, width, height, num_frames = read_video_frames(video_path.split(",")[1], use_type='cv2', info=True)
            assert frames is not None, "Video read error"
            input_data['frames_2'] = frames
    if 'image' in input_params:
        assert image_path is not None, "Please set image or check configs"
        image, width, height = read_image(image_path.split(",")[0], use_type='pil', info=True)
        assert image is not None, "Image read error"
        input_data['image'] = image
    if 'image_2' in input_params:
        # assert image_path is not None and len(image_path.split(",")[1]) >= 2, "Please set two images or check configs"
        if len(image_path.split(",")) >= 2:
            image, width, height = read_image(image_path.split(",")[1], use_type='pil', info=True)
            assert image is not None, "Image read error"
            input_data['image_2'] = image
    if 'images' in input_params:
        assert image_path is not None, "Please set image or check configs"
        images = [ read_image(path, use_type='pil', info=True)[0] for path in image_path.split(",") ]
        input_data['images'] = images
    if 'mask' in input_params:
        # assert mask_path is not None, "Please set mask or check configs"
        if mask_path is not None:
            mask, width, height = read_mask(mask_path.split(",")[0], use_type='pil', info=True)
            assert mask is not None, "Mask read error"
            input_data['mask'] = mask
    if 'bbox' in input_params:
        # assert bbox is not None, "Please set bbox"
        if bbox is not None:
            input_data['bbox'] = bbox[0] if len(bbox) == 1 else bbox
    if 'label' in input_params:
        # assert label is not None, "Please set label or check configs"
        input_data['label'] = label.split(',') if label is not None else None
    if 'caption' in input_params:
        # assert caption is not None, "Please set caption or check configs"
        input_data['caption'] = caption
    if 'mode' in input_params:
        input_data['mode'] = args.mode
    if 'direction' in input_params:
        if args.direction is not None:
            input_data['direction'] = args.direction.split(',')
    if 'expand_ratio' in input_params:
        if args.expand_ratio is not None:
            input_data['expand_ratio'] = args.expand_ratio
    if 'expand_num' in input_params:
        # assert args.expand_num is not None, "Please set expand_num or check configs"
        if args.expand_num is not None:
            input_data['expand_num'] = args.expand_num
    if 'mask_cfg' in input_params:
        # assert args.maskaug_mode is not None and args.maskaug_ratio is not None, "Please set maskaug_mode and maskaug_ratio or check configs"
        if args.maskaug_mode is not None:
            if args.maskaug_ratio is not None:
                input_data['mask_cfg'] = {"mode": args.maskaug_mode, "kwargs": {'expand_ratio': args.maskaug_ratio, 'expand_iters': 5}}
            else:
                input_data['mask_cfg'] = {"mode": args.maskaug_mode}

    # processing
    pre_ins = getattr(annotators, class_name)(cfg=task_cfg)
    results = pre_ins.forward(**input_data)

    # output data
    save_fps = fps if fps is not None else save_fps
    if args.pre_save_dir is None:
        pre_save_dir = os.path.join('processed', task_name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    else:
        pre_save_dir = args.pre_save_dir
    if not os.path.exists(pre_save_dir):
        os.makedirs(pre_save_dir)

    ret_data = {}
    if 'frames' in output_params:
        frames =  results['frames'] if isinstance(results, dict) else results
        if frames is not None:
            save_path = os.path.join(pre_save_dir, f'src_video-{task_name}.mp4')
            save_one_video(save_path, frames, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data['src_video'] = save_path
    if 'masks' in output_params:
        frames = results['masks'] if isinstance(results, dict) else results
        if frames is not None:
            save_path = os.path.join(pre_save_dir, f'src_mask-{task_name}.mp4')
            save_one_video(save_path, frames, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data['src_mask'] = save_path
    if 'image' in output_params:
        ret_image =  results['image'] if isinstance(results, dict) else results
        if ret_image is not None:
            save_path = os.path.join(pre_save_dir, f'src_ref_image-{task_name}.png')
            save_one_image(save_path, ret_image, use_type='pil')
            print(f"Save image result to {save_path}")
            ret_data['src_ref_images'] = save_path
    if 'images' in output_params:
        ret_images = results['images'] if isinstance(results, dict) else results
        if ret_images is not None:
            src_ref_images = []
            for i, img in enumerate(ret_images):
                if img is not None:
                    save_path = os.path.join(pre_save_dir, f'src_ref_image_{i}-{task_name}.png')
                    save_one_image(save_path, img, use_type='pil')
                    print(f"Save image result to {save_path}")
                    src_ref_images.append(save_path)
            if len(src_ref_images) > 0:
                ret_data['src_ref_images'] = ','.join(src_ref_images)
            else:
                ret_data['src_ref_images'] = None
    if 'mask' in output_params:
        ret_image =  results['mask'] if isinstance(results, dict) else results
        if ret_image is not None:
            save_path = os.path.join(pre_save_dir, f'src_mask-{task_name}.png')
            save_one_image(save_path, ret_image, use_type='pil')
            print(f"Save mask result to {save_path}")
    return ret_data


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
