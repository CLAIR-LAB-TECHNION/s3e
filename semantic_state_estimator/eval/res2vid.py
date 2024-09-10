import glob
import json
import os

import cv2
import fire
from natsort import natsorted
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from semantic_state_estimator.constants import RENDERS_DIR, PROCESSED_DIR


def get_annotated_image(frame, res):
    img = Image.fromarray(frame)
    imd = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("LiberationMono-Regular.ttf", 30)
    for i, predicate in enumerate(sorted(res.keys())):
        imd.text((28, 36 + 30 * i), f'{predicate}: {res[predicate] > 0.5}', font=fnt, fill=(0, 255, 0))
    return img


def pil2vid(images: list[Image.Image], fps: int, save_path: str):
    videodims = images[0].size[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    video = cv2.VideoWriter(save_path, fourcc, fps, videodims)
    
    for img in tqdm(images, desc=f'saving to {save_path}', leave=False):
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    video.release()


def res2vid(data_dir, res_dir_name, fps, out_dir=None):
    sorted_frame_files = natsorted(glob.glob(os.path.join(data_dir, RENDERS_DIR, '*.npz')))
    sorted_res_files = natsorted(glob.glob(os.path.join(data_dir, PROCESSED_DIR, res_dir_name, '*.json')))

    annotated_frames = {}

    assert len(sorted_frame_files) == len(sorted_res_files), f"expected {len(sorted_frame_files)} res files but got {len(sorted_res_files)}"
    for frame_file, res_file in tqdm(zip(sorted_frame_files, sorted_res_files), total=len(sorted_frame_files), desc='annotating'):
        with open(res_file, 'r') as f:
            res = json.load(f)
        view_to_frame = np.load(frame_file)
        for view, frame in view_to_frame.items():
            annotated_frames.setdefault(view, []).append(get_annotated_image(frame, res))
    
    if out_dir is None:
        out_dir = os.path.join(data_dir, 'video', res_dir)
    os.makedirs(out_dir, exist_ok=True)
    for view, images in tqdm(annotated_frames.items(), desc='saving'):
        pil2vid(images, fps, os.path.join(out_dir, view + '.mp4'))

if __name__ == "__main__":
    fire.Fire(res2vid)
