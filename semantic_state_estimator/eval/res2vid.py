import glob
from itertools import combinations
import json
import os

import cv2
import fire
from natsort import natsorted
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from semantic_state_estimator.constants import RENDERS_DIR, PROCESSED_DIR
from semantic_state_estimator.utils.statistics import get_cooccurrence_matrix


def get_annotated_image(frame, predictions):
    img = Image.fromarray(frame)
    imd = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("LiberationMono-Regular.ttf", 30)
    for i, predicate in enumerate(sorted(predictions.keys())):
        imd.text(
            (28, 36 + 30 * i),
            f"{predicate}: {predictions[predicate]}",
            font=fnt,
            fill=(0, 255, 0),
        )
    return img


def res2pred(res, confidence=0.5, cooc_matrix=None):
    predictions = {k: v > confidence for k, v in res.items()}

    if cooc_matrix is not None:
        for pred1, pred2 in combinations(cooc_matrix.columns, 2):
            if (
                predictions[pred1]
                and predictions[pred2]
                and not cooc_matrix[pred1][pred2]
            ):
                if res[pred1] > res[pred2]:
                    predictions[pred2] = False
                else:
                    predictions[pred1] = False

    return predictions


def pil2vid(images: list[Image.Image], fps: int, save_path: str):
    videodims = images[0].size[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, fps, videodims)

    for img in tqdm(images, desc=f"saving to {save_path}", leave=False):
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    video.release()


def res2vid(
    data_dir,
    res_dir_name,
    fps,
    prediction_confidence_threshold=0.5,
    out_dir=None,
    cooc_data_dir=None,
):
    # get all frame files
    # files are sorted using `natsort` for natural number sorting, even if not enough 
    # preceeding zeros.
    sorted_frame_files = natsorted(
        glob.glob(os.path.join(data_dir, RENDERS_DIR, "*.npz"))
    )

    cooc_matrix = None
    if cooc_data_dir is not None:
        cooc_matrix = get_cooccurrence_matrix(cooc_data_dir)
    
    annotated_frames = {}
    for frame_file in tqdm(
        sorted_frame_files,
        total=len(sorted_frame_files),
        desc="annotating",
    ):
        view_to_frame = np.load(frame_file)

        res_filename = os.path.splitext(os.path.basename(frame_file))[0] + '.json'
        res_file = os.path.join(data_dir, PROCESSED_DIR, res_dir_name, res_filename)
        try:
            with open(res_file, "r") as f:
                res = json.load(f)
            predictions = res2pred(res, prediction_confidence_threshold, cooc_matrix)
        except FileNotFoundError:
            raise
            predictions = {}  # will yeild an empty annotation in `get_annotated_image`

        for view, frame in view_to_frame.items():
            annotated_frames.setdefault(view, []).append(
                get_annotated_image(frame, predictions)
            )

    if out_dir is None:
        out_dir = os.path.join(data_dir, "video", res_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    for view, images in tqdm(annotated_frames.items(), desc="saving"):
        pil2vid(images, fps, os.path.join(out_dir, view + ".mp4"))


if __name__ == "__main__":
    fire.Fire(res2vid)
