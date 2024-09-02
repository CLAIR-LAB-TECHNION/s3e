import datetime
import fire
import glob
import os
import json
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from semantic_state_estimator.constants import (
    LLAMA_70B_INSTRUCT,
    LLAVA_7B_OV,
    RENDERS_DIR,
)


PROCESSED_DATA_DIR = "processed"


def predict_dp_state(renders, se):
    imgs = [Image.fromarray(img) for img in renders.values()]
    prob_map = se.estimate_state(imgs)
    return prob_map


def process_datapoints(
    data_dir,
    domain,
    problem,
    out_dir=None,
    se_class=None,
    **se_kwargs,
):
    # these modules import heavy packages.
    # import here to avoid waiting when calling with `--help`
    from semantic_state_estimator.utils.misc import (
        load_from_entrypoint,
    )
    from semantic_state_estimator.semantic_state_estimator import (
        SemanticStateEstimator,
        SemanticStateEstimatorWithLLaMA,
    )

    # set output dirname if not specified
    out_dir = out_dir or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # load state estimator class
    if se_class is None:
        se_class = SemanticStateEstimatorWithLLaMA
    elif isinstance(se_class, str):
        se_class = load_from_entrypoint(se_class)

    # handle default state estimator class
    if se_class == SemanticStateEstimatorWithLLaMA:
        se_kwargs.setdefault("nl_converter_model_id", LLAMA_70B_INSTRUCT)
        se_kwargs.setdefault("vqa_model_id", LLAVA_7B_OV)
    elif se_class == SemanticStateEstimator:
        se_kwargs.setdefault("vqa_model_id", LLAVA_7B_OV)

    # load state estimator
    se = se_class(domain=domain, problem=problem, **se_kwargs)

    # get problem literals
    # lit_map = get_lit_map(domain, problem, data_dir)

    # collect datapoint renders
    render_files = glob.glob(os.path.join(data_dir, RENDERS_DIR, "*.npz"))

    # process all datapoints
    for renders_file in tqdm(render_files):
        # load datapoint
        renders = np.load(renders_file)

        # process datapoint
        prob_map = predict_dp_state(renders, se)

        # save processed datapoint
        out_filename = os.path.splitext(os.path.basename(renders_file))[0] + ".json"
        out_file = os.path.join(data_dir, PROCESSED_DATA_DIR, out_dir, out_filename)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(prob_map, f, indent=4)


if __name__ == "__main__":
    fire.Fire(process_datapoints)
