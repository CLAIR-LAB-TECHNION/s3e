import datetime
import fire
import glob
import os
import json
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from semantic_state_estimator.constants import RENDERS_DIR, PROCESSED_DIR
from semantic_state_estimator.utils.misc import load_se_from_args


def predict_dp_state(renders, se):
    imgs = [Image.fromarray(img) for img in renders.values()]
    prob_map = se.estimate_state(imgs)
    return prob_map


def process_datapoints(
    data_dir,
    domain,
    problem,
    query_swapper=None,
    out_dir=None,
    separate_images=False,
    se_class=None,
    **se_kwargs,
):
    # set output dirname if not specified
    out_dir = out_dir or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # load state estimator
    se = load_se_from_args(se_class, se_kwargs, domain, problem)

    # get problem literals
    # lit_map = get_lit_map(domain, problem, data_dir)

    # collect datapoint renders
    render_files = glob.glob(os.path.join(data_dir, RENDERS_DIR, "*.npz"))

    # create output directory
    out_dir = os.path.join(data_dir, PROCESSED_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # process all datapoints
    for renders_file in tqdm(render_files, desc="processing data points"):
        # skip if result exists
        out_filename = os.path.splitext(os.path.basename(renders_file))[0] + ".json"
        out_file = os.path.join(out_dir, out_filename)
        if os.path.exists(out_file):
            continue

        # load datapoint
        with np.load(renders_file) as data:
            renders = {k: data[k] for k in data}

        if query_swapper is not None:
            se.swap_queries(*query_swapper(renders_file))

        # process datapoint
        if separate_images:
            prob_map = {
                k: predict_dp_state({k: rnd}, se)
                for k, rnd in renders.items()
            }
        else:
            prob_map = predict_dp_state(renders, se)

        # save processed datapoint
        with open(out_file, "w") as f:
            json.dump(prob_map, f, indent=4)


if __name__ == "__main__":
    fire.Fire(process_datapoints)
