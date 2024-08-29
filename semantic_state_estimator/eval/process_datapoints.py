import fire
import glob
import os
import pickle
import json
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from semantic_state_estimator.constants import (
    TRUE_STATE_KEY,
    TRUE_STATE_ARR_KEY,
    ESTIMATED_STATE_ARR_PROB_KEY,
    LLAMA_70B_INSTRUCT,
    LLAVA_7B_OV,
    LIT_MAP_FILE_NAME,
)
from semantic_state_estimator.utils.up_utils import (
    create_up_problem,
    get_all_grounded_predicates_for_objects,
)


def get_lit_map(domain, problem, data_dir):
    lit_map_file = os.path.join(data_dir, LIT_MAP_FILE_NAME)

    # load if already exists
    if os.path.exists(lit_map_file):
        with open(lit_map_file, "r") as f:
            return json.load(f)

    # create UP problem
    up_problem = create_up_problem(domain, problem)

    # enumerate literals and map to index
    ground_literals = get_all_grounded_predicates_for_objects(up_problem)
    lit_map = {str(lit): i for i, lit in enumerate(ground_literals)}

    # save for reference
    with open(lit_map_file, "w") as f:
        json.dump(lit_map, f)

    return lit_map


def predict_dp_state(dp, se):
    imgs = [Image.fromarray(img) for img in dp["renders"].values()]
    prob_map = se.estimate_state(imgs)
    for img in imgs:
        del img
    return prob_map


def set_dp_true_state_arr(dp, lit_map):
    if TRUE_STATE_ARR_KEY in dp:
        return  # skip

    ts = dp[TRUE_STATE_KEY]
    ts_arr = np.zeros(len(lit_map))
    for lit in ts.literals:
        ts_arr[lit_map[str(lit)]] = 1
    dp[TRUE_STATE_ARR_KEY] = ts_arr


def set_dp_est_state_arr(dp, se, lit_map, seed):
    if ESTIMATED_STATE_ARR_PROB_KEY in dp and seed in dp[ESTIMATED_STATE_ARR_PROB_KEY]:
        return  # skip

    prob_map = predict_dp_state(dp, se)

    es_array = np.zeros(len(lit_map))
    for lit_str in prob_map:
        if lit_str not in lit_map:
            continue
        es_array[lit_map[lit_str]] = prob_map[lit_str]

    dp.setdefault(ESTIMATED_STATE_ARR_PROB_KEY, {})[seed] = es_array


def process_datapoints(
    data_dir,
    domain,
    problem,
    seed=42,
    se_class=None,
    **se_kwargs,
):
    # these modules import heavy packages.
    # import here to avoid waiting when calling with `--help`
    from semantic_state_estimator.utils.misc import (
        load_from_entrypoint,
        set_random_seed,
    )
    from semantic_state_estimator.semantic_state_estimator import (
        SemanticStateEstimator,
        SemanticStateEstimatorWithLLaMA,
    )

    # set random seed for reproducibility
    set_random_seed(seed)

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
    lit_map = get_lit_map(domain, problem, data_dir)

    # load datapoints
    datapoints = sorted(glob.glob(f"{data_dir}/*.pkl"))

    # process all datapoints
    for dp_file in tqdm(datapoints):
        # load datapoint
        with open(dp_file, "rb") as f:
            dp = pickle.load(f)

        # process datapoint
        set_dp_true_state_arr(dp, lit_map)
        set_dp_est_state_arr(dp, se, lit_map, seed)

        # save processed datapoint
        with open(dp_file, "wb") as f:
            pickle.dump(dp, f)


if __name__ == "__main__":
    fire.Fire(process_datapoints)
