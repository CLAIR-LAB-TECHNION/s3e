import fire
import json
import os
from semantic_state_estimator.constants import DP_FNAME_FORMAT


def generate_result_files(annotation_file, out_dir):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    os.makedirs(out_dir, exist_ok=True)
    for annotation in annotations:
        for i in range(annotation['start'], annotation['stop'] + 1):
            with open(os.path.join(out_dir, DP_FNAME_FORMAT.format(i))) as f:
                json.dump(annotation['predictions'], f, indent=4)


if __name__ == "__main__":
    fire.Fire(generate_result_files)
