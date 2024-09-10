import os
from semantic_state_estimator.eval.process_datapoints import process_datapoints
from semantic_state_estimator.semantic_state_estimator import SemanticEstimatorMultiImageRun
from semantic_state_estimator.utils.llava_next_utils import DEFAULT_IMAGE_TOKEN
from semantic_state_estimator.constants import LLAVA_72B_OV

from PIL import Image


if __name__ == "__main__":
    milk_img = Image.open('grip-milk-full.png')

    THIS_DIR = os.path.dirname(__file__)

    process_datapoints(
        'data_dir',
        domain='domain.pddl',
        problem='problem.pddl',
        out_dir='llama-llava-iter-images-with-milk-instruct-72B',
        se_class=SemanticEstimatorMultiImageRun,
        vqa_model_id=LLAVA_72B_OV,
        additional_instructions=f"Here is an example of the milk carton being gripped by the robot: {DEFAULT_IMAGE_TOKEN}.",
        additional_images=[milk_img]
    )

    # process_datapoints(
    #     'data_dir',
    #     domain=os.path.join(THIS_DIR, 'domain.pddl'),
    #     problem=os.path.join(THIS_DIR, 'problem.pddl'),
    #     out_dir='llama-llava-iter-images-with-milk-instruct',
    #     se_class=SemanticEstimatorMultiImageRun,
    #     additional_instructions=f"Here is an example of the milk carton being gripped by the robot: {DEFAULT_IMAGE_TOKEN}.",
    #     additional_images=[milk_img]
    # )
