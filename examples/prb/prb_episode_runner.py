import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import socket
import time
import pickle
import subprocess
from tempfile import TemporaryDirectory

import fire
from PIL import Image
import numpy as np

from prb_env import PRBEnv, random
from prb_gt_state_estimator import PRBGTStateEstimator
from prb_skill_executer import PRBSkillExecuter
from semantic_state_estimator.constants import LLAMA_70B_INSTRUCT
from semantic_state_estimator.eval.run_episodes import EpisodeRunner
from semantic_state_estimator.utils.misc import load_se_from_args, load_from_entrypoint


DOMAIN_FILE = os.path.join(os.path.dirname(__file__), 'domain.pddl')
PROBLEM_FILE = os.path.join(os.path.dirname(__file__), 'all_objects_problem.pddl')
OUT_DIR = 'data_dir'
RENDER_CAMS = ['frontview']


def start_renderer_process():
    blender_exec = "photorealistic_blocksworld/blender-2.83.2-linux64/blender"

    # Start the renderer in a separate process to avoid memory leaks
    return subprocess.Popen(
        [blender_exec, "-noaudio", "--background", "--python", 'prb_render_server.py']
    )


def send_render_request(data):
    num_trials = 0

    # Send data to the renderer process via socket
    while num_trials < 5:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('localhost', 65432))  # Ensure this matches the renderer's settings
            except ConnectionRefusedError:
                print('Connection refused. Retrying...')
                sys.stdout.flush()
                num_trials += 1
                time.sleep(3)
                continue
            s.sendall(pickle.dumps(data))
            print('awaiting response')
            sys.stdout.flush()

            # Wait for the rendered image or confirmation
            response = s.recv(1024)

            print('got response')
            sys.stdout.flush()

            if response == b"done":
                return
            
            print(f"Render failed:\n{response}", file=sys.stderr)
            num_trials += 1

    raise Exception("Render failed too many times")



class PRBEnvWrapper(PRBEnv):
    def __init__(self, num_objects_low, num_objects_high):
        super().__init__(2)
        self.num_objects_low = num_objects_low
        self.num_objects_high = num_objects_high

        self.render_proc = start_renderer_process()

    def reset(self, *args, **kwargs):
        num_objects = random.randint(self.num_objects_low, self.num_objects_high)
        super().reset(num_objects=num_objects)

    def get_state(self):
        return self.state

    def render(self, *args, **kwargs):
        print('rendering')
        # create a random location to save the renderings
        with TemporaryDirectory() as tmpdir:
            # place genrated files in the temporary directory
            render_path = os.path.join(tmpdir, "render.png")
            scene_path = os.path.join(
                tmpdir, "scene.json"
            )  # will automatically save the scene json. we don't need it.

            # render the scene with server
            data = {
                'args': self.args,
                'output_image': render_path,
                'output_scene': scene_path,
                'objects': self.state.for_rendering()
            }
            print('sending request')
            send_render_request(data)
            
            img = Image.open(render_path)
            output = np.array(img)

        return output

    def __del__(self):
        self.render_proc.terminate()


def query_swapper(env: PRBEnvWrapper):
    with open(DOMAIN_FILE, 'r') as f:
        domain_str = f.read()

    return domain_str, env.get_problem_file_str(), LLAMA_70B_INSTRUCT


def main(run_name, num_objects_low, num_objects_high, task_horizon, se_class, **se_kwargs):
    env = PRBEnvWrapper(num_objects_low, num_objects_high)
    env.reset()

    domain_str, problem_str, _ = query_swapper(env)
    gt_se = PRBGTStateEstimator(domain_str, problem_str)
    gt_se.env = env
    exec = PRBSkillExecuter(env)

    if 'random' in se_class.lower():
        random_cls = load_from_entrypoint(se_class)
        se = random_cls(DOMAIN_FILE, PROBLEM_FILE, se_kwargs['success_rate'], gt_se)
    else:
        se = load_se_from_args(se_class, se_kwargs, DOMAIN_FILE, PROBLEM_FILE)

    try:
        runner = EpisodeRunner(
            gt_se.up_problem, gt_se, se, exec, RENDER_CAMS, OUT_DIR, run_name + f'__({num_objects_low},{num_objects_high})',
            query_swapper=query_swapper
        )

        runner.run(num_episodes=100, task_horizon=task_horizon)
    except KeyboardInterrupt:
        del env
        raise


if __name__ == "__main__":
    fire.Fire(main)
