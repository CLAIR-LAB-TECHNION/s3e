from argparse import Namespace
import os
from tempfile import TemporaryDirectory
import numpy as np
import json

# from imageio.v3 import imread

from photorealistic_blocksworld.blocks import (
    State,
    load_colors,
    Block,
    random_dict,
    random,
    properties,
)
from photorealistic_blocksworld.render_utils import render_scene
import numpy as np

try:
    import bpy
except ImportError:
    print("no blender detected. cannot render")


PRB_DIR = os.path.join(os.path.dirname(__file__), "photorealistic_blocksworld")
PROBLEM_TEMPLATE = """(define (problem shape-stacking)
    (:domain shape-stacking)
    (:objects
        {objects} - block
    )

    (:init)
    (:goal (and))
)
"""

COLOR_TO_NAME = {
    tuple(np.array([87, 87, 87]) / 255) + (1.0,): "grey",
    tuple(np.array([173, 35, 35]) / 255) + (1.0,): "red",
    tuple(np.array([42, 75, 215]) / 255) + (1.0,): "blue",
    tuple(np.array([29, 105, 20]) / 255) + (1.0,): "green",
    tuple(np.array([129, 74, 25]) / 255) + (1.0,): "brown",
    tuple(np.array([129, 38, 192]) / 255) + (1.0,): "purple",
    tuple(np.array([41, 208, 208]) / 255) + (1.0,): "cyan",
    tuple(np.array([255, 238, 51]) / 255) + (1.0,): "yellow",
}


def undump(obj):
    # a copy from photorealistic_blocksworld.blocks.py
    # need the "eval" to evaluate THIS state
    def rec(obj):
        if isinstance(obj, dict):
            if "__class__" in obj:
                cls = eval(obj["__class__"])
                res = cls.__new__(cls)
                for k, v in obj.items():
                    if k != "__class__":
                        vars(res)[k] = rec(v)
                return res
            else:
                return {k: rec(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return list(rec(v) for v in obj)
        else:
            return obj

    return rec(obj)


def props_to_block_name(size, color, mat, shape):
    return f"{size}-{color}-{mat}-{shape}"


class NewBlock(Block):
    def __init__(self, i):
        self.shape_name, self.shape = random_dict(properties["shapes"])
        self.color = random.choice(properties["colors"])
        self.color_name = COLOR_TO_NAME[tuple(self.color)]
        self.size_name, self.size = random_dict(properties["sizes"])
        self.mat_name, self.material = random_dict(properties["materials"])
        self.rotation = 360.0 * random.random()
        self.stackable = properties["stackable"][self.shape_name] == 1
        self.location = [0, 0, 0]
        self.id = i
        pass

    def get_block_name(self):
        return props_to_block_name(
            self.size_name, self.color_name, self.mat_name, self.shape_name
        )


class NewState(State):
    def __init__(self, args):
        objects = []
        for i in range(args.num_objects):
            while True:
                o1 = NewBlock(i)
                if args.allow_duplicates:
                    break
                ok = True
                for o2 in objects:
                    if o1.similar(o2):
                        ok = False
                        print("duplicate object!")
                        break
                if ok:
                    break
            objects.append(o1)

        self.table_size = args.table_size
        self.object_jitter = args.object_jitter
        self.objects = objects
        self.shuffle()
        pass

    def get_all_objects_str(self, sep=" "):
        return sep.join(map(lambda o: o.get_block_name(), self.objects))

    def get_object_by_name(self, name):
        for o in self.objects:
            if o.get_block_name() == name:
                return o
        return None

    @staticmethod
    def undump(data):
        return undump(data)


def load_with_bpy_to_numpy_rgb(image_path):
    image = bpy.data.images.load(image_path)  # load as flat RGBA image
    width, height = image.size

    image_arr = np.array(image.pixels[:])  # convert to flat numpy array
    image_arr = image_arr.reshape((height, width, 4))  # reshape to image size (RGBA)
    image_arr = image_arr[:, :, :3]  # remove alpha dimension
    image_arr = (image_arr * 255).astype(np.uint8)  # convert to int values
    image_arr = np.flipud(image_arr)  # flip upside down image

    return image_arr


class PRBEnv:
    def __init__(
        self,
        # state args
        num_objects,
        allow_duplicates=False,
        table_size=5,
        object_jitter=0.0,
        # scene args
        base_scene_blendfile=os.path.join(PRB_DIR, "data", "base_scene.blend"),
        properties_json=os.path.join(PRB_DIR, "data", "properties.json"),
        shape_dir=os.path.join(PRB_DIR, "data", "shapes"),
        material_dir=os.path.join(PRB_DIR, "data", "materials"),
        randomize_colors=False,
        # render args
        use_gpu=0,
        width=320,
        height=240,
        key_light_jitter=0.0,
        fill_light_jitter=0.0,
        back_light_jitter=0.0,
        camera_jitter=0.0,
        render_num_samples=512,
        render_min_bounces=8,
        render_max_bounces=8,
        render_tile_size=256,
    ):
        self.args = Namespace(
            num_objects=num_objects,
            allow_duplicates=allow_duplicates,
            table_size=table_size,
            object_jitter=object_jitter,
            base_scene_blendfile=base_scene_blendfile,
            properties_json=properties_json,
            shape_dir=shape_dir,
            material_dir=material_dir,
            randomize_colors=randomize_colors,
            use_gpu=use_gpu,
            width=width,
            height=height,
            key_light_jitter=key_light_jitter,
            fill_light_jitter=fill_light_jitter,
            back_light_jitter=back_light_jitter,
            camera_jitter=camera_jitter,
            render_num_samples=render_num_samples,
            render_min_bounces=render_min_bounces,
            render_max_bounces=render_max_bounces,
            render_tile_size=render_tile_size,
        )

        load_colors(self.args)

        self.state = None

    def reset(self, num_objects=None):
        if num_objects is not None:
            self.args.num_objects = num_objects
        self.state = NewState(self.args)

    def load_state(self, scene_json):
        with open(scene_json, "r") as f:
            data = json.load(f)
        self.state = NewState.undump(data)

    def render(
        self, *args, **kwargs
    ):  # args and kwargs to swallow up unwanted arguments
        # create a random location to save the renderings
        with TemporaryDirectory() as tmpdir:
            # place genrated files in the temporary directory
            render_path = os.path.join(tmpdir, "render.png")
            scene_path = os.path.join(
                tmpdir, "scene.json"
            )  # will automatically save the scene json. we don't need it.

            # render the scene
            render_scene(
                self.args, render_path, scene_path, objects=self.state.for_rendering()
            )
            output = load_with_bpy_to_numpy_rgb(render_path)

        return output

    def get_problem_file_str(self):
        return PROBLEM_TEMPLATE.format(objects=self.state.get_all_objects_str())

    @property
    def _env(self):  # to match the use in `collect_datapoints.py`
        return self
