import os.path
import time

import numpy as np

import pybullet as p
import trimesh
import trimesh.path
import trimesh.transformations as tra
import h5py


base_dir = os.path.abspath(__file__+'/../../../../../')
elements_root = os.path.abspath(os.path.join(base_dir, '..', 'data/scene_elements'))

class SceneElement():
    def __init__(self, obj_type='box', obj_root = elements_root, base_pos_world=[[0,0,0],[0,0,0,1]]):

        self.obj_type = obj_type
        self.obj_urdf = obj_type +'.urdf'

        self.urdf_path = os.path.join(obj_root, self.obj_urdf)

        ## Environment Variables ##
        self.base_pos_world = base_pos_world[0]
        self.base_ori_world = base_pos_world[1]
        self.body_id = 0

    def load_in_pybullet(self):
        self.body_id = p.loadURDF(
            self.urdf_path, self.base_pos_world, self.base_ori_world, useFixedBase=True)


if __name__ == "__main__":
    obj = SceneElement(base_pos_world=[[0.5,0.,0.],[0,0,0,1]])

    ## Test Load in Pybullet ##
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=.99 --background_color_green=.99 --background_color_blue=0.60')
    p.setGravity(0, 0, -9.8)

    obj.load_in_pybullet()

    while(True):
        time.sleep(1.)
