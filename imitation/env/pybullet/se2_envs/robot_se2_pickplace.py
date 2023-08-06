# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import os
import numpy as np

from .robot_se2 import SE2Bot

from ..objects import GraspableObject2D
from imitation.utils import xyz_quat_2_H, invert_H


class SE2BotPickPlace(SE2Bot):
    def __init__(
        self, objects_list = ['cube'], obj_poses = [[[1., 0., 0.], [0,0,0,1]]]
    ):
        super(SE2BotPickPlace, self).__init__(objs_info = [[],[]])

        self.obj_list = objects_list
        self.obj_poses = obj_poses
        self.grasp_objects = []
        self._objects_loader(self.obj_list, self.obj_poses)

        ## Environment variables
        self._grasped = False

    def _objects_loader(self, objects_list, base_positions):
        ## Load Objects to Environment ##
        for object_i, base_pos_i in zip(objects_list,base_positions):
            grasp_obj_i = GraspableObject2D(obj_type=object_i, base_pos_world=base_pos_i)
            grasp_obj_i.load_in_pybullet()
            self.grasp_objects.append(grasp_obj_i)

    def reset(self, joints=None):
        grasp_obj = self._get_grasp_objs_pose()
        robot, obstacles = super().reset(joints)
        return [robot, obstacles, grasp_obj]

    def step(self, a=np.zeros(9)):
        robot, obstacles = super().step(a)
        self._move_obj()
        grasp_obj = self._get_grasp_objs_pose()
        return [robot, obstacles, grasp_obj]

    def setControlMode(self, mode):
        return super().setControlMode(mode)

    def _get_grasp_objs_pose(self):
        objects_H = []
        for obj_i in self.grasp_objects:
            Hi = obj_i.compute_center_pose()
            objects_H.append(Hi)
        return objects_H

    def get_grasp_set(self):
        return self.grasp_objects[0].all_grasp_abs

    def set_grasp(self, grasp=True):
        self._grasped = grasp
        H_obj = self._get_grasp_objs_pose()[0]

        H_ee = self.get_ee_pose()
        H_ee_inv = invert_H(H_ee)
        self.H_ee_obj = H_ee_inv@H_obj

    def _move_obj(self):
        if self._grasped:
            Hee = self.get_ee_pose()
            Hobj_t = Hee@self.H_ee_obj
            print(Hobj_t)
            self.grasp_objects[0].set_pose(Hobj_t)
        else:
            pass

    def get_ee_pose(self):
        pos, rot, *_ = p.getLinkState(self.robot, 6)
        pos = np.array(pos)
        q_rot = np.array([rot[3], rot[0], rot[1], rot[2]])
        return xyz_quat_2_H(pos=pos[None,:], quat=q_rot[None,:])[0,...]


if __name__ == "__main__":

    robot = SE2BotPickPlace()
    s = robot.reset()

    grasps = robot.grasp_objects[0].all_grasp_abs

    robot.setControlMode("position")
    while(True):
        s = robot.step()

