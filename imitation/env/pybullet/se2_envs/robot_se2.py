# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import os
import numpy as np

from ..utils import add_obstacle, Debug_Joint_Slider


class SE2Bot:
    def __init__(
        self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0, 0, -0.65],
            JOINT_SLIDER = False, objs_info = None
    ):
        ## Arguments
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "torque"
        self.qlimits = [[-2.96, 2.96], [-2.96, 2.96], [-3.96, 3.96]]
        self.JOINT_ID = [1, 3, 5]
        self.dof = 3
        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []

        ## Control Gains
        self.position_control_gain_p = [
            0.01,
            0.01,
            0.01
        ]
        self.position_control_gain_d = [
            1.0,
            1.0,
            1.0
        ]
        f_max = 250
        self.max_torque = [
            f_max,
            f_max,
            f_max
        ]

        ## Connect Pybullet
        p.connect(p.GUI_SERVER, 1234,
                  options='--background_color_red=.99 --background_color_green=.99 --background_color_blue=0.60')
        p.resetDebugVisualizerCamera(cameraDistance= 2., cameraYaw=-180., cameraPitch=269., cameraTargetPosition = [0., 0., 0.])
        p.setGravity(0,0,-9.8)

        # load robot_model
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # #p.setAdditionalSearchPath(current_dir + "/robot_model")

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        ## URDF FILE ##
        base_dir = os.path.abspath(__file__ + '/../../../../../')
        robot_dir = os.path.join(base_dir, 'assets/robot/se2_bot_description/robot')
        robot_file = 'robot.urdf'
        urdf_file = os.path.join(robot_dir, robot_file)

        self.robot = p.loadURDF(
            urdf_file,
            useFixedBase=True,
            #flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )

        self._base_position = [
            -0.05 - base_shift[0],
            0.0 - base_shift[1],
            -0.65 - base_shift[2],
        ]


        ## Debugger ##
        self.SLIDER_ON = JOINT_SLIDER
        self.q_slider = Debug_Joint_Slider(limits=self.qlimits, p=p, JOINT_ID=self.JOINT_ID, robot=self.robot)
        p.addUserDebugLine([0., 0., -0.189], [1.5, 0., -0.189], [1., 0., 0.])
        ##############

        ## initial q
        self.initial_q = [-1.285, 0.0, -1.285]
        self.target_pos = self.initial_q

        ## Obstacle
        self.objects = []
        self.objects_sizes = []
        if objs_info is None:
            self._obstacle_loader()
        else:
            self._obstacle_loader(objs_info)

        ## End-Effector
        self.ee_link = 6

    def _obstacle_loader(self, objects_info = [[0.5], [[1., 1., 0.]]]):
        for size_i, obst_pos_i in zip(objects_info[0], objects_info[1]):
            obst_i = add_obstacle(obs_state=obst_pos_i, size=size_i)
            self.objects.append(obst_i)
            self.objects_sizes.append(size_i)

    def reset(self, joints=None):
        self.t = 0.0
        self.control_mode = "torque"

        if joints is None:
            joints = self.initial_q

        joints = list(joints)
        for idx, j in enumerate(self.JOINT_ID):
            p.resetJointState(self.robot, j, targetValue=joints[idx])

        self.setTargetPositions(joints)
        return [self.getJointStates(), self._get_obstacles_pose()]

    def step(self, a=None):
        self.set_action(a)
        self.t += self.stepsize
        p.stepSimulation()
        return [self.getJointStates(),  self._get_obstacles_pose()]

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception("wrong control mode")

    def set_action(self,a = None):
        if self.SLIDER_ON:
                q = self.q_slider.read()
                self.setTargetPositions(q)
        else:
            if a is None:
                a = self.q
            self.setTargetPositions(a)

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        for idx, j in enumerate(self.JOINT_ID):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.target_pos[idx],
                force=self.max_torque[idx],
                positionGain=self.position_control_gain_p[idx],
                velocityGain=self.position_control_gain_d[idx],
            )

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.JOINT_ID)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        self.q = joint_pos
        self.dq = joint_vel

        return joint_pos, joint_vel

    def _get_obstacles_pose(self):
        objects_H = []
        for obj_i in self.objects:
            pos_rot = p.getBasePositionAndOrientation(obj_i)

            pos = np.array([pos_rot[0]])
            objects_H.append(pos[0,...])

        return objects_H, self.objects_sizes


if __name__ == "__main__":
    robot = SE2Bot(realtime=1, JOINT_SLIDER=True)
    robot.reset()
    robot.setControlMode("position")
    while(True):
        s = robot.step()
