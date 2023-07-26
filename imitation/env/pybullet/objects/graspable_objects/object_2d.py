import os.path
import numpy as np
from imitation.utils import xyz_quat_2_H, quat_2_rot, rot2quat_np

import pybullet as p



base_dir = os.path.abspath(__file__+'/../../../../')
Obj_Root = os.path.abspath(os.path.join(base_dir, '../..', 'data/examples'))

class GraspableObject2D():
    def __init__(self, obj_type, obj_root = Obj_Root, n_grasps = 30, base_pos_world= [[0,0,0], [0,0,0,1]]):

        self.obj_type = obj_type
        self.n_grasp = n_grasps

        ## Object Properties ##
        self.mass = 1.
        self.mesh_scale = 1.

        ## Objects Properties ##
        if obj_type == 'cube':
            self.params = {
                'obj_shape':[0.1, 0.1, 0.1]
            }

        ## Environment Variables ##
        self.base_pos_world = base_pos_world[0]
        self.base_ori_world = base_pos_world[1]
        self.body_id = 0

        ## Grasp Set ##
        self.all_grasp_set_rel  = self._compute_grasp_poses()
        self.load_in_pybullet()
        self._compute_abs_grasps()

    def _compute_grasp_poses(self):
        if self.obj_type == 'cube':
            # Face 1
            x = -self.params['obj_shape'][0] * np.ones(int(self.n_grasp/4))
            y = np.linspace(-self.params['obj_shape'][1], self.params['obj_shape'][1], int(self.n_grasp/4))
            H1 = np.tile(np.eye(4).reshape(1,4,4), (int(self.n_grasp/4), 1, 1))
            H1[:,0,-1] = x
            H1[:,1,-1] = y

            #Face 2
            x = self.params['obj_shape'][0] * np.ones(int(self.n_grasp/4))
            y = np.linspace(-self.params['obj_shape'][1], self.params['obj_shape'][1], int(self.n_grasp/4))
            H2 = np.tile(np.eye(4).reshape(1,4,4), (int(self.n_grasp/4), 1, 1))
            H2[:,0,-1] = x
            H2[:,1,-1] = y
            quat = np.tile(np.array([[0,0,0,1]]), (int(self.n_grasp/4),1))
            R = quat_2_rot(quat)
            H2[:, :3, :3] = R

            #Face 3
            y = -self.params['obj_shape'][1] * np.ones(int(self.n_grasp/4))
            x = np.linspace(-self.params['obj_shape'][0], self.params['obj_shape'][0], int(self.n_grasp/4))
            H3 = np.tile(np.eye(4).reshape(1,4,4), (int(self.n_grasp/4), 1, 1))
            H3[:,0,-1] = x
            H3[:,1,-1] = y
            quat = np.tile(np.array([[0.707,0,0,0.707]]), (int(self.n_grasp/4),1))
            R = quat_2_rot(quat)
            H3[:, :3, :3] = R

            #Face 4
            y = self.params['obj_shape'][1] * np.ones(int(self.n_grasp / 4))
            x = np.linspace(-self.params['obj_shape'][0], self.params['obj_shape'][0], int(self.n_grasp / 4))
            H4 = np.tile(np.eye(4).reshape(1,4,4), (int(self.n_grasp/4), 1, 1))
            H4[:,0,-1] = x
            H4[:,1,-1] = y
            quat = np.tile(np.array([[0.707, 0., 0., -0.707]]), (int(self.n_grasp/4),1))
            R = quat_2_rot(quat)
            H4[:, :3, :3] = R

            return np.concatenate((H1,H2,H3,H4), 0)

    def load_in_pybullet(self):
        obst_collision = -1
        obst_visual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=self.params['obj_shape'])

        obj_id = p.createMultiBody(baseMass=0.0,
                                 baseCollisionShapeIndex=obst_collision,
                                 baseVisualShapeIndex=obst_visual,
                                 basePosition=self.base_pos_world,
                                 baseOrientation = self.base_ori_world)
        self.body_id  = obj_id

        return obj_id

    def compute_center_pose(self):
        pos_rot = p.getBasePositionAndOrientation(self.body_id)

        pos = np.array([pos_rot[0]])
        q_rot = np.array([[pos_rot[1][3], pos_rot[1][0], pos_rot[1][1], pos_rot[1][2]]])

        self.H_center = xyz_quat_2_H(pos=pos, quat=q_rot)[0,...]

        return self.H_center

    def _compute_abs_grasps(self):
        self.compute_center_pose()
        Hw = np.einsum('dm,bmi->bdi',self.H_center, self.all_grasp_set_rel)
        self.all_grasp_abs = Hw
        return Hw

    def set_pose(self, H):
        quat = rot2quat_np(H[:3, :3])
        quat = [quat[1], quat[2], quat[3], quat[0]]
        xyz  = H[:3,-1]
        p.resetBasePositionAndOrientation(self.body_id, posObj= xyz, ornObj=quat)


if __name__ == "__main__":
    obj_type = 'cube'
    obj = GraspableObject2D(obj_type)



