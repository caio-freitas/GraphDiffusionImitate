import os.path
import numpy as np

import pybullet as p
import trimesh
import trimesh.path
import trimesh.transformations as tra
import h5py


base_dir = os.path.abspath(__file__+'/../../../../')
Obj_Root = os.path.abspath(os.path.join(base_dir, '../..', 'data/examples'))

class GraspableObject():
    def __init__(self, obj_type, obj_root = Obj_Root, n_grasps = 100, base_pos_world= [[0,0,0], [0,0,0,1]]):

        self.n_grasp = n_grasps

        self.f = os.path.join(obj_root, obj_type)
        obj_file = h5py.File(self.f)
        self.obj_file = obj_file
        mesh_file = obj_file['/object/file'][()].decode('utf-8')
        self.obj_mesh_path = os.path.join(obj_root, mesh_file)

        ## Object Properties ##
        self.mass = obj_file["object/mass"][()]
        self.mesh_scale = obj_file["object/scale"][()]

        ## Object Mesh ##
        obj_mesh = trimesh.load(self.obj_mesh_path)
        self.obj_mesh = obj_mesh.apply_scale(self.mesh_scale)

        ## Environment Variables ##
        self.base_pos_world = base_pos_world[0]
        self.base_ori_world = base_pos_world[1]
        self.body_id = 0

        ## Grasp Set ##
        self.all_grasp_set = np.array(obj_file["grasps/transforms"])
        success_idx = np.where(np.array(obj_file["grasps/qualities/flex/object_in_gripper"])==1)
        self.success_grasp_set = self.all_grasp_set[success_idx]

    def load_in_pybullet(self):
        # self.obj_bullet = p.loadURDF(self.obj_mesh_path, basePosition=self.base_pos_world)
        shift = [0, .0, 0]
        meshScale = [self.mesh_scale]*3
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=self.obj_mesh_path,
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=self.obj_mesh_path,
                                                  collisionFramePosition=shift,
                                                  meshScale=meshScale)

        self.body_id = p.createMultiBody(baseMass=self.mass,
                          baseInertialFramePosition=[0, 0, 0],
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=self.base_pos_world,
                          useMaximalCoordinates=True)


if __name__ == "__main__":
    mesh_path = 'grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5'
    obj = GraspableObject(mesh_path)



