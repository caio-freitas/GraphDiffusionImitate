import os
import pybullet as p
from scipy.spatial.transform import Rotation
from plan_lib.utils import to_numpy



def plot_tf_in_bullet(H, torch=True):
    if torch:
        x = to_numpy(H[:3,-1])
        R  = to_numpy(H[:3,:3])
    else:
        x = H[:3, -1]
        R = H[:3, :3]

    x = list(x)
    r = Rotation.from_matrix(R)
    quat = r.as_quat()

    base_dir = os.path.abspath(__file__+'/../../..')
    tf_file  = os.path.join(base_dir,'assets','visuals','tf.urdf')
    tf_id = p.loadURDF(tf_file,
                    basePosition=x,
                    baseOrientation=quat,
                    useFixedBase=True)
    return tf_id

def change_base_position(H, id, torch=True):
    if torch:
        x = to_numpy(H[:3,-1])
        R  = to_numpy(H[:3,:3])
    else:
        x = H[:3, -1]
        R = H[:3, :3]

    x = list(x)
    r = Rotation.from_matrix(R)
    quat = r.as_quat()

    p.resetBasePositionAndOrientation(id,
                    posObj=x,
                    ornObj=quat)
