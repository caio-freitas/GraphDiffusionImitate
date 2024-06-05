import torch
import os
from scipy.spatial.transform import Rotation as R


from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda

def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device):
    if isinstance(x, list):
        return torch.Tensor(x).float().to(device)
    else:
        return torch.from_numpy(x).float().to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def calculate_panda_joints_positions(joints): # TODO generalize for other robots
    robot_fk = DifferentiableFrankaPanda()
    q = torch.tensor([joints]).to("cpu")
    q.requires_grad_(True)
    data = robot_fk.compute_forward_kinematics_all_links(q)
    data = data[0]
    joint_positions = []
    # add joint positions
    for i in range(9):
        joint_quat = torch.tensor(R.from_matrix(data[i, :3, :3].detach().numpy()).as_quat())
        joint_positions.append(torch.cat([data[i, :3, 3], joint_quat]).reshape(1,-1))

    return torch.cat(joint_positions, dim=0)
