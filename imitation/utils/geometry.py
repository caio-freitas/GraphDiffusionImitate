import numpy as np
import torch

def invert_H(H):
    dim = H.shape[0]
    if isinstance(H, np.ndarray):
        Hout = np.eye(dim)

    elif isinstance(H, torch.Tensor):
        Hout = torch.eye(dim).to(H)

    Hout[:-1,:-1] = H[:-1,:-1].T
    Hout[:-1,-1] = -H[:-1, :-1].T@H[:-1,-1]
    return Hout

def invert_batch_H(H):
    Hout = torch.eye(4).view((1,4,4)).repeat((H.shape[0],1,1)).float()


    Hout[..., :-1,:-1] = H[..., :-1, :-1].transpose_(-2,-1)
    Hout[...,:-1,-1] = -torch.einsum('bmn,bn->bm',Hout[..., :-1,:-1], H[..., :-1, -1])
    return Hout

def xyz_quat_2_H(pos=np.array([[0,0,0]]), quat = np.array([[0,0,0,1]])):
    if pos.ndim !=2:
        print('Not proper shape input. Data input should be Batch Times 3 for pose and batch times 4 for quaternion')
        return False
    H = np.tile(np.eye(4),(pos.shape[0],1,1))
    H[:,:3,-1] = pos
    H[:,:3,:3] = quat_2_rot(quat)
    return H

def quat_2_rot(q=np.array([[0,0,0,1]])):
    r11 = q[:,0]**2 + q[:,1]**2 - q[:,2]**2 - q[:,3]**2
    r12 = 2*(q[:,1]*q[:,2] + q[:,0]*q[:,3])
    r13 = 2*(q[:,1]*q[:,3] - q[:,0]*q[:,2])

    r21 = 2*(q[:,1]*q[:,2] - q[:,0]*q[:,3])
    r22 = q[:,0]**2 - q[:,1]**2 + q[:,2]**2 - q[:,3]**2
    r23 = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])

    r31 = 2*(q[:,1]*q[:,3] + q[:,0]*q[:,2])
    r32 = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    r33 = q[:,0]**2 - q[:,1]**2 - q[:,2]**2 + q[:,3]**2

    rotMat = np.zeros((q.shape[0],3,3))
    rotMat[:,0,0] = r11
    rotMat[:,1,0] = r12
    rotMat[:,2,0] = r13
    rotMat[:,0,1] = r21
    rotMat[:,1,1] = r22
    rotMat[:,2,1] = r23
    rotMat[:,0,2] = r31
    rotMat[:,1,2] = r32
    rotMat[:,2,2] = r33
    return rotMat

def rot2quat_torch(R):
    qw = torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return torch.tensor([qw, qx, qy, qz]).to(R)

def rot2quat_np(R):
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])

