
import gymnasium as gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from scipy.spatial.transform import Rotation as R
import torch
import torch_geometric
from functools import lru_cache
import logging
from tqdm import tqdm

from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from imitation.utils.generic import calculate_panda_joints_positions

log = logging.getLogger(__name__)
 
class RobomimicGymWrapper(GymWrapper):
    # override the default observation keys
    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    log.info("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)


class RobomimicGraphWrapper(gym.Env):
    def __init__(self,
                 object_state_keys,
                 object_state_sizes,
                 node_feature_dim = 2,
                 max_steps=5000,
                 task="Lift",
                 has_renderer=True,
                 robots=["Panda"],
                 output_video=False,
                 control_mode="JOINT_VELOCITY",
                 controller_config=None,
                 base_link_shift=[0.0, 0.0, 0.0],
                 base_link_rotation=[[0.0, 0.0, 0.0, 1.0]]
                 ):
        '''
        Environment wrapper for Robomimic's GraphDiffusionImitate dataset in the same Graph representation as 
        in the RobomimicGraphDataset class.
        '''
        self.object_state_sizes = object_state_sizes
        self.object_state_keys = object_state_keys
        self.node_feature_dim = node_feature_dim
        self.control_mode = control_mode
        controller_config = load_controller_config(default_controller=self.control_mode)
        # override default controller config with user-specified values
        for key in controller_config.keys():
            controller_config[key] = controller_config[key] if key in controller_config else controller_config[key]
        self.robots = [*robots] 
        self.num_robots = len(robots)
        keys = [ "robot0_proprio-state", 
                *[f"robot{i}_proprio-state" for i in range(1, self.num_robots)],
                "object-state"]
        self.has_renderer = has_renderer
        self.env = RobomimicGymWrapper(
            suite.make(
                task,
                robots=self.robots,
                use_camera_obs=output_video,  # do not use pixel observations
                has_offscreen_renderer=output_video,  # not needed since not using pixel obs
                has_renderer=has_renderer,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # must match dataset recording frequency (20 Hz for lift/ph)
                horizon=max_steps,  # long horizon so we can sample high rewards
                controller_configs=controller_config,
            ),
            keys = keys
        )
        self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_objects = len(object_state_keys)

        self.NUM_GRAPH_NODES = self.num_robots*9 + self.num_objects # TODO add multi-robot support
        self.BASE_LINK_SHIFT = base_link_shift
        self.BASE_LINK_ROTATION = base_link_rotation
        self.ROBOT_NODE_TYPE = 1
        self.OBJECT_NODE_TYPE = -1

        self.ROBOT_LINK_EDGE = 1
        self.OBJECT_ROBOT_EDGE = 2
        self.rotation_transformer = RotationTransformer(
            from_rep="quaternion",
            to_rep="rotation_6d"
        )
        self.eef_idx = [-1, 8] # end-effector index
        if self.num_robots == 2:
            self.eef_idx += [17]
        # Initialise the last action buffer (used by _get_node_feats for OSC_POSE).
        # For OSC_POSE, actions are 7D: [dx, dy, dz, ax, ay, az, gripper]
        self._last_osc_action = np.zeros(7 * self.num_robots, dtype=np.float32)


    def scaled_tanh(self, x, max_val=0.01, min_val=-0.07, k=200, threshold=-0.03):
        return np.tanh(k * (x - threshold)) * (max_val - min_val) / 2 + (max_val + min_val) / 2

    def control_loop(self, tgt_jpos, max_n=20, eps=0.02):
        obs = self.env._get_observations()
        # tgt_jpos[-1] = self.scaled_tanh(tgt_jpos[-1])
        for i in range(max_n):
            obs = self.env._get_observations()
            current_jpos = []
            for j in range(self.num_robots):
                current_jpos = [*current_jpos, *obs[f"robot{j}_joint_pos"], obs[f"robot{j}_gripper_qpos"][1]] # use only last action for gripper
            q_diff = np.array(tgt_jpos) - current_jpos
            q_diff_max = np.max(abs(q_diff))
            
            action = list(q_diff)
            assert len(action) == 8*self.num_robots, len(action)
            obs_final, reward, done, _, info = self.env.step(action)
            if q_diff_max < eps or done:
                break
            if self.has_renderer:
                self.env.render()
        return obs_final, reward, done, _, info
    

    def _get_object_pos(self, data):
        obj_state_tensor = torch.zeros((self.num_objects, 9)) # 3 for position, 6 for rotation
        obj_buf = data["object"]
        buf_len = len(obj_buf)

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0       # offset into the flat object buffer
            out_col = 0 # column in obj_state_tensor (quat->6d expands by 2)
            for object_state in object_state_items:
                field_size = self.object_state_sizes[object_state]
                if i + field_size > buf_len:
                    # Field not present in this observation buffer — leave as zeros
                    out_col += 6 if "quat" in object_state else field_size
                    i += field_size
                    continue
                if "quat" in object_state:
                    assert field_size == 4, "Quaternion must have size 4"
                    rot = self.rotation_transformer.forward(
                        torch.tensor(obj_buf[i:i + field_size], dtype=torch.float32)
                    )
                    obj_state_tensor[object, out_col:out_col + 6] = rot
                    out_col += 6
                else:
                    obj_state_tensor[object, out_col:out_col + field_size] = torch.from_numpy(
                        obj_buf[i:i + field_size]
                    )
                    out_col += field_size
                i += field_size

        return obj_state_tensor


    def _get_node_pos(self, data):
        node_pos = []
        for i in range(self.num_robots):
            node_pos_robot = calculate_panda_joints_positions([*data[f"robot{i}_joint_pos"], *data[f"robot{i}_gripper_qpos"]])
            rotation_matrix = R.from_quat(self.BASE_LINK_ROTATION[i])
            node_pos_robot[:,:3] = torch.matmul(node_pos_robot[:,:3], torch.tensor(rotation_matrix.as_matrix()))
            node_pos_robot[:,3:] = torch.tensor((R.from_quat(node_pos_robot[:,3:].detach().numpy()) * rotation_matrix).as_quat())
            # add base link shift
            node_pos_robot[:,:3] += torch.tensor(self.BASE_LINK_SHIFT[i])
            node_pos.append(node_pos_robot)
        node_pos = torch.cat(node_pos, dim=0)
        # use rotation transformer to convert quaternion to 6d rotation
        node_pos = torch.cat([node_pos[:,:3], self.rotation_transformer.forward(node_pos[:,3:])], dim=1)
        obj_pos_tensor = self._get_object_pos(data)
        node_pos = torch.cat((node_pos, obj_pos_tensor), dim=0)
        return node_pos


    def _get_x_feats(self, data):
        '''
        Returns observation node features from data.
        Output shape: (num_nodes, obs_feat_dim) where obs_feat_dim includes node_type.
        '''
        x = []
        for i in range(self.num_robots):
            x.append(torch.tensor([*data[f"robot{i}_joint_pos"], *data[f"robot{i}_gripper_qpos"]],
                                  dtype=torch.float32).reshape(-1, 1))  # (9, 1)
        x = torch.cat(x, dim=0)  # (9, 1)
        obj_state_tensor = self._get_object_pos(data)

        # pad robot features to match object feature width for concatenation
        x = torch.cat([x, torch.zeros((x.shape[0], obj_state_tensor.shape[1] - x.shape[1]))], dim=1)  # (9, 9)
        x = torch.cat([x, obj_state_tensor], dim=0)  # (10, 9)

        return x

    @lru_cache(maxsize=128)
    def _get_edge_index(self, num_nodes):
        '''
        Returns edge index for graph.
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        assert len(self.eef_idx) == self.num_robots + 1
        edge_index = []
        if len(self.eef_idx) == 3: # 2 robots
            edge_index = [[self.eef_idx[0]+ 1, self.eef_idx[1] + 1]] # robot0 base link to robot1 base link
        for robot in range(self.num_robots):
            # Connectivity of all robot nodes to the previous robot node
            edge_index += [[idx, idx+1] for idx in range(self.eef_idx[robot]+ 1, self.eef_idx[robot+1])]
        # Connectivity of all other nodes to all robot nodes
        edge_index += [[node_idx, idx] for idx in range(self.eef_idx[-1] + 1, num_nodes) for node_idx in range(self.eef_idx[self.num_robots] + 1)]
            # edge_index.append(torch.tensor([node_idx, idx]) for node_idx in range(self.eef_idx[self.num_robots] + 1))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    @lru_cache(maxsize=128)
    def _get_edge_attrs(self, edge_index):
        '''
        Attribute edge types to edges
        - self.ROBOT_LINK_EDGE for edges between robot nodes
        - self.OBJECT_ROBOT_EDGE for edges between robot and object nodes
        '''
        edge_attrs = []
        num_nodes = torch.max(edge_index)
        for edge in edge_index.t():
            # num nodes - self.num_objects is the index of the last robot node
            if edge[0] <= num_nodes - self.num_objects and edge[1] <= num_nodes - self.num_objects:
                edge_attrs.append(self.ROBOT_LINK_EDGE)
            # there are no object-to-object edges
            else:
                edge_attrs.append(self.OBJECT_ROBOT_EDGE)
        return torch.tensor(edge_attrs, dtype=torch.long)


    def _robosuite_obs_to_robomimic_graph(self, obs):
        '''
        Converts robosuite Gym Wrapper (robot0_proprio-state, object-state) flat
        observations into the RobomimicGraphDataset format.

        robot0_proprio-state layout (32 elements per robot):
          [0:7]   cos(joint_pos)    — 7 joint cosines
          [7:14]  sin(joint_pos)    — 7 joint sines
          [14:21] joint_vel         — 7 joint velocities
          [21:24] eef_pos           — 3D end-effector position
          [24:28] eef_quat_raw      — 4D end-effector quaternion
          [28:30] gripper_qpos      — 2 gripper finger positions
          [30:32] gripper_qvel      — 2 gripper finger velocities
        '''
        PROPRIO_SIZE = 32   # elements per robot in proprio-state
        robot_i_data = {}
        for i in range(self.num_robots):
            j = i * PROPRIO_SIZE

            # Reconstruct raw joint_pos from sin/cos (robosuite stores sin/cos, not raw)
            joint_cos = obs[j:j + 7]
            joint_sin = obs[j + 7:j + 14]
            robot_joint_pos = np.arctan2(joint_sin, joint_cos)

            robot_joint_vel  = obs[j + 14:j + 21]
            eef_pose         = obs[j + 21:j + 24]
            eef_quat_raw     = obs[j + 24:j + 28]
            eef_6d           = self.rotation_transformer.forward(
                torch.tensor(eef_quat_raw, dtype=torch.float32)
            )
            gripper_pose     = obs[j + 28:j + 30]
            gripper_vel      = obs[j + 30:j + 32]

            robot_i_data.update({
                f"robot{i}_joint_pos":    robot_joint_pos,
                f"robot{i}_joint_vel":    robot_joint_vel,
                f"robot{i}_eef_pos":      eef_pose,
                f"robot{i}_eef_quat":     eef_6d,
                f"robot{i}_gripper_qpos": gripper_pose,
                f"robot{i}_gripper_qvel": gripper_vel,
                f"_osc_action_{i}":       self._last_osc_action[i*7:(i+1)*7],
            })
        robot_i_data["object"] = obs[self.num_robots * PROPRIO_SIZE:]
        
        node_pos = self._get_node_pos(robot_i_data)
        observations = self._get_x_feats(robot_i_data)

        # Use total node count (robot + objects) for edge computation
        num_nodes = observations.shape[0]
        edge_index = self._get_edge_index(num_nodes)
        edge_attrs = self._get_edge_attrs(edge_index)        

        # create graph: x = observations, y is not set at inference
        # (the policy generates actions via diffusion, it doesn't need y)
        graph = torch_geometric.data.Data(
            x=observations, 
            edge_index=edge_index, 
            edge_attr=edge_attrs,
            pos=node_pos
        )

        return graph
    

    def reset(self):
        obs, _ =  self.env.reset()
        return self._robosuite_obs_to_robomimic_graph(obs)
    

    def step(self, action):
        if self.control_mode == "OSC_POSE":
            # OSC_POSE actions are 7-D EEF vectors — pass through directly.
            obs, reward, done, _, info = self.env.step(action)
        else:
            # JOINT_VELOCITY / JOINT_POSITION: action is a 9-D graph vector per robot
            # (7 joint DOF + 2 gripper fingers).  Robosuite expects 8-D (7 + 1 gripper).
            # Use finger 0 (index j+7) — the two fingers move symmetrically.
            final_action = []
            for i in range(self.num_robots):
                j = i * 9
                robot_joint_pos = action[j:j + 7]
                robot_gripper_pos = action[j + 7]
                final_action = [*final_action, *robot_joint_pos, robot_gripper_pos]
            if self.control_mode == "JOINT_VELOCITY":
                obs, reward, done, _, info = self.env.step(final_action)
            elif self.control_mode == "JOINT_POSITION":
                obs, reward, done, _, info = self.control_loop(final_action)
            else:
                raise ValueError(f"Invalid control mode: {self.control_mode}")
        
        if reward == 1:
            done = True
            info = {"success": True}
        else:
            info = {"success": False}
        graph_obs = self._robosuite_obs_to_robomimic_graph(obs)
        return graph_obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()