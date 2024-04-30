
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
                control_freq=30,  # control should happen fast enough so that simulation looks smooth
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


    def scaled_tanh(self, x, max_val=0.01, min_val=-0.07, k=200, threshold=-0.03):
        return np.tanh(k * (x - threshold)) * (max_val - min_val) / 2 + (max_val + min_val) / 2

    def control_loop(self, tgt_jpos, max_n=20, eps=0.02):
        obs = self.env._get_observations()
        tgt_jpos[-1] = self.scaled_tanh(tgt_jpos[-1])
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

    def _get_object_feats(self, data):
        # create tensor of same dimension return super()._get_node_feats(data, t) as node_feats
        obj_state_tensor = torch.zeros((self.num_objects, self.node_feature_dim))
        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        obj_state_tensor[:, -1] = self.OBJECT_NODE_TYPE
        return obj_state_tensor

    def _get_object_pos(self, data):
        obj_state_tensor = torch.zeros((self.num_objects, 9)) # 3 for position, 6 for rotation

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0
            for object_state in object_state_items:
                if "quat" in object_state:
                    assert self.object_state_sizes[object_state] == 4, "Quaternion must have size 4"
                    rot = self.rotation_transformer.forward(torch.tensor(data["object"][i:i + self.object_state_sizes[object_state]]))
                    obj_state_tensor[object,i:i + 6] = rot
                else:
                    obj_state_tensor[object,i:i + self.object_state_sizes[object_state]] = torch.from_numpy(data["object"][i:i + self.object_state_sizes[object_state]])

                i += self.object_state_sizes[object_state]

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

    

    def _get_node_feats(self, data, control_mode=None):
        '''
        Returns node features from data
        '''
        if control_mode is None:
            control_mode = self.control_mode
        node_feats = []
        for i in range(self.num_robots):
            if control_mode == "OSC_POSE":
                node_feats.append(torch.cat([torch.tensor(data[f"robot{i}_eef_pos"]), torch.tensor(data[f"robot{i}_eef_quat"])], dim=0).reshape(1, -1)) # add dimension
            elif control_mode == "JOINT_VELOCITY":
                node_feats.append(torch.tensor([*data[f"robot{i}_joint_vel"], *data[f"robot{i}_gripper_qvel"]]).reshape(1,-1).T)
            elif control_mode == "JOINT_POSITION":
                node_feats.append(torch.tensor([*data[f"robot{i}_joint_pos"], *data[f"robot{i}_gripper_qpos"]]).reshape(1,-1).T)
        node_feats = torch.cat(node_feats, dim=0)
        return node_feats


    def _get_y_feats(self, data):
        '''
        Returns observation node features from data
        '''
        y = []
        for i in range(self.num_robots):
            y.append(torch.tensor([*data[f"robot{i}_joint_pos"], *data[f"robot{i}_gripper_qpos"]]).reshape(1,-1).T)
        y = torch.cat(y, dim=0)
        obj_state_tensor = self._get_object_pos(data)
        # remove positions
        obj_state_tensor = obj_state_tensor[:,3:]

        # add dimensions to match with obj_state_tensor for concatenation
        y = torch.cat([y, torch.zeros((y.shape[0], obj_state_tensor.shape[1]-y.shape[1]))], dim=1) 
        y = torch.cat([y, obj_state_tensor], dim=0)

        # add node ID to node features
        node_id = torch.arange(y.shape[0]).reshape(-1, 1)
        y = torch.cat((y, node_id), dim=-1)
        return y

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
        Converts robosuite Gym Wrapper observations to the RobomimicGraphDataset format
        * requires robot_joint to be "flagged" in robomimic environment
        '''
        node_feats = torch.tensor([])
        node_pos = torch.tensor([])
        node_obs = torch.tensor([])
        robot_i_data = {}
        for i in range(self.num_robots):
            j = i*39
            
            # 7  - joint angle values
            robot_joint_pos = obs[j:j + 7]
            # 7  - sin of joint angles
            # robot_joint_sin = obs[j + 7:j + 14]
            # 7  - cos of joint angles
            # robot_joint_cos = obs[j + 14:j + 21]
            # 7  - joint velocities
            robot_joint_vel = obs[j + 21:j + 28]
            eef_pose = obs[j + 28:j + 31]
            eef_quat = obs[j + 31:j + 35]
            eef_6d = self.rotation_transformer.forward(eef_quat)
            gripper_pose = obs[j + 35:j + 37]
            gripper_vel = obs[j + 37:j + 39]
            # Skip 2  - gripper joint velocities
            robot_i_data.update({
                f"robot{i}_joint_pos": robot_joint_pos,
                f"robot{i}_joint_vel": robot_joint_vel,
                f"robot{i}_eef_pos": eef_pose,
                f"robot{i}_eef_quat": eef_6d,
                f"robot{i}_gripper_qpos": gripper_pose,
                f"robot{i}_gripper_qvel": gripper_vel
            })
        robot_i_data["object"] = obs[self.num_robots*39:]

        node_feats = torch.cat([node_feats, self._get_node_feats(robot_i_data)], dim=0)
        
        node_pos = self._get_node_pos(robot_i_data)
        y = torch.cat([node_obs, self._get_y_feats(robot_i_data)], dim=0)

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)
        
        obj_feats_tensor = self._get_object_feats(obs)
        
        node_feats = torch.cat((node_feats, obj_feats_tensor), dim=0)

        edge_index = self._get_edge_index(node_feats.shape[0])
        edge_attrs = self._get_edge_attrs(edge_index)        

        # create graph
        graph = torch_geometric.data.Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs, y=y, pos=node_pos)

        return graph
    

    def reset(self):
        obs, _ =  self.env.reset()
        return self._robosuite_obs_to_robomimic_graph(obs)
    

    def step(self, action):
        final_action = []
        for i in range(self.num_robots):
            '''
            Robosuite's action space is composed of 7 joint velocities and 1 gripper velocity, while 
            in the robomimic datasets, it's composed of 7 joint velocities and 2 gripper velocities (for each "finger").
            '''
            j = i*9
            robot_joint_pos = action[j:j + 7]
            robot_gripper_pos = action[j + 8]
            final_action = [*final_action, *robot_joint_pos, robot_gripper_pos]
        if self.control_mode == "JOINT_VELOCITY":
            obs, reward, done, _, info = self.env.step(final_action)
        elif self.control_mode == "JOINT_POSITION":
            obs, reward, done, _, info = self.control_loop(final_action)
        else:
            raise ValueError("Invalid control mode")
        
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