
import gymnasium as gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

import torch
import torch_geometric

import logging
from tqdm import tqdm

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
                 mode="task-joint-space"
                 ):
        '''
        Environment wrapper for Robomimic's GraphDiffusionImitate dataset in the same Graph representation as 
        in the RobomimicGraphDataset class.
        '''
        self.object_state_sizes = object_state_sizes
        self.object_state_keys = object_state_keys
        self.node_feature_dim = node_feature_dim
        controller_config = load_controller_config(default_controller="JOINT_POSITION")
        # tune up controller gains
        controller_config["kp"] = 100 # default is 50
        self.robots = [*robots] # gambiarra to make it work with robots list
        keys = [ "robot0_proprio-state", 
                *[f"robot{i}_proprio-state" for i in range(1, len(self.robots))],
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
        self.mode = mode
        self.num_objects = len(object_state_keys)

        self.NUM_GRAPH_NODES = 9 + self.num_objects # TODO add multi-robot support

        self.ROBOT_NODE_TYPE = 1
        self.OBJECT_NODE_TYPE = -1

        self.ROBOT_LINK_EDGE = 1
        self.OBJECT_ROBOT_EDGE = 2


    def scaled_tanh(self, x, max_val=0.04, min_val=-0.1, k=200, threshold=-0.03):
        return np.tanh(k * (x - threshold)) * (max_val - min_val) / 2 + (max_val + min_val) / 2

    def control_loop(self, tgt_jpos, max_n=1, eps=0.02):
        obs = self.env._get_observations()
        tgt_jpos[-1] = self.scaled_tanh(tgt_jpos[-1])
        for i in range(max_n):
            obs = self.env._get_observations()
            joint_pos = np.array([*obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][1]])  # use only last action for gripper
            q_diff = np.array(tgt_jpos) - joint_pos[:len(tgt_jpos)]
            q_diff_max = np.max(abs(q_diff))
            
            action = list(q_diff)
            assert len(action) == 8, len(action)
            obs_final, reward, done, _, info = self.env.step(action)
            if q_diff_max < eps:
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
        obj_state_tensor = torch.zeros((self.num_objects, 7)) # 3 for position, 4 for quaternion

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0
            for object_state in object_state_items:
                obj_state_tensor[object,i:i + self.object_state_sizes[object_state]] = torch.from_numpy(data["object"][i:i + self.object_state_sizes[object_state]])
                i += self.object_state_sizes[object_state]

        return obj_state_tensor


    def _get_node_pos(self, data):
        node_pos = []
        node_pos.append(calculate_panda_joints_positions([*data["robot0_joint_pos"], *data["robot0_gripper_qpos"]]))
        node_pos = torch.cat(node_pos)
        obj_pos_tensor = self._get_object_pos(data)
        node_pos = torch.cat((node_pos, obj_pos_tensor), dim=0)
        return node_pos

    

    def _get_node_feats(self, data):
        '''
        Returns node features from data
        '''
        node_feats = []
        if self.mode == "end-effector":
            node_feats = torch.cat([torch.tensor(data["robot0_eef_pos"]), torch.tensor(data["robot0_eef_quat"])], dim=0)
            node_feats = node_feats.reshape(1, -1) # add dimension
        else:
            if self.mode == "task-space":
                node_feats.append(torch.zeros((1,9)))
                node_feats = torch.cat(node_feats, dim=0).T
            elif self.mode == "joint-space":
                node_feats.append(torch.tensor([*data["robot0_joint_pos"], *data["robot0_gripper_qpos"]]).reshape(1,-1))
                node_feats = torch.cat(node_feats).T
            elif self.mode == "task-joint-space":
                node_feats = []
                # [node, node_feats]
                node_feats.append(torch.tensor([*data[f"robot0_joint_pos"], *data["robot0_gripper_qpos"]]).reshape(1,-1))
                node_feats = torch.cat(node_feats, dim=0).T
        return node_feats


    def _get_edges(self):
        '''
        Returns edge index and attributes for graph
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        eef_idx = 8
        edge_index = []
        edge_attrs = []
        for idx in range(eef_idx):
            edge_index.append([idx, idx+1])
            edge_attrs.append(self.ROBOT_LINK_EDGE)

        # Connectivity of all other nodes to the last node of robot
        for idx in range(eef_idx + 1, self.NUM_GRAPH_NODES):
            edge_index.append([idx, eef_idx])
            edge_attrs.append(self.OBJECT_ROBOT_EDGE)


        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.long)


        return edge_index, edge_attrs


    def _robosuite_obs_to_robomimic_graph(self, obs):
        '''
        Converts robosuite Gym Wrapper observations to the RobomimicGraphDataset format
        * requires robot_joint to be "flagged" in robomimic environment
        '''
        node_feats = torch.tensor([])
        node_pos = torch.tensor([])
        for i in range(len(self.robots)):
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
            gripper_pose = obs[j + 35:j + 37]
            # Skip 2  - gripper joint velocities
            robot_i_data = {
                "robot0_joint_pos": robot_joint_pos,
                "robot0_joint_vel": robot_joint_vel,
                "robot0_eef_pos": eef_pose,
                "robot0_eef_quat": eef_quat,
                "robot0_gripper_qpos": gripper_pose,
            }
            node_feats = torch.cat([node_feats, self._get_node_feats(robot_i_data)], dim=0)

        robot_i_data["object"] = obs[len(self.robots)*39:]
        node_pos = self._get_node_pos(robot_i_data)

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)
        
        obj_feats_tensor = self._get_object_feats(obs)
        
        node_feats = torch.cat((node_feats, obj_feats_tensor), dim=0)

        edge_index, edge_attrs = self._get_edges()

        y = node_pos # observations are the task-space positions


        # create graph
        graph = torch_geometric.data.Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs, y=y, pos=node_pos)

        return graph
    

    def reset(self):
        obs, _ =  self.env.reset()
        return self._robosuite_obs_to_robomimic_graph(obs)
    

    def step(self, action):
        final_action = []
        for i in range(len(self.robots)):
            '''
            Robosuite's action space is composed of 7 joint velocities and 1 gripper velocity, while 
            in the robomimic datasets, it's composed of 7 joint velocities and 2 gripper velocities (for each "finger").
            '''
            j = i*9
            robot_joint_pos = action[j:j + 7]
            robot_gripper_pos = -action[j + 7]
            final_action = [*final_action, *robot_joint_pos, robot_gripper_pos]
        obs, reward, done, _, info = self.control_loop(final_action)
        # obs, reward, done, _, info = self.env.step(final_action)
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