"""
Test Suite: Policy / Dataset Replay Consistency
================================================

Goal
----
Validate that the full policy inference pipeline (obs_deque → get_action →
env.step) produces behaviour that is consistent with the offline dataset.

These tests expose disconnects between the training data format and the
live observation format that would cause the policy to perform well on the
offline validation set but poorly in the real environment.

Tests
-----
1. test_obs_deque_y_matches_dataset_y
   The y tensor passed to the policy at inference is assembled from an
   obs_deque of RobomimicGraphWrapper observations.  The y tensor used
   during training comes from RobomimicGraphDataset.get_y_feats().
   They must agree for the same joint state.

2. test_dataset_playback_obs_format
   Feed a sequence of dataset samples through the policy's obs_deque
   assembly logic (mirrors get_action()'s first few lines) and verify that
   the resulting nobs tensor has the correct shape and is normalised to [-1,1].

3. test_action_step_matches_dataset_transition
   Step the live robosuite environment with the *dataset's recorded actions*
   (not policy-predicted actions) and verify the resulting joint positions
   agree with the next dataset observation within tolerance.
   This confirms that the action representation used in the dataset is
   compatible with the wrapper's step() interface.

4. test_dataset_y_and_wrapper_y_feature_order_match
   The obs feature vector (y) must have the same column ordering between
   dataset and wrapper, because the normalizer is fit on the dataset's y.
   Columns: [joint_pos(7), gripper_qpos(2), node_id(1)] for robot nodes.
"""

import importlib.util
import json
import os
import types
import collections

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/lift/ph/low_dim_v141.hdf5"
EPISODE_KEY  = "demo_0"

# ── lift-task config (lift_graph.yaml) ────────────────────────────────────────
BASE_LINK_SHIFT    = [[-0.56, 0.0, 0.912]]
BASE_LINK_ROTATION = [[0.0, 0.0, 0.0, 1.0]]

# ── tolerances ────────────────────────────────────────────────────────────────
Y_MATCH_TOL     = 1e-5   # m  — y tensors from dataset vs wrapper must agree
NORM_RANGE_TOL  = 1.05   # normalised values must lie in [-NORM_RANGE_TOL, NORM_RANGE_TOL]
JOINT_STEP_TOL  = 0.08   # rad — max joint error after one env.step from dataset action


# ── module loaders ────────────────────────────────────────────────────────────

def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(os.path.dirname(__file__), "..", rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def episode_data():
    """Load demo_0 from the HDF5 file."""
    with h5py.File(DATASET_PATH, "r") as f:
        ep          = f[f"data/{EPISODE_KEY}"]
        joint_pos   = ep["obs/robot0_joint_pos"][:]
        gripper_qpos = ep["obs/robot0_gripper_qpos"][:]
        gripper_qvel = ep["obs/robot0_gripper_qvel"][:]
        joint_vel   = ep["obs/robot0_joint_vel"][:]
        object_obs  = ep["obs/object"][:]
        actions     = ep["actions"][:]
        states      = ep["states"][:]
    return dict(
        joint_pos=joint_pos,
        gripper_qpos=gripper_qpos,
        gripper_qvel=gripper_qvel,
        joint_vel=joint_vel,
        object_obs=object_obs,
        actions=actions,
        states=states,
    )


@pytest.fixture(scope="module")
def dataset():
    """Instantiate a real RobomimicGraphDataset (uses the processed cache)."""
    mod = _load_module("rg_dataset", "imitation/dataset/robomimic_graph_dataset.py")
    ds = mod.RobomimicGraphDataset(
        dataset_path=DATASET_PATH,
        robots=["Panda"],
        object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
        object_state_keys={"cube": ["cube_pos", "cube_quat"]},
        pred_horizon=16,
        obs_horizon=4,
        control_mode="JOINT_VELOCITY",
        base_link_shift=BASE_LINK_SHIFT,
        base_link_rotation=BASE_LINK_ROTATION,
    )
    return ds


@pytest.fixture(scope="module")
def wrapper_get_y_fn():
    """Return a bound _get_y_feats callable from RobomimicGraphWrapper."""
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    mod = _load_module("rg_wrapper", "imitation/env/robomimic_graph_wrapper.py")

    mock = types.SimpleNamespace(
        num_robots=1,
        BASE_LINK_SHIFT=BASE_LINK_SHIFT,
        BASE_LINK_ROTATION=BASE_LINK_ROTATION,
        rotation_transformer=RotationTransformer(from_rep="quaternion", to_rep="rotation_6d"),
        object_state_keys={"cube": ["cube_pos", "cube_quat"]},
        object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
        num_objects=1,
    )
    get_obj_pos  = mod.RobomimicGraphWrapper._get_object_pos.__get__(mock)
    mock._get_object_pos = get_obj_pos
    get_y_feats  = mod.RobomimicGraphWrapper._get_y_feats.__get__(mock)
    return get_y_feats


@pytest.fixture(scope="module")
def dataset_get_y_fn():
    """Return a bound get_y_feats callable from RobomimicGraphDataset."""
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    mod = _load_module("rg_dataset2", "imitation/dataset/robomimic_graph_dataset.py")

    mock = types.SimpleNamespace(
        num_robots=1,
        BASE_LINK_SHIFT=BASE_LINK_SHIFT,
        BASE_LINK_ROTATION=BASE_LINK_ROTATION,
        rotation_transformer=RotationTransformer(from_rep="quaternion", to_rep="rotation_6d"),
        object_state_keys={"cube": ["cube_pos", "cube_quat"]},
        object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
        num_objects=1,
        obs_feature_dim=7,
    )
    get_obj_pos = mod.RobomimicGraphDataset._get_object_pos.__get__(mock)
    mock._get_object_pos = get_obj_pos
    get_y_feats = mod.RobomimicGraphDataset.get_y_feats.__get__(mock)
    return get_y_feats


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_wrapper_obs_dict(episode_data, t):
    """Build the flat obs dict expected by wrapper._get_y_feats at timestep t."""
    return {
        "robot0_joint_pos":    episode_data["joint_pos"][t],
        "robot0_joint_vel":    episode_data["joint_vel"][t],
        "robot0_gripper_qpos": episode_data["gripper_qpos"][t],
        "robot0_gripper_qvel": episode_data["gripper_qvel"][t],
        "object":              episode_data["object_obs"][t],
    }


def _build_dataset_data_dict(episode_data):
    """Build the time-indexed data dict expected by dataset.get_y_feats."""
    return {
        "robot0_joint_pos":    episode_data["joint_pos"],
        "robot0_gripper_qpos": episode_data["gripper_qpos"],
        "robot0_joint_vel":    episode_data["joint_vel"],
        "robot0_gripper_qvel": episode_data["gripper_qvel"],
        "object":              episode_data["object_obs"],
    }


# ── Test 1: obs y tensors agree ───────────────────────────────────────────────

class TestObsYConsistency:
    """
    The y tensor (observations) fed to the GDDPM must be identical whether
    it comes from the dataset (training path) or from the wrapper (eval path).

    A mismatch here means the network sees a completely different conditioning
    signal at eval time than it was trained on — a guaranteed performance cliff.
    """

    def test_wrapper_y_matches_dataset_y_at_each_step(
        self, episode_data, wrapper_get_y_fn, dataset_get_y_fn
    ):
        """
        For every timestep t, compare:
          - wrapper._get_y_feats(obs_dict_at_t)         → shape (num_nodes, feat)
          - dataset.get_y_feats(data_dict, t_vals=[t])  → shape (num_nodes, 1, feat)

        Both should agree on the robot-node rows (indices 0..8).
        The last column (node ID) is part of y in both paths; it is the
        running index 0..num_nodes-1 and should be identical.
        """
        data_dict = _build_dataset_data_dict(episode_data)
        T = len(episode_data["joint_pos"])

        max_err = 0.0
        worst_t = -1
        for t in range(T):
            obs_dict   = _build_wrapper_obs_dict(episode_data, t)
            y_wrapper  = wrapper_get_y_fn(obs_dict)              # (num_nodes, feat)
            y_dataset  = dataset_get_y_fn(data_dict, [t])        # (num_nodes, 1, feat)
            y_ds_t     = y_dataset[:, 0, :]                      # (num_nodes, feat)

            # Compare robot nodes only (first 9)
            robot_rows_w = y_wrapper[:9, :]
            robot_rows_d = y_ds_t[:9, :]

            err = float(torch.max(torch.abs(robot_rows_w - robot_rows_d)).item())
            if err > max_err:
                max_err = err
                worst_t = t

        print(f"\n── Wrapper y vs dataset y (robot nodes) ─────────────────")
        print(f"  Steps checked : {T}")
        print(f"  Max element error : {max_err:.6f}  at step {worst_t}")

        assert max_err <= Y_MATCH_TOL, (
            f"Wrapper._get_y_feats and dataset.get_y_feats disagree by "
            f"{max_err:.2e} at step {worst_t} (tolerance {Y_MATCH_TOL:.0e}).\n"
            f"The network sees different obs conditioning at train vs eval time.\n"
            f"Check that both use the same feature ordering: "
            f"[joint_pos(7), gripper_qpos(2), node_id(1)] for robot nodes."
        )

    def test_obs_y_feature_shape_is_consistent(
        self, episode_data, wrapper_get_y_fn, dataset_get_y_fn
    ):
        """
        y from the wrapper (single step) and from the dataset (single step)
        must have the same number of columns (feature dimensionality).
        """
        data_dict = _build_dataset_data_dict(episode_data)
        obs_dict  = _build_wrapper_obs_dict(episode_data, 0)

        y_wrapper = wrapper_get_y_fn(obs_dict)
        y_dataset = dataset_get_y_fn(data_dict, [0])

        print(f"\n── y feature shape ──────────────────────────────────────")
        print(f"  wrapper  y.shape : {tuple(y_wrapper.shape)}")
        print(f"  dataset  y.shape : {tuple(y_dataset[:, 0, :].shape)}")

        assert y_wrapper.shape == y_dataset[:, 0, :].shape, (
            f"y shape mismatch: wrapper {tuple(y_wrapper.shape)} vs "
            f"dataset {tuple(y_dataset[:, 0, :].shape)}.\n"
            f"The policy obs conditioning tensor has the wrong number of features."
        )


# ── Test 2: nobs format when assembled via obs_deque ─────────────────────────

class TestObsDequeAssembly:
    """
    In get_action(), the policy assembles nobs as:

        for i in range(len(obs_deque)):
            obs_cond.append(obs_deque[i].y.unsqueeze(1))
        obs = torch.cat(obs_cond, dim=1)       # (nodes, obs_horizon, feat)
        nobs = dataset.normalize_data(obs, 'obs')

    This test simulates that assembly using dataset samples and verifies:
    (a) shape is (num_nodes, obs_horizon, obs_feat_dim)
    (b) normalised values are in [-1, 1]
    """

    OBS_HORIZON = 4

    def _assemble_nobs(self, dataset, start_idx):
        """Assemble nobs the same way get_action() does, using dataset samples."""
        obs_cond = []
        for i in range(self.OBS_HORIZON):
            idx  = max(0, start_idx - (self.OBS_HORIZON - 1 - i))
            data = dataset.get(idx)
            # data.y shape: (nodes, obs_horizon, feat) — take last step
            obs_cond.append(data.y[:, -1:, :])   # (nodes, 1, feat)
        obs = torch.cat(obs_cond, dim=1)          # (nodes, obs_horizon, feat)
        return obs

    def test_nobs_shape(self, dataset):
        """nobs assembled from obs_deque has the expected shape."""
        num_nodes = 9 + 1   # 9 robot + 1 object for lift task
        obs = self._assemble_nobs(dataset, start_idx=10)

        print(f"\n── nobs shape check ─────────────────────────────────────")
        print(f"  Assembled nobs shape : {tuple(obs.shape)}")
        print(f"  Expected             : ({num_nodes}, {self.OBS_HORIZON}, *)")

        assert obs.shape[0] == num_nodes, (
            f"nobs has {obs.shape[0]} nodes, expected {num_nodes}."
        )
        assert obs.shape[1] == self.OBS_HORIZON, (
            f"nobs has {obs.shape[1]} obs steps, expected {self.OBS_HORIZON}."
        )

    def test_nobs_normalised_range(self, dataset):
        """Normalized nobs (excluding node-ID column) is in [-NORM_RANGE_TOL, NORM_RANGE_TOL]."""
        CHECK_N = 20
        step    = max(1, dataset.len() // CHECK_N)

        all_norm = []
        for start_idx in range(0, dataset.len(), step):
            obs   = self._assemble_nobs(dataset, start_idx=start_idx)
            nobs  = dataset.normalize_data(obs, stats_key="obs")
            # Exclude node-ID column (last feature)
            nobs_no_id = nobs[:, :, :-1]
            all_norm.append(nobs_no_id.reshape(-1).detach().numpy())

        import numpy as np
        all_norm = np.concatenate(all_norm)
        out_of_range = np.abs(all_norm) > NORM_RANGE_TOL
        frac_oob = out_of_range.mean()

        print(f"\n── nobs normalisation range ─────────────────────────────")
        print(f"  Samples   : {CHECK_N}")
        print(f"  Min norm  : {all_norm.min():.4f}")
        print(f"  Max norm  : {all_norm.max():.4f}")
        print(f"  Frac OOB  : {frac_oob*100:.3f} %")

        assert frac_oob == 0.0, (
            f"{frac_oob*100:.2f}% of nobs values outside ±{NORM_RANGE_TOL} "
            f"(min={all_norm.min():.4f}, max={all_norm.max():.4f}).\n"
            f"Obs normalizer saturates the conditioning signal before it reaches "
            f"the GDDPM, causing it to discard information."
        )


# ── Test 3: dataset action → env step → next obs ─────────────────────────────

class TestActionStepMatchesDatasetTransition:
    """
    Replay dataset-recorded actions through the wrapper's step() and verify
    that the resulting joint positions match dataset obs[t+1] within tolerance.

    This is the decisive end-to-end check: it validates that the action format
    the policy outputs (graph node velocities) is correctly interpreted by the
    wrapper.  A failure here means even a *perfect* policy would fail at eval.

    To map dataset actions (shape (T, 7) OSC_POSE / JOINT_VELOCITY) to the
    9-element graph action format:
        graph_action[0:7]  = dataset joint velocities (7 DOF)
        graph_action[7]    = dataset gripper velocity (finger 0) ← often 0 or ±1
        graph_action[8]    = dataset gripper velocity (finger 1)

    The wrapper then uses action[:7] + action[8], discarding action[7].
    """

    N_STEPS = 20   # replay first N steps to keep test fast

    def test_dataset_action_produces_correct_next_obs(self, episode_data):
        """
        Restore the simulator to the recorded t=0 state, then apply the
        first N_STEPS dataset actions through a live robosuite env with the
        SAME control_freq as the dataset.  Compare resulting joint_pos with
        dataset obs[t+1].

        Uses make_env() (reads control_freq from HDF5) so this test is
        self-consistent regardless of the wrapper's hard-coded control_freq.
        """
        import robosuite as suite

        # Build the env from the recorded env_args (bypasses wrapper)
        with h5py.File(DATASET_PATH, "r") as f:
            env_args   = json.loads(f["data"].attrs["env_args"])
        env_kwargs = dict(env_args["env_kwargs"])
        env_kwargs["has_renderer"]          = False
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["reward_shaping"]         = False
        env = suite.make(env_args["env_name"], **env_kwargs)

        env.reset()
        env.sim.set_state_from_flattened(episode_data["states"][0])
        env.sim.forward()

        actions   = episode_data["actions"]     # (T, 7) — raw OSC / JV actions
        joint_pos = episode_data["joint_pos"]   # (T, 7) — ground truth obs

        max_err = 0.0
        worst_t = -1
        per_step = []
        for t in range(min(self.N_STEPS, len(actions) - 1)):
            live_obs, _, _, _ = env.step(actions[t])
            # Raw robosuite env (no GymWrapper) provides sin/cos instead of
            # joint_pos directly. Reconstruct via arctan2.
            sin_q = live_obs["robot0_joint_pos_sin"]   # (7,)
            cos_q = live_obs["robot0_joint_pos_cos"]   # (7,)
            live_jpos = np.arctan2(sin_q, cos_q)
            ds_jpos   = joint_pos[t + 1]
            err = float(np.max(np.abs(live_jpos - ds_jpos)))
            per_step.append(err)
            if err > max_err:
                max_err = err
                worst_t = t

        env.close()

        print(f"\n── Dataset action → env step → joint_pos match ──────────")
        print(f"  Steps replayed   : {len(per_step)}")
        print(f"  Max joint error  : {max_err:.5f} rad  at step {worst_t}")
        print(f"  Mean joint error : {np.mean(per_step):.5f} rad")

        assert max_err <= JOINT_STEP_TOL, (
            f"Applying dataset action at step {worst_t} yielded joint_pos error "
            f"{max_err:.5f} rad (tolerance {JOINT_STEP_TOL} rad).\n"
            f"This means the dataset action format is NOT compatible with the "
            f"environment's step() interface — the policy will fail at eval even if "
            f"it perfectly reproduces the training actions.\n"
            f"Likely cause: action convention mismatch (OSC_POSE vs JOINT_VELOCITY) "
            f"or control_freq mismatch."
        )

    def test_wrapper_step_with_dataset_action_matches_next_obs(self, episode_data):
        """
        Same as above, but stepping through RobomimicGraphWrapper.step() to
        test the wrapper's action interpretation end-to-end.

        The wrapper expects a 9-element action vector (one per graph node).
        We pad the 7-DOF dataset action with zeros at positions 7 and 8
        (gripper fingers), matching the expected format.

        NOTE: this test deliberately targets the first N_STEPS of demo_0
        to keep runtime short.  A failure indicates the wrapper's step()
        action slicing is wrong.
        """
        from imitation.env.robomimic_graph_wrapper import RobomimicGraphWrapper

        wrapper = RobomimicGraphWrapper(
            object_state_keys={"cube": ["cube_pos", "cube_quat"]},
            object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
            task="Lift",
            has_renderer=False,
            robots=["Panda"],
            control_mode="JOINT_VELOCITY",
            base_link_shift=BASE_LINK_SHIFT,
            base_link_rotation=BASE_LINK_ROTATION,
        )

        # Restore to recorded t=0 via the inner robosuite env
        wrapper.env.env.reset()
        wrapper.env.env.sim.set_state_from_flattened(episode_data["states"][0])
        wrapper.env.env.sim.forward()

        actions_raw = episode_data["actions"]   # (T, 7) raw velocities
        joint_pos   = episode_data["joint_pos"]  # (T, 7) ground truth

        max_err = 0.0
        worst_t = -1
        per_step = []

        for t in range(min(self.N_STEPS, len(actions_raw) - 1)):
            # Pad to 9-element graph action: [j0..j6, gripper_f0, gripper_f1]
            # Dataset stores 7-DOF velocities; gripper comes from gripper_qvel
            gripper_vel = episode_data["gripper_qvel"][t]

            # Build and pass a 9-element action to wrapper.step()
            graph_action = np.concatenate([
                actions_raw[t],              # 7 joint velocities / OSC DOF
                gripper_vel[:2],             # 2 gripper DOF
            ])                               # total: 9 elements

            graph_obs, _, done, _ = wrapper.step(graph_action)

            # Extract joint_pos from the graph observation's y field
            # y shape: (num_nodes, feat) where feat = [jp0..jp6, gp0, gp1, node_id]
            # Robot nodes 0..8; joint pos is stored in y[:9, 0..6]
            live_jpos = graph_obs.y[:7, 0].detach().numpy()   # nodes 0-6 → 7 joints
            ds_jpos   = joint_pos[t + 1]

            err = float(np.max(np.abs(live_jpos - ds_jpos)))
            per_step.append(err)
            if err > max_err:
                max_err = err
                worst_t = t

            if done:
                break

        wrapper.close()

        print(f"\n── Wrapper step joint_pos match ─────────────────────────")
        print(f"  Steps replayed   : {len(per_step)}")
        print(f"  Max joint error  : {max_err:.5f} rad  at step {worst_t}")
        print(f"  Mean joint error : {np.mean(per_step):.5f} rad")

        assert max_err <= JOINT_STEP_TOL, (
            f"wrapper.step() joint_pos error {max_err:.5f} rad at step {worst_t} "
            f"exceeds {JOINT_STEP_TOL} rad.\n"
            f"This verifies that even with perfect dataset actions, the wrapper "
            f"does not correctly advance the simulator state.\n"
            f"Likely causes: (1) control_freq mismatch between dataset and wrapper, "
            f"(2) wrong action slicing (action[j+8] instead of action[j+7]), "
            f"(3) wrong control mode (OSC_POSE vs JOINT_VELOCITY)."
        )


# ── Test 4: y feature ordering ────────────────────────────────────────────────

class TestObsYFeatureOrdering:
    """
    Validate that wrapper._get_y_feats and dataset.get_y_feats produce
    identical feature *ordering* for robot nodes:
        col 0..6  : joint_pos  (7 values)
        col 7..8  : gripper_qpos (2 values)
        col 9     : node_id

    A column-ordering mismatch would mean the normalizer scales the wrong
    physical quantities, making the policy conditioning signal meaningless.
    """

    def test_robot_y_columns_are_joint_pos_then_gripper_then_id(
        self, episode_data, wrapper_get_y_fn, dataset_get_y_fn
    ):
        """
        At a known timestep, check that columns 0-6 of y[:9] match joint_pos,
        columns 7-8 match gripper_qpos, and column 9 (if present) matches
        the node index 0..8.

        This pins down the actual in-memory layout, making any accidental
        reordering immediately visible.
        """
        t = 5     # arbitrary mid-episode step
        obs_dict  = _build_wrapper_obs_dict(episode_data, t)
        data_dict = _build_dataset_data_dict(episode_data)

        y_wrapper = wrapper_get_y_fn(obs_dict)           # (num_nodes, feat)
        y_dataset = dataset_get_y_fn(data_dict, [t])[:, 0, :]  # (num_nodes, feat)

        jp  = torch.tensor(episode_data["joint_pos"][t])    # (7,)
        gp  = torch.tensor(episode_data["gripper_qpos"][t]) # (2,)

        # Robot nodes 0..6 correspond to 7 joints; nodes 7 & 8 are gripper nodes.
        # get_y_feats packs each robot node with its own joint feature:
        #   node i → [joint_i_val, 0, 0, ..., node_id]  (sparse, one joint per node)

        print(f"\n── y feature ordering check (t={t}) ──────────────────────")
        print(f"  Wrapper y[:10,:] =\n{y_wrapper[:10,:]}")
        print(f"  Dataset y[:10,:] =\n{y_dataset[:10,:]}")
        print(f"  Expected jp: {jp.numpy()}")
        print(f"  Expected gp: {gp.numpy()}")

        # Verify node IDs (last column) for both wrapper and dataset
        num_robot_nodes = 9
        expected_node_ids = torch.arange(num_robot_nodes, dtype=y_wrapper.dtype)

        wrapper_node_ids = y_wrapper[:num_robot_nodes, -1]
        dataset_node_ids = y_dataset[:num_robot_nodes, -1]

        assert torch.allclose(wrapper_node_ids, expected_node_ids, atol=1e-3), (
            f"Wrapper y node IDs {wrapper_node_ids.tolist()} != expected {expected_node_ids.tolist()}.\n"
            f"The node-ID column ordering is wrong in the wrapper."
        )
        assert torch.allclose(dataset_node_ids, expected_node_ids, atol=1e-3), (
            f"Dataset y node IDs {dataset_node_ids.tolist()} != expected {expected_node_ids.tolist()}.\n"
            f"The node-ID column ordering is wrong in the dataset."
        )


# ── Test 5: OSC_POSE node features shape and content ─────────────────────────

class TestOscPoseNodeFeats:
    """
    Verify that RobomimicGraphWrapper._get_node_feats for control_mode='OSC_POSE'
    produces the same 9-node structure as JOINT modes, preserving graph topology.

    Expected behavior (node_feature_dim=1):
    - Shape: (9, 1) -- 9 robot nodes x 1 scalar feature each
    - Nodes 0-2: eef_pos  components (3D position)
    - Nodes 3-6: eef_quat components (4D quaternion)
    - Node 7:    unused (0.0)
    - Node 8:    mean gripper_qpos
    """

    def _make_wrapper_node_feats_fn(self):
        """Build bound _get_node_feats callable with OSC_POSE control mode."""
        import types
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer

        mod = _load_module("rg_wrapper_osc", "imitation/env/robomimic_graph_wrapper.py")
        mock = types.SimpleNamespace(
            num_robots=1,
            control_mode="OSC_POSE",
            BASE_LINK_SHIFT=BASE_LINK_SHIFT,
            BASE_LINK_ROTATION=BASE_LINK_ROTATION,
            rotation_transformer=RotationTransformer(from_rep="quaternion", to_rep="rotation_6d"),
            object_state_keys={"cube": ["cube_pos", "cube_quat"]},
            object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
            num_objects=1,
        )
        return mod.RobomimicGraphWrapper._get_node_feats.__get__(mock)

    def _build_obs_dict_osc(self, episode_data, t):
        """Build obs dict with eef_pos and eef_quat_raw for OSC_POSE _get_node_feats."""
        with h5py.File(DATASET_PATH, "r") as f:
            ep = f["data/demo_0"]
            eef_pos  = ep["obs/robot0_eef_pos"][t]
            eef_quat = ep["obs/robot0_eef_quat"][t]
        return {
            "robot0_joint_pos":    episode_data["joint_pos"][t],
            "robot0_joint_vel":    episode_data["joint_vel"][t],
            "robot0_gripper_qpos": episode_data["gripper_qpos"][t],
            "robot0_gripper_qvel": episode_data["gripper_qvel"][t],
            "robot0_eef_pos":      eef_pos,
            "robot0_eef_quat_raw": eef_quat,  # raw 4D
        }

    def test_osc_pose_node_feats_shape(self, episode_data):
        """OSC_POSE node features must be (9, 1) -- matching JOINT mode topology."""
        get_node_feats = self._make_wrapper_node_feats_fn()
        obs_dict = self._build_obs_dict_osc(episode_data, 10)
        feats = get_node_feats(obs_dict, control_mode="OSC_POSE")
        assert feats.shape == (9, 1), (
            f"OSC_POSE node features have shape {tuple(feats.shape)}, expected (9, 1).\n"
            f"The graph topology must have 9 robot nodes for GDDPM compatibility."
        )

    def test_osc_pose_nodes_0to2_match_eef_pos(self, episode_data):
        """Nodes 0-2 must match the raw eef_pos values."""
        get_node_feats = self._make_wrapper_node_feats_fn()
        t = 10
        obs_dict = self._build_obs_dict_osc(episode_data, t)
        feats = get_node_feats(obs_dict, control_mode="OSC_POSE")
        expected_pos = torch.tensor(obs_dict["robot0_eef_pos"], dtype=torch.float32)
        assert torch.allclose(feats[:3, 0], expected_pos, atol=1e-5), (
            f"OSC_POSE nodes 0-2 (eef_pos) mismatch:\n"
            f"  got      {feats[:3, 0].tolist()}\n"
            f"  expected {expected_pos.tolist()}"
        )

    def test_osc_pose_nodes_3to6_match_eef_quat(self, episode_data):
        """Nodes 3-6 must match the raw eef_quat (4D) values."""
        get_node_feats = self._make_wrapper_node_feats_fn()
        t = 10
        obs_dict = self._build_obs_dict_osc(episode_data, t)
        feats = get_node_feats(obs_dict, control_mode="OSC_POSE")
        expected_quat = torch.tensor(obs_dict["robot0_eef_quat_raw"], dtype=torch.float32)
        assert torch.allclose(feats[3:7, 0], expected_quat, atol=1e-5), (
            f"OSC_POSE nodes 3-6 (eef_quat) mismatch:\n"
            f"  got      {feats[3:7, 0].tolist()}\n"
            f"  expected {expected_quat.tolist()}"
        )

    def test_osc_pose_node7_is_zero(self, episode_data):
        """Node 7 (unused) must be 0.0."""
        get_node_feats = self._make_wrapper_node_feats_fn()
        obs_dict = self._build_obs_dict_osc(episode_data, 10)
        feats = get_node_feats(obs_dict, control_mode="OSC_POSE")
        assert float(feats[7, 0]) == 0.0, (
            f"OSC_POSE node 7 (unused) is {float(feats[7, 0])}, expected 0.0."
        )

    def test_osc_pose_node8_is_gripper(self, episode_data):
        """Node 8 must contain the mean gripper_qpos."""
        get_node_feats = self._make_wrapper_node_feats_fn()
        for t in range(len(episode_data["gripper_qpos"])):
            if np.any(np.abs(episode_data["gripper_qpos"][t]) > 0.01):
                break
        obs_dict = self._build_obs_dict_osc(episode_data, t)
        feats = get_node_feats(obs_dict, control_mode="OSC_POSE")
        expected_val = float(np.mean(episode_data["gripper_qpos"][t]))
        assert abs(float(feats[8, 0]) - expected_val) < 1e-5, (
            f"OSC_POSE gripper node {float(feats[8, 0]):.6f} != expected {expected_val:.6f}"
        )



# ── Test 6: OSC_POSE wrapper step replay ─────────────────────────────────────

class TestOscPoseWrapperStep:
    """
    Replay OSC_POSE dataset actions (7D) through RobomimicGraphWrapper with
    control_mode='OSC_POSE' and verify end-to-end correctness:

    1. Graph observations have the expected 10-node count (9 robot + 1 object).
    2. Node positions (pos) contain no NaN or Inf values.
    3. Joint positions after each step match dataset obs[t+1] within tolerance.

    The dataset was recorded with OSC_POSE, so replaying its actions in the
    same environment with OSC_POSE should reproduce the original trajectory.
    """

    N_STEPS = 20          # replay first N steps (fast smoke test)
    JOINT_TOL = 0.08      # rad — same as TestActionStepMatchesDatasetTransition

    def test_osc_pose_step_graph_structure(self, episode_data):
        """
        Each step must return a graph with 10 nodes (9 robot + 1 cube object).
        """
        from imitation.env.robomimic_graph_wrapper import RobomimicGraphWrapper

        wrapper = RobomimicGraphWrapper(
            object_state_keys={"cube": ["cube_pos", "cube_quat"]},
            object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
            task="Lift",
            has_renderer=False,
            robots=["Panda"],
            control_mode="OSC_POSE",
            base_link_shift=BASE_LINK_SHIFT,
            base_link_rotation=BASE_LINK_ROTATION,
        )
        wrapper.env.env.reset()
        wrapper.env.env.sim.set_state_from_flattened(episode_data["states"][0])
        wrapper.env.env.sim.forward()

        actions = episode_data["actions"]  # (T, 7) — OSC_POSE raw actions
        for t in range(min(self.N_STEPS, len(actions) - 1)):
            graph_obs, reward, done, info = wrapper.step(actions[t])
            assert graph_obs.x.shape[0] == 10, (
                f"Step {t}: graph has {graph_obs.x.shape[0]} nodes, expected 10 "
                f"(9 robot + 1 object). OSC_POSE changed the graph topology."
            )
            if done:
                break

        wrapper.close()

    def test_osc_pose_step_pos_no_nan(self, episode_data):
        """
        Node positions must be finite (no NaN / Inf) after each OSC_POSE step.
        """
        from imitation.env.robomimic_graph_wrapper import RobomimicGraphWrapper

        wrapper = RobomimicGraphWrapper(
            object_state_keys={"cube": ["cube_pos", "cube_quat"]},
            object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
            task="Lift",
            has_renderer=False,
            robots=["Panda"],
            control_mode="OSC_POSE",
            base_link_shift=BASE_LINK_SHIFT,
            base_link_rotation=BASE_LINK_ROTATION,
        )
        wrapper.env.env.reset()
        wrapper.env.env.sim.set_state_from_flattened(episode_data["states"][0])
        wrapper.env.env.sim.forward()

        actions = episode_data["actions"]
        for t in range(min(self.N_STEPS, len(actions) - 1)):
            graph_obs, reward, done, info = wrapper.step(actions[t])
            pos = graph_obs.pos
            assert torch.all(torch.isfinite(pos)), (
                f"Step {t}: graph_obs.pos contains NaN or Inf:\n{pos}"
            )
            if done:
                break

        wrapper.close()

    def test_osc_pose_step_joint_pos_matches_dataset(self, episode_data):
        """
        After replaying dataset OSC_POSE actions, the resulting joint positions
        must match dataset obs[t+1] within JOINT_TOL radians.

        This is the key functional test: it verifies that the wrapper correctly
        forwards 7D OSC_POSE actions to robosuite without any reformatting.
        """
        from imitation.env.robomimic_graph_wrapper import RobomimicGraphWrapper

        wrapper = RobomimicGraphWrapper(
            object_state_keys={"cube": ["cube_pos", "cube_quat"]},
            object_state_sizes={"cube_pos": 3, "cube_quat": 4, "gripper_to_cube_pos": 3},
            task="Lift",
            has_renderer=False,
            robots=["Panda"],
            control_mode="OSC_POSE",
            base_link_shift=BASE_LINK_SHIFT,
            base_link_rotation=BASE_LINK_ROTATION,
        )
        wrapper.env.env.reset()
        wrapper.env.env.sim.set_state_from_flattened(episode_data["states"][0])
        wrapper.env.env.sim.forward()

        actions   = episode_data["actions"]    # (T, 7) OSC_POSE actions
        joint_pos = episode_data["joint_pos"]  # (T, 7) ground truth

        max_err = 0.0
        worst_t = -1
        per_step = []

        for t in range(min(self.N_STEPS, len(actions) - 1)):
            graph_obs, reward, done, info = wrapper.step(actions[t])

            # y field stores [joint_pos(7), gripper(2), node_id] per robot node
            # Each robot node i stores its own joint value at y[i, 0]
            live_jpos = graph_obs.y[:7, 0].detach().numpy()
            ds_jpos   = joint_pos[t + 1]
            err = float(np.max(np.abs(live_jpos - ds_jpos)))
            per_step.append(err)
            if err > max_err:
                max_err = err
                worst_t = t

            if done:
                break

        wrapper.close()

        print(f"\n── OSC_POSE wrapper replay (joint_pos) ─────────────────────")
        print(f"  Steps replayed   : {len(per_step)}")
        print(f"  Max joint error  : {max_err:.5f} rad  at step {worst_t}")
        print(f"  Mean joint error : {np.mean(per_step):.5f} rad")

        assert max_err <= self.JOINT_TOL, (
            f"OSC_POSE wrapper.step() joint_pos error {max_err:.5f} rad at step {worst_t} "
            f"exceeds tolerance {self.JOINT_TOL} rad.\n"
            f"Likely causes:\n"
            f"  (1) Actions not forwarded as-is to robosuite (check step() for OSC_POSE branch)\n"
            f"  (2) control_freq mismatch between HDF5 env_args and wrapper\n"
            f"  (3) Sim state restoration at t=0 incomplete"
        )

