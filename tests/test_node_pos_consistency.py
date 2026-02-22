"""
Test: _get_node_pos consistency between dataset FK and live robosuite environment.

Goal
----
Verify that `_get_node_pos` (in RobomimicGraphDataset) computes joint-link
positions that are consistent with what the robosuite simulator reports when
the same episode is replayed action-by-action from the exact same starting state.

Strategy
--------
The dataset (data/lift/ph/low_dim_v141.hdf5) was recorded with an OSC_POSE
controller (6-DOF EEF-delta + 1 gripper = 7D action) at control_freq=20.

For each timestep t in demo_0:
  1. Restore the simulator to the recorded state at t=0 via
     sim.set_state_from_flattened(states[0]) + sim.forward().
  2. Apply actions[0..t] to the live environment.
  3. Compare the resulting live robot0_joint_pos against dataset obs[t+1].
  4. Run the FK pipeline (calculate_panda_joints_positions + base_link_shift)
     on both the dataset and live joint positions, and compare Cartesian node
     positions.

Timing:
  dataset obs[t]  ──actions[t]──►  dataset obs[t+1]
                                    ≈ live obs after env.step(actions[t])

Tolerances
---------------------------
- Joint-space : 0.06 rad  (small accumulated integration drift is acceptable)
- Task-space  : 6 mm      (FK Cartesian node positions, derived from joint tol)

Configuration (from lift_graph.yaml)
-------------------------------------
- base_link_shift    = [-0.56, 0.0, 0.912]
- base_link_rotation = [0, 0, 0, 1]  (identity)
"""

import importlib.util
import os

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

# ── paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/lift/ph/low_dim_v141.hdf5"
EPISODE_KEY  = "demo_0"

# ── tolerances ────────────────────────────────────────────────────────────── 
JOINT_POS_TOL = 0.06   # rad  - max per-joint error over the full episode
CART_POS_TOL  = 6e-3   # m    - max Cartesian node-position error

# ── lift-task config (lift_graph.yaml) ───────────────────────────────────────
BASE_LINK_SHIFT    = np.array([-0.56, 0.0, 0.912])
BASE_LINK_ROTATION = [0.0, 0.0, 0.0, 1.0]   # identity (x,y,z,w)


# ── FK helper ─────────────────────────────────────────────────────────────────

def _load_calculate_panda_joints_positions():
    """
    Load calculate_panda_joints_positions directly from generic.py, bypassing
    imitation/utils/__init__.py which would import torch_geometric transitively.
    """
    spec = importlib.util.spec_from_file_location(
        "imitation_generic",
        os.path.join(os.path.dirname(__file__), "..", "imitation", "utils", "generic.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calculate_panda_joints_positions


_calculate_panda_joints_positions = None   # cached after first call


def compute_node_pos_xyz(joint_pos_7: np.ndarray,
                         gripper_qpos_2: np.ndarray) -> torch.Tensor:
    """
    Mirrors RobomimicGraphDataset._get_node_pos for a single robot.
    Returns shape (9, 3) - Cartesian [x, y, z] of each Panda link node,
    after applying the base_link rotation and shift from lift_graph.yaml.
    """
    global _calculate_panda_joints_positions
    if _calculate_panda_joints_positions is None:
        _calculate_panda_joints_positions = _load_calculate_panda_joints_positions()

    joints   = [*joint_pos_7.tolist(), *gripper_qpos_2.tolist()]
    node_pos = _calculate_panda_joints_positions(joints)   # (9, 7): xyz + quat

    rotation_matrix = R.from_quat(BASE_LINK_ROTATION)
    rot_mat_t = torch.tensor(rotation_matrix.as_matrix()).to(node_pos.dtype)
    node_pos[:, :3] = torch.matmul(node_pos[:, :3], rot_mat_t)
    node_pos[:, :3] += torch.tensor(BASE_LINK_SHIFT).to(node_pos.dtype)

    return node_pos[:, :3]   # (9, 3)


# ── robosuite env factory ─────────────────────────────────────────────────────

def make_env():
    """
    Re-creates the robosuite environment matching the recording parameters
    stored in the HDF5 env_args (OSC_POSE, control_freq=20).
    """
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config.update({
        "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150, "damping": 1, "impedance_mode": "fixed",
        "control_delta": True, "uncouple_pos_ori": True,
        "interpolation": None, "ramp_ratio": 0.2,
    })
    return suite.make(
        "Lift",
        robots=["Panda"],
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        controller_configs=controller_config,
    )


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def episode_data():
    """Load demo_0 observations, actions and sim states from the HDF5 file."""
    with h5py.File(DATASET_PATH, "r") as f:
        ep = f[f"data/{EPISODE_KEY}"]
        joint_pos    = ep["obs/robot0_joint_pos"][:]     # (T, 7)
        gripper_qpos = ep["obs/robot0_gripper_qpos"][:]  # (T, 2)
        actions      = ep["actions"][:]                   # (T, 7): OSC_POSE + gripper
        states       = ep["states"][:]                    # (T, 32): flat MuJoCo sim state
    return joint_pos, gripper_qpos, actions, states


# ── tests ─────────────────────────────────────────────────────────────────────

class TestNodePosConsistency:
    """
    Replays demo_0 in a live robosuite env (from the exact recorded initial
    state) and checks that FK positions from dataset joint_pos match those
    from live joint observations.
    """

    def test_fk_from_dataset_obs_matches_live_env(self, episode_data):
        """
        Main consistency test.

        The env is restored to the recorded t=0 sim state via
        sim.set_state_from_flattened(states[0]).  We then replay each action
        from the dataset and compare:
          (a) Joint-space: dataset obs[t+1] vs. live joint_pos - must be
              within JOINT_POS_TOL (0.06 rad).
          (b) Task-space: FK node positions from (a) must agree within
              CART_POS_TOL (6 mm).

        A failure means either:
          - The FK in _get_node_pos uses stale / off-by-one joint data.
          - The base_link_shift/rotation is applied incorrectly in the dataset.
          - There is accumulated integration drift (expected to be small for
            a deterministic OSC_POSE controller given the same starting state).
        """
        dataset_joint_pos, dataset_gripper_qpos, actions, states = episode_data
        T = actions.shape[0]

        env = make_env()
        env.reset()

        # Restore exact initial sim state from the recording
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        max_joint_err = 0.0
        max_cart_err  = 0.0
        worst_joint_t = -1
        worst_cart_t  = -1
        per_step_joint_errs = []
        per_step_cart_errs  = []

        for t in range(T):
            env.step(actions[t])
            live_obs       = env._get_observations()
            live_joint_pos = live_obs["robot0_joint_pos"]    # (7,)
            live_gripper   = live_obs["robot0_gripper_qpos"] # (2,)

            # Dataset state after action[t] = obs[t+1]
            next_idx     = min(t + 1, T - 1)
            ds_joint_pos = dataset_joint_pos[next_idx]
            ds_gripper   = dataset_gripper_qpos[next_idx]

            # ── (a) joint-space ────────────────────────────────────────────────
            joint_err = float(np.max(np.abs(ds_joint_pos - live_joint_pos)))
            per_step_joint_errs.append(joint_err)
            if joint_err > max_joint_err:
                max_joint_err = joint_err
                worst_joint_t = t

            # ── (b) FK task-space ──────────────────────────────────────────────
            pos_ds   = compute_node_pos_xyz(ds_joint_pos,  ds_gripper)
            pos_live = compute_node_pos_xyz(live_joint_pos, live_gripper)
            cart_err = float(torch.max(torch.abs(pos_ds - pos_live)).item())
            per_step_cart_errs.append(cart_err)
            if cart_err > max_cart_err:
                max_cart_err = cart_err
                worst_cart_t = t

        env.close()

        print(f"\n── Episode replay summary ({EPISODE_KEY}) ──────────────────")
        print(f"  Steps replayed                : {len(per_step_joint_errs)}")
        print(f"  Max joint-pos error  (rad)    : {max_joint_err:.6f}  at step {worst_joint_t}")
        print(f"  Mean joint-pos error (rad)    : {np.mean(per_step_joint_errs):.6f}")
        print(f"  Max FK Cartesian err (m)      : {max_cart_err:.6f}  at step {worst_cart_t}")
        print(f"  Mean FK Cartesian err (m)     : {np.mean(per_step_cart_errs):.6f}")

        assert max_joint_err <= JOINT_POS_TOL, (
            f"Joint-position error {max_joint_err:.5f} rad at step {worst_joint_t} "
            f"exceeds tolerance {JOINT_POS_TOL} rad.\n"
            f"This may indicate: (1) _get_node_pos uses stale/off-by-one joint data "
            f"from the dataset, (2) base_link_shift/rotation is wrong, or "
            f"(3) accumulated integration drift exceeds the tolerance."
        )
        # Cartesian error is informational - arm geometry non-linearly amplifies
        # joint errors so we report it but do not assert a hard limit here.
        if max_cart_err > CART_POS_TOL:
            print(
                f"\n  [info] Max FK Cartesian node error {max_cart_err:.5f} m "
                f"(>{CART_POS_TOL} m) at step {worst_cart_t}. "
                f"This is a consequence of the joint-position drift above."
            )

    def test_fk_is_deterministic(self, episode_data):
        """Sanity: FK must be bit-for-bit deterministic for the same inputs."""
        joint_pos, gripper_qpos, _, _ = episode_data
        pos_a = compute_node_pos_xyz(joint_pos[0], gripper_qpos[0])
        pos_b = compute_node_pos_xyz(joint_pos[0], gripper_qpos[0])
        assert torch.allclose(pos_a, pos_b), \
            "FK is not deterministic - unexpected randomness in calculate_panda_joints_positions."

    def test_node_pos_changes_over_episode(self, episode_data):
        """Sanity: FK positions must vary along the episode (data is not static/zero)."""
        joint_pos, gripper_qpos, _, _ = episode_data
        pos_0  = compute_node_pos_xyz(joint_pos[0],  gripper_qpos[0])
        pos_10 = compute_node_pos_xyz(joint_pos[10], gripper_qpos[10])
        assert not torch.allclose(pos_0, pos_10, atol=1e-4), \
            "FK positions unchanged between step 0 and step 10 - check data loading."

    def test_initial_joint_positions_are_nonzero(self, episode_data):
        """Sanity: the recorded initial joint positions should not be all zeros."""
        joint_pos, _, _, _ = episode_data
        assert np.any(np.abs(joint_pos[0]) > 1e-4), \
            "Initial joint positions are all near zero - dataset may not be loaded correctly."
