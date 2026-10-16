"""PyBullet scene: UR5 with a Robotiq 2F-85, two totes, the target and the collapsed pile of pieces."""
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MotionResult:
    reached: bool
    elapsed: float
    max_joint_error: float
    final_joints: List[float] = field(default_factory=list)
    timed_out: bool = False


class PyBulletSim:

    DEFAULT_MAX_JOINT_VELOCITY = 3.0

    def __init__(
        self,
        gui=True,
        pile_count=200,
        settle_steps=3000,
        seed=13,

        target_urdf=None,
        cube_urdf=None,
        tote_urdf=None,

        side_in_tote="x-",
        base_grid=(4, 3),
        layer_count=10,
        spacing=0.038,
        layer_height=0.035,
        lean_per_layer=0.012,
        tilt_deg_min=2.0,
        tilt_deg_max=6.0,
        low_friction=0.5,
        regular_stack=False,
        semi_tilt_max=1.5,

        pile_style="wall",
        mound_count=9,
        mound_radius=0.03,
        mound_base_x=0.041,
        mound_rows=1,
        scatter_count=12,
        scatter_min_dist=0.078,
        scatter_adjacent=None,
        mound_offset_y=0.0,
        mound_end_feet=False,
        mound_cols=3,
        aval_count=90,
        aval_batch=6,
        aval_radius=0.075,
        aval_bury_batches=5,
        aval_height=0.26,
        aval_tilt_deg=35.0,
        aval_batch_steps=150,
        aval_slide_speed=0.0,
        aval_slide_from=0.16,
        hybrid_target_margin=0.05,
        restore_poses=None,
        heterogeneous=False,
        heterogeneous_shape_weights=None,

        hidden_target=False,
        randomize_target=True,
        target_jitter=0.045,
        randomize_side=True,
        require_buried=True,
        buried_thresh=0.05,

        n_clusters=1,
        cubes_per_cluster=18,
        cluster_grid=3,
        cluster_spacing=0.042,
        cluster_min_dist=0.14,
    ):
        self._rng = np.random.default_rng(seed)
        self._sim_steps = 0

        self._hidden_target = bool(hidden_target)
        self._randomize_target = bool(randomize_target)
        self._target_jitter = float(target_jitter)
        self._require_buried = bool(require_buried)
        self._buried_thresh = float(buried_thresh)
        self._initial_exposure = None
        self._pile_style = str(pile_style)
        self._mound_count = int(mound_count)
        self._mound_radius = float(mound_radius)
        self._mound_base_x = float(mound_base_x)
        self._hybrid_target_margin = float(hybrid_target_margin)
        self._scatter_adjacent = scatter_adjacent
        self._mound_offset_y = float(mound_offset_y)
        self._mound_end_feet = bool(mound_end_feet)
        self._mound_cols = int(mound_cols)
        self._aval_count = int(aval_count)
        self._aval_batch = int(aval_batch)
        self._aval_radius = float(aval_radius)
        self._aval_bury_batches = int(aval_bury_batches)
        self._aval_height = float(aval_height)
        self._aval_tilt_deg = float(aval_tilt_deg)
        self._aval_batch_steps = int(aval_batch_steps)
        self._aval_slide_speed = float(aval_slide_speed)
        self._aval_slide_from = float(aval_slide_from)
        self._restore_poses = restore_poses
        self._construct_fast = False
        self._wall_settle_report = None
        self._wall_cube_ids = []
        self._mound_rows = int(mound_rows)
        self._scatter_count = int(scatter_count)
        self._scatter_min_dist = float(scatter_min_dist)
        self._n_clusters = int(n_clusters)
        self._cubes_per_cluster = int(cubes_per_cluster)
        self._cluster_grid = int(cluster_grid)
        self._cluster_spacing = float(cluster_spacing)
        self._cluster_min_dist = float(cluster_min_dist)
        self._cluster_centers = None
        self._hider_idx = None
        if self._randomize_target and bool(randomize_side):
            side_in_tote = str(self._rng.choice(["x-", "x+", "y-", "y+"]))
        self._side_in_tote = side_in_tote

        _assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        if target_urdf is None:
            target_urdf = os.path.join(_assets, "objects", "dumbbell.urdf")
        if cube_urdf is None:
            cube_urdf = os.path.join(_assets, "objects", "cube.urdf")
        if tote_urdf is None:
            tote_urdf = os.path.join(_assets, "tote", "toteA_tall.urdf")

        self._cube_urdf = cube_urdf

        self._regular_stack = regular_stack
        self._semi_tilt_max = semi_tilt_max
        self._heterogeneous = heterogeneous
        self._heterogeneous_shape_weights = heterogeneous_shape_weights

        self._workspace1_bounds = np.array([
            [-0.6, -0.1],
            [-0.25, 0.25],
            [0.0, 0.6],
        ])
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)


        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=150)

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        ur5_urdf = os.path.join(assets_dir, "ur5", "ur5.urdf")
        mount_urdf = os.path.join(assets_dir, "ur5", "mount.urdf")

        self.robot_body_id = p.loadURDF(
            ur5_urdf,
            [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0])
        )
        self._mount_body_id = p.loadURDF(
            mount_urdf,
            [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0])
        )

        self._gripper_body_id = None
        self.robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        robot_joint_info = [p.getJointInfo(self.robot_body_id, i)
                            for i in range(p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._joint_epsilon = 1e-3

        self.robot_home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.robot_goal_joint_config = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.move_joints(self.robot_home_joint_config, speed=1.0)

        self.load_gripper()
        self.open_gripper()

        self._toteA_pos = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2.0
        self._toteA_pos[2] = 0.01
        self._toteA_id = p.loadURDF(
            tote_urdf, self._toteA_pos, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True
        )

        self._toteB_pos = (self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2.0
        self._toteB_pos[2] = 0.01
        self._toteB_id = p.loadURDF(
            tote_urdf, self._toteB_pos, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True
        )

        block_urdf = os.path.join(assets_dir, "obstacles", "block.urdf")
        self.obstacles = [
            p.loadURDF(block_urdf,
                       basePosition=[0, 0.6, 0.9], useFixedBase=True),
            p.loadURDF(block_urdf,
                       basePosition=[0, 0.6, 0.1], useFixedBase=True),
            p.loadURDF(block_urdf,
                       basePosition=[0.45, 0.6, 0.5], useFixedBase=True)
        ]

        self._construct_fast = self._restore_poses is not None
        _multi = (self._n_clusters > 1 and self._hidden_target)
        if _multi:
            self._build_multi_cluster(
                target_urdf=target_urdf,
                target_orn=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
                layer_h=layer_height,
                tilt_deg_min=tilt_deg_min, tilt_deg_max=tilt_deg_max,
                low_friction=low_friction,
            )
        if not _multi:
            if side_in_tote == "x-":
                dumb_offset = np.array([-0.05, 0.0, 0.0])
            elif side_in_tote == "x+":
                dumb_offset = np.array([0.05, 0.0, 0.0])
            elif side_in_tote == "y-":
                dumb_offset = np.array([0.0, -0.05, 0.0])
            else:
                dumb_offset = np.array([0.0, 0.05, 0.0])

            if self._randomize_target and self._pile_style == "avalanche":
                amin_t, amax_t = p.getAABB(self._toteA_id)
                mm = 0.10
                tx = float(self._rng.uniform(amin_t[0] + mm, amax_t[0] - mm))
                ty = float(self._rng.uniform(amin_t[1] + mm, amax_t[1] - mm))
                dumbbell_pos = [tx, ty, 0.013]
            elif self._randomize_target and self._pile_style == "hybrid":
                amin_t, amax_t = p.getAABB(self._toteA_id)
                mm = 0.09
                c = np.array(self._toteA_pos)[:2]
                xlo, xhi = amin_t[0] + mm, amax_t[0] - mm
                ylo, yhi = amin_t[1] + mm, amax_t[1] - mm
                keep = self._hybrid_target_margin
                if side_in_tote == "x-":
                    xlo = max(xlo, c[0] + keep)
                elif side_in_tote == "x+":
                    xhi = min(xhi, c[0] - keep)
                elif side_in_tote == "y-":
                    ylo = max(ylo, c[1] + keep)
                else:
                    yhi = min(yhi, c[1] - keep)
                tx = float(self._rng.uniform(min(xlo, xhi), max(xlo, xhi)))
                ty = float(self._rng.uniform(min(ylo, yhi), max(ylo, yhi)))
                dumbbell_pos = [tx, ty, 0.013]
            elif self._randomize_target and self._pile_style == "mound":
                amin_t, amax_t = p.getAABB(self._toteA_id)
                mm = 0.09
                tx = float(self._rng.uniform(amin_t[0] + mm, amax_t[0] - mm))
                ty = float(self._rng.uniform(amin_t[1] + mm, amax_t[1] - mm))
                dumbbell_pos = [tx, ty, 0.013]
            elif self._randomize_target:
                base_xy, _sx, _sd = self._compute_base_positions(side_in_tote, base_grid, spacing)
                m = 0.4 * float(spacing)
                xmin, ymin = base_xy.min(axis=0)
                xmax, ymax = base_xy.max(axis=0)
                xlo, xhi = sorted([xmin + m, xmax - m])
                ylo, yhi = sorted([ymin + m, ymax - m])
                tx = float(self._rng.uniform(xlo, xhi))
                ty = float(self._rng.uniform(ylo, yhi))
                dumbbell_pos = [tx, ty, 0.013]
            else:
                if getattr(self, '_regular_stack', False):
                    _j = self._rng.uniform(-0.0, 0.0, size=2)
                    dumb_offset = dumb_offset + np.array([_j[0], _j[1], 0.0])
                dumbbell_pos = [
                    float(self._toteA_pos[0] + dumb_offset[0]),
                    float(self._toteA_pos[1] + dumb_offset[1]),
                    0.013,
                ]
            dumbbell_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])

            try:
                _hit = p.rayTest([dumbbell_pos[0], dumbbell_pos[1], 0.5], [dumbbell_pos[0], dumbbell_pos[1], -0.1])[0]
                _floor_top = float(_hit[3][2]) if _hit[0] == self._toteA_id else 0.0168
            except Exception:
                _floor_top = 0.0168
            dumbbell_pos[2] = _floor_top + 0.0125 + 0.001
            self._target_spawn_z = float(dumbbell_pos[2])

            self._target_body_id = p.loadURDF(
                target_urdf, dumbbell_pos, dumbbell_orn, useFixedBase=False
            )
            try:
                p.changeVisualShape(self._target_body_id, -1, rgbaColor=[0.85, 0.1, 0.1, 1.0])
            except Exception:
                pass
            try:
                p.changeDynamics(self._target_body_id, -1,
                                 linearDamping=0.1,
                                 angularDamping=0.2)
            except Exception:
                pass

            self._objects_body_ids = []
            self._object_colors = get_tableau_palette()
            if self._pile_style == "mound":
                self._build_mound_pile(low_friction=low_friction)
            elif self._pile_style == "avalanche":
                self._build_avalanche_pile(low_friction=low_friction)
            elif self._pile_style == "hybrid":
                self._build_in_tote_side_pile(
                    tote_center=np.array(self._toteA_pos),
                    side_in_tote=side_in_tote,
                    base_grid=base_grid,
                    layer_count=layer_count,
                    spacing=spacing,
                    layer_h=layer_height,
                    lean_per_layer=lean_per_layer,
                    pile_count=pile_count,
                    tilt_deg_min=tilt_deg_min,
                    tilt_deg_max=tilt_deg_max,
                    low_friction=low_friction
                )
                for _ in range(int(settle_steps)):
                    self.step_simulation(1)
                self._wall_settle_report = self.settle_until_quiescent()
                self._wall_cube_ids = list(self._objects_body_ids)
                if self._construct_fast:
                    self._apply_saved_poses(self._restore_poses, partial=True)
                else:
                    self._wall_poses_snapshot = [
                        {"id": int(b), "pos": [float(x) for x in p.getBasePositionAndOrientation(b)[0]],
                         "orn": [float(x) for x in p.getBasePositionAndOrientation(b)[1]]}
                        for b in [int(self._target_body_id)] + [int(b) for b in self._objects_body_ids]]
                self._build_mound_pile(low_friction=low_friction)
            else:
                self._build_in_tote_side_pile(
                    tote_center=np.array(self._toteA_pos),
                    side_in_tote=side_in_tote,
                    base_grid=base_grid,
                    layer_count=layer_count,
                    spacing=spacing,
                    layer_h=layer_height,
                    lean_per_layer=lean_per_layer,
                    pile_count=pile_count,
                    tilt_deg_min=tilt_deg_min,
                    tilt_deg_max=tilt_deg_max,
                    low_friction=low_friction
                )

        for _ in range(int(settle_steps)):
            self.step_simulation(1)
        self.settle_until_quiescent()
        if self._construct_fast:
            self._construct_fast = False
            self._apply_saved_poses(self._restore_poses)

        if self._hidden_target and self._require_buried:
            e0 = self._target_exposure_oracle()
            self._initial_exposure = float(e0)
            if e0 >= self._buried_thresh:
                raise RuntimeError(
                    f"P0 burial check FAILED: initial target exposure {e0:.3f} "
                    f">= buried_thresh {self._buried_thresh} "
                    f"(seed={seed}, side={self._side_in_tote}). Target not buried -> "
                    f"scene is trivial. Lower target_jitter or resample this seed.")
    def save_poses(self, path: str) -> None:
        import json as _json
        bodies = [int(self._target_body_id)] + [int(b) for b in self._objects_body_ids]
        out = {"bodies": [], "settle": getattr(self, "_settle_report", None),
               "wall_bodies": getattr(self, "_wall_poses_snapshot", None)}
        for b in bodies:
            pos, orn = p.getBasePositionAndOrientation(b)
            out["bodies"].append({"id": b, "pos": [float(x) for x in pos], "orn": [float(x) for x in orn]})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _json.dump(out, open(path, "w"))

    def _apply_saved_poses(self, path: str, partial: bool = False) -> None:
        import json as _json
        data = _json.load(open(path))
        bodies = [int(self._target_body_id)] + [int(b) for b in self._objects_body_ids]
        saved = data["bodies"]
        if partial:
            saved = (data.get("wall_bodies") or saved)[:len(bodies)]
        if len(saved) != len(bodies) or any(int(r["id"]) != b for r, b in zip(saved, bodies)):
            raise RuntimeError(f"scene cache {path} does not match this build ({len(saved)} vs {len(bodies)} bodies)")
        for r, b in zip(saved, bodies):
            p.resetBasePositionAndOrientation(b, r["pos"], r["orn"])
            p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])
        if partial:
            return
        self.step_simulation(240)
        self._settle_report = self.settle_until_quiescent()
        self._settle_report["restored_from"] = path

    def settle_until_quiescent(self, move_tol: float = 5e-4, window_steps: int = 240,
                               quiet_windows: int = 2, max_steps: int = 14400) -> dict:
        ids = list(getattr(self, "_objects_body_ids", []) or [])
        if not ids:
            return {"steps": 0, "quiescent": True, "max_move": 0.0}
        prev = {b: np.array(p.getBasePositionAndOrientation(b)[0]) for b in ids}
        done, quiet, last_move = 0, 0, 0.0
        while done < max_steps:
            self.step_simulation(window_steps)
            done += window_steps
            cur = {b: np.array(p.getBasePositionAndOrientation(b)[0]) for b in ids}
            last_move = max(float(np.linalg.norm(cur[b] - prev[b])) for b in ids)
            prev = cur
            quiet = quiet + 1 if last_move < move_tol else 0
            if quiet >= quiet_windows:
                break
        self._settle_report = {"steps": done, "quiescent": quiet >= quiet_windows,
                               "max_move": last_move}
        return self._settle_report

    def load_gripper(self):
        if self._gripper_body_id is not None:
            return
        gripper_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gripper", "robotiq_2f_85.urdf")
        self._gripper_body_id = p.loadURDF(gripper_urdf)
        p.resetBasePositionAndOrientation(self._gripper_body_id, [0.5, 0.1, 0.2],
                                          p.getQuaternionFromEuler([np.pi, 0, 0]))
        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index,
                           self._gripper_body_id, 0,
                           jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=self._robot_tool_offset,
                           childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        self.step_simulation(1e3)
        self._calibrate_gripper_kinematics()

    _GRIPPER_TABLE_ANGLES = tuple(round(0.05 * i, 2) for i in range(17))

    def _calibrate_gripper_kinematics(self):
        g = self._gripper_body_id
        n = p.getNumJoints(g)
        self._gripper_kin_table = {}
        for ang in self._GRIPPER_TABLE_ANGLES:
            p.setJointMotorControl2(g, 1, p.POSITION_CONTROL, targetPosition=ang,
                                    force=200, maxVelocity=1.0)
            self.step_simulation(300)
            self._gripper_kin_table[ang] = [p.getJointState(g, j)[0] for j in range(n)]
        p.setJointMotorControl2(g, 1, p.VELOCITY_CONTROL, targetVelocity=-2.0, force=200)
        self.step_simulation(400)

    def gripper_driver_angle_for_gap(self, inner_gap_m: float) -> float:
        angle = (0.0808 - float(inner_gap_m)) / 0.1033
        return max(0.0, min(0.80, angle))

    def gripper_joint_positions(self):
        g = self._gripper_body_id
        if g is None:
            return []
        return [p.getJointState(g, j)[0] for j in range(p.getNumJoints(g))]

    def set_gripper_joints_kinematic(self, inner_gap_m: float):
        g = self._gripper_body_id
        table = getattr(self, "_gripper_kin_table", None)
        if g is None or not table:
            return
        angle = self.gripper_driver_angle_for_gap(inner_gap_m)
        keys = sorted(table)
        lo = max(k for k in keys if k <= angle + 1e-9)
        hi = min(k for k in keys if k >= angle - 1e-9)
        t = 0.0 if hi == lo else (angle - lo) / (hi - lo)
        for j, (a, b) in enumerate(zip(table[lo], table[hi])):
            p.resetJointState(g, j, (1 - t) * a + t * b)

    def restore_gripper_joints(self, positions):
        g = self._gripper_body_id
        if g is None:
            return
        for j, q in enumerate(positions):
            p.resetJointState(g, j, q)

    def move_joints(self, target_joint_state, speed=0.1, timeout: float = 5.0,
                     pos_tol: float = 0.01, max_velocity: Optional[float] = None) -> MotionResult:
        assert len(self._robot_joint_indices) == len(target_joint_state)
        kp = max(float(speed), 0.03)

        vmax = self.DEFAULT_MAX_JOINT_VELOCITY if max_velocity is None else float(max_velocity)
        start = [p.getJointState(self.robot_body_id, j)[0] for j in self._robot_joint_indices]
        delta = [t - s for t, s in zip(target_joint_state, start)]
        dt = 1.0 / 240.0
        n_ramp = int(math.ceil(max(abs(d) for d in delta) / (vmax * dt))) if vmax > 0 else 0
        ramp_i = 0

        def _command(frac):
            sp = [s + d * frac for s, d in zip(start, delta)]
            p.setJointMotorControlArray(
                self.robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                sp, positionGains=kp * np.ones(len(self._robot_joint_indices)),
            )

        _command(1.0 if n_ramp <= 1 else 1.0 / n_ramp)

        t0 = self.now()
        curr = list(target_joint_state)
        while True:
            try:
                curr = [p.getJointState(self.robot_body_id, j)[0]
                        for j in self._robot_joint_indices]
            except Exception:
                return MotionResult(reached=False, elapsed=self.now() - t0,
                                     max_joint_error=float("inf"),
                                     final_joints=list(curr), timed_out=True)

            max_err = max(abs(curr[i] - target_joint_state[i])
                          for i in range(len(self._robot_joint_indices)))
            elapsed = self.now() - t0
            if max_err <= float(pos_tol):
                return MotionResult(reached=True, elapsed=elapsed,
                                     max_joint_error=float(max_err),
                                     final_joints=list(curr), timed_out=False)

            if elapsed > float(timeout):
                print(f"Timeout: >{timeout}s to reach target (max_err={max_err:.4f} rad, tol={pos_tol})")
                return MotionResult(reached=False, elapsed=elapsed,
                                     max_joint_error=float(max_err),
                                     final_joints=list(curr), timed_out=True)

            self.step_simulation(1)
            if ramp_i < n_ramp:
                ramp_i += 1
                _command(min(1.0, (ramp_i + 1) / n_ramp) if n_ramp > 1 else 1.0)

    def move_tool(self, position, orientation, speed=0.1, timeout: float = 5.0,
                  pos_tol: float = 0.01) -> MotionResult:
        target_joint_state = p.calculateInverseKinematics(
            self.robot_body_id, self.robot_end_effector_link_index,
            position, orientation,
            maxNumIterations=200, residualThreshold=1e-4
        )
        return self.move_joints(target_joint_state, speed=speed, timeout=timeout, pos_tol=pos_tol)

    def robot_go_home(self, speed=0.2):
        self.move_joints(self.robot_home_joint_config, speed)

    def close_gripper(self):
        if self._gripper_body_id is None:
            return None
        p.setJointMotorControl2(self._gripper_body_id, 1, p.VELOCITY_CONTROL,
                                targetVelocity=2.0, force=200)
        self.step_simulation(4e2)
        try:
            return p.getJointState(self._gripper_body_id, 1)[0]
        except Exception:
            return None

    def hold_gripper_gently(self):
        if self._gripper_body_id is None:
            return
        try:
            cur_angle = p.getJointState(self._gripper_body_id, 1)[0]
        except Exception:
            return
        p.setJointMotorControl2(self._gripper_body_id, 1, p.POSITION_CONTROL,
                                targetPosition=cur_angle, positionGain=0.9,
                                velocityGain=1.0, force=200)

    def set_gripper_opening(self, inner_gap_m: float, settle_steps: int = 300):
        if self._gripper_body_id is None:
            return None
        angle = (0.0808 - float(inner_gap_m)) / 0.1033
        angle = max(0.0, min(0.80, angle))
        p.setJointMotorControl2(self._gripper_body_id, 1, p.POSITION_CONTROL,
                                targetPosition=angle, force=200, maxVelocity=1.0)
        self.step_simulation(int(settle_steps))
        try:
            return p.getJointState(self._gripper_body_id, 1)[0]
        except Exception:
            return None

    def open_gripper(self):
        if self._gripper_body_id is None:
            return None
        p.setJointMotorControl2(self._gripper_body_id, 1, p.VELOCITY_CONTROL,
                                targetVelocity=-2.0, force=200)
        self.step_simulation(4e2)
        try:
            return p.getJointState(self._gripper_body_id, 1)[0]
        except Exception:
            return None

    def execute_grasp(self, grasp_position, grasp_yaw, approach_height=0.25, grasp_height=0.05):
        x, y, z = grasp_position
        tool_tip = np.array([x, y, z]) + self._tool_tip_to_ee_joint
        orn = p.getQuaternionFromEuler([math.pi, 0, grasp_yaw])
        self.move_tool([tool_tip[0], tool_tip[1], z + approach_height], orn, speed=0.15)
        self.move_tool([tool_tip[0], tool_tip[1], z + grasp_height], orn, speed=0.08)
        self.close_gripper()
        self.step_simulation(2e2)
        return True

    def now(self) -> float:
        return float(self._sim_steps) / 240.0

    def step_simulation(self, num_steps):
        if getattr(self, "_construct_fast", False):
            return
        for _ in range(int(num_steps)):
            p.stepSimulation()
            self._sim_steps += 1

            if self._gripper_body_id is None:
                continue

            if not p.isConnected():
                self._gripper_body_id = None
                continue

            try:
                num_joints = p.getNumJoints(self._gripper_body_id)
                if num_joints <= 0:
                    self._gripper_body_id = None
                    continue

                gp = np.array(
                    [p.getJointState(self._gripper_body_id, i)[0]
                     for i in range(num_joints)]
                )

                p.setJointMotorControlArray(
                    self._gripper_body_id,
                    [6, 3, 8, 5, 10],
                    p.POSITION_CONTROL,
                    [gp[1], -gp[1], -gp[1], gp[1], gp[1]],
                    positionGains=0.3 * np.ones(5)
                )
            except Exception:
                self._gripper_body_id = None
                continue

    def _build_multi_cluster(self, target_urdf, target_orn, layer_h,
                             tilt_deg_min, tilt_deg_max, low_friction):
        amin, amax = p.getAABB(self._toteA_id)
        margin = 0.035
        xlo, xhi = float(amin[0]) + margin, float(amax[0]) - margin
        ylo, yhi = float(amin[1]) + margin, float(amax[1]) - margin
        N = self._n_clusters
        min_d = self._cluster_min_dist

        centers, tries = [], 0
        while len(centers) < N and tries < 5000:
            tries += 1
            c = np.array([self._rng.uniform(xlo, xhi), self._rng.uniform(ylo, yhi)])
            if all(np.linalg.norm(c - o) >= min_d for o in centers):
                centers.append(c)
        if len(centers) < N:
            raise RuntimeError(
                f"multi-cluster: only fit {len(centers)}/{N} piles at min_dist={min_d} "
                f"in tote interior [{xlo:.2f},{xhi:.2f}]x[{ylo:.2f},{yhi:.2f}]. "
                f"Reduce n_clusters or cluster_min_dist, or the tote is too small.")
        self._cluster_centers = [tuple(map(float, c)) for c in centers]
        self._hider_idx = int(self._rng.integers(N))
        hider = centers[self._hider_idx]

        self._target_body_id = p.loadURDF(
            target_urdf, [float(hider[0]), float(hider[1]), 0.013], target_orn,
            useFixedBase=False)
        try:
            p.changeVisualShape(self._target_body_id, -1, rgbaColor=[0.85, 0.1, 0.1, 1.0])
        except Exception:
            pass
        try:
            p.changeDynamics(self._target_body_id, -1, linearDamping=0.1, angularDamping=0.2)
        except Exception:
            pass

        self._objects_body_ids = []
        self._object_colors = get_tableau_palette()
        g = max(1, self._cluster_grid)
        sp = self._cluster_spacing
        npc = self._cubes_per_cluster
        idx = 0
        for c in centers:
            placed = 0
            layer = 0
            while placed < npc:
                for gx in range(g):
                    for gy in range(g):
                        if placed >= npc:
                            break
                        jx, jy = self._rng.uniform(-0.004, 0.004, size=2)
                        ox = (gx - (g - 1) / 2.0) * sp + jx
                        oy = (gy - (g - 1) / 2.0) * sp + jy
                        z = 0.02 + layer * layer_h
                        pos = [float(c[0] + ox), float(c[1] + oy), float(z)]
                        yaw = float(self._rng.uniform(-10.0, 10.0))
                        tilt = float(self._rng.uniform(tilt_deg_min, tilt_deg_max))
                        if self._rng.random() < 0.5:
                            euler = [math.radians(tilt), 0.0, math.radians(yaw)]
                        else:
                            euler = [0.0, math.radians(tilt), math.radians(yaw)]
                        orn = p.getQuaternionFromEuler(euler)
                        bid = p.loadURDF(self._cube_urdf, basePosition=pos,
                                         baseOrientation=orn, useFixedBase=False)
                        self._objects_body_ids.append(bid)
                        col = self._object_colors[idx % len(self._object_colors)]
                        try:
                            p.changeVisualShape(bid, -1, rgbaColor=(*col, 1.0))
                        except Exception:
                            pass
                        p.changeDynamics(bid, -1, lateralFriction=low_friction,
                                         spinningFriction=0.005, rollingFriction=0.005,
                                         linearDamping=0.05, angularDamping=0.1,
                                         frictionAnchor=True)
                        placed += 1
                        idx += 1
                layer += 1

    def get_cluster_info_oracle(self):
        return {"centers": self._cluster_centers, "hider_idx": self._hider_idx}

    def _build_avalanche_pile(self, low_friction):
        tpos = np.array(p.getBasePositionAndOrientation(self._target_body_id)[0])
        amin, amax = p.getAABB(self._toteA_id)
        margin = 0.055
        xlo, xhi = float(amin[0]) + margin, float(amax[0]) - margin
        ylo, yhi = float(amin[1]) + margin, float(amax[1]) - margin
        floor_z = 0.0168
        idx = 0
        sample_spec = build_from_spec = diverse_colors = None
        if self._heterogeneous:
            from rubble_generator import sample_rubble_spec as sample_spec
            from rubble_generator import build_rubble_from_spec as build_from_spec
            from rubble_generator import get_diverse_palette
            diverse_colors = get_diverse_palette(32)
        n_batches = max(1, int(np.ceil(self._aval_count / max(1, self._aval_batch))))
        made = 0
        for b in range(n_batches):
            batch_xy = []
            for _ in range(min(self._aval_batch, self._aval_count - made)):
                for _try in range(60):
                    if b < self._aval_bury_batches and self._aval_slide_speed > 0.0:
                        xy = tpos[:2] + np.array([-self._aval_slide_from + float(self._rng.uniform(-0.05, 0.05)),
                                                  float(self._rng.uniform(-self._aval_radius, self._aval_radius))])
                    elif b < self._aval_bury_batches:
                        r = float(self._aval_radius * np.sqrt(self._rng.random()))
                        th = float(self._rng.uniform(0.0, 2 * np.pi))
                        xy = tpos[:2] + r * np.array([np.cos(th), np.sin(th)])
                    else:
                        xy = np.array([self._rng.uniform(xlo, xhi), self._rng.uniform(ylo, yhi)])
                    xy[0] = float(np.clip(xy[0], xlo, xhi))
                    xy[1] = float(np.clip(xy[1], ylo, yhi))
                    if all(np.linalg.norm(xy - q) > 0.052 for q in batch_xy):
                        batch_xy.append(np.array(xy, dtype=float))
                        break
            if not batch_xy:
                continue
            tops = []
            for xy in batch_xy:
                hit = p.rayTest([float(xy[0]), float(xy[1]), 0.9], [float(xy[0]), float(xy[1]), -0.05])[0]
                tops.append(float(hit[3][2]) if int(hit[0]) >= 0 else floor_z)
            z_drop = max(max(tops), floor_z) + self._aval_height
            for xy in batch_xy:
                t = np.deg2rad(self._aval_tilt_deg)
                euler = [float(self._rng.uniform(-t, t)), float(self._rng.uniform(-t, t)),
                         float(self._rng.uniform(-np.pi, np.pi))]
                orn = p.getQuaternionFromEuler(euler)
                if self._heterogeneous:
                    spec = sample_spec(self._rng, shape_weights=self._heterogeneous_shape_weights,
                                       max_horizontal=0.040)
                    rgba = (*diverse_colors[idx % len(diverse_colors)], 1.0)
                    bid = build_from_spec(spec, [float(xy[0]), float(xy[1]), float(z_drop)], orn, color=rgba)
                    self._objects_body_ids.append(bid)
                    p.changeDynamics(bid, -1, lateralFriction=low_friction, frictionAnchor=True)
                else:
                    bid = p.loadURDF(self._cube_urdf, basePosition=[float(xy[0]), float(xy[1]), float(z_drop)],
                                     baseOrientation=orn, useFixedBase=False)
                    self._objects_body_ids.append(bid)
                    col = self._object_colors[idx % len(self._object_colors)]
                    try:
                        p.changeVisualShape(bid, -1, rgbaColor=(*col, 1.0))
                    except Exception:
                        pass
                    p.changeDynamics(bid, -1, lateralFriction=low_friction,
                                     spinningFriction=0.005, rollingFriction=0.005,
                                     linearDamping=0.05, angularDamping=0.1,
                                     frictionAnchor=True)
                if b < self._aval_bury_batches and self._aval_slide_speed > 0.0:
                    p.resetBaseVelocity(bid, [float(self._aval_slide_speed), float(self._rng.uniform(-0.05, 0.05)), 0.0],
                                        [0.0, 0.0, float(self._rng.uniform(-2.0, 2.0))])
                idx += 1
                made += 1
            self.step_simulation(int(self._aval_batch_steps))
        return

    def _build_mound_pile(self, low_friction):
        tpos = np.array(p.getBasePositionAndOrientation(self._target_body_id)[0])
        amin, amax = p.getAABB(self._toteA_id)
        margin = 0.045 if self._pile_style != "hybrid" else 0.036
        xlo, xhi = float(amin[0]) + margin, float(amax[0]) - margin
        ylo, yhi = float(amin[1]) + margin, float(amax[1]) - margin
        rubble_xy = [np.array(p.getBasePositionAndOrientation(b)[0][:2], dtype=float)
                     for b in self._objects_body_ids]
        placed_xy = []
        idx = len(self._objects_body_ids)

        def _load(xy, z, yaw_max=np.pi / 8, yaw0=0.0):
            nonlocal idx
            yaw = float(yaw0) + float(self._rng.uniform(-yaw_max, yaw_max))
            orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            bid = p.loadURDF(self._cube_urdf, basePosition=[float(xy[0]), float(xy[1]), float(z)],
                             baseOrientation=orn, useFixedBase=False)
            self._objects_body_ids.append(bid)
            col = self._object_colors[idx % len(self._object_colors)]
            try:
                p.changeVisualShape(bid, -1, rgbaColor=(*col, 1.0))
            except Exception:
                pass
            p.changeDynamics(bid, -1, lateralFriction=low_friction,
                             spinningFriction=0.005, rollingFriction=0.005,
                             linearDamping=0.05, angularDamping=0.1,
                             frictionAnchor=True)
            idx += 1
            placed_xy.append(np.array(xy, dtype=float))
            for _ in range(120):
                self.step_simulation(1)

        jit = lambda sd: float(self._rng.normal(0.0, sd))
        bx = self._mound_base_x
        if self._mound_rows <= 1:
            heap = [([-bx + jit(0.0015), -0.025 + jit(0.002)], 0.032),
                    ([+bx + jit(0.0015), -0.025 + jit(0.002)], 0.032),
                    ([-bx + jit(0.0015), +0.025 + jit(0.002)], 0.032),
                    ([+bx + jit(0.0015), +0.025 + jit(0.002)], 0.032),
                    ([jit(0.003), -0.025 + jit(0.003)], 0.075),
                    ([jit(0.003), +0.025 + jit(0.003)], 0.075),
                    ([jit(0.004), jit(0.005)], 0.121)]
            top_z = 0.121
        else:
            heap = []
            oy = self._mound_offset_y
            bxo = 0.034 if abs(oy) < 1e-6 else 0.036
            ncol = int(self._mound_cols)
            pitch = 0.044
            cols = [(k - (ncol - 1) / 2.0) * pitch for k in range(ncol)]
            if ncol == 3:
                for sy in (-0.06, 0.06):
                    heap.append(([-bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
                    heap.append(([+bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
                heap.append(([-0.040 + jit(0.001), oy + jit(0.001)], 0.032, np.pi / 4))
                heap.append(([+0.040 + jit(0.001), oy + jit(0.001)], 0.032, np.pi / 4))
                end_y = 0.104
            else:
                gaps = [(cols[k] + cols[k + 1]) / 2.0 for k in range(ncol - 1)]
                ends = [cols[0] - 0.022, cols[-1] + 0.022]
                for sy in gaps + ends:
                    heap.append(([-bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
                    heap.append(([+bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
                end_y = cols[-1] + 0.022 + 0.044
            if self._mound_end_feet:
                for sy in (-end_y, end_y):
                    heap.append(([-bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
                    heap.append(([+bxo + jit(0.001), oy + sy + jit(0.001)], 0.032, 0.0))
            z = 0.078
            for r in range(int(self._mound_rows)):
                for u in cols:
                    heap.append(([jit(0.001), oy + u + jit(0.001)], z, 0.0))
                z += 0.047
            top_z = z - 0.047
        for item in heap:
            off, z = item[0], item[1]
            if len(item) > 2:
                _load(tpos[:2] + np.array(off), z, yaw_max=np.pi / 60, yaw0=item[2])
            else:
                _load(tpos[:2] + np.array(off), z, yaw_max=np.pi / 36)
        for k in range(max(0, self._mound_count - len(heap))):
            if self._mound_rows > 1:
                r = float(self._rng.uniform(0.06, 0.06 + self._mound_radius))
                th = float(self._rng.uniform(0.0, 2 * np.pi))
                _load(tpos[:2] + r * np.array([np.cos(th), np.sin(th)]), 0.07 + 0.01 * k)
            else:
                r = float(self._rng.uniform(0.04, 0.04 + self._mound_radius))
                th = float(self._rng.uniform(0.0, 2 * np.pi))
                _load(tpos[:2] + r * np.array([np.cos(th), np.sin(th)]), top_z + 0.03 + 0.05 * k)
        tries = 0
        n_scatter = 0
        n_heap = len(placed_xy)
        n_adjacent = self._scatter_count // 3 if self._scatter_adjacent is None else int(self._scatter_adjacent)
        adj_tries = 0
        max_tries = 20000 if self._pile_style == "hybrid" else 4000
        while n_scatter < self._scatter_count and tries < max_tries:
            tries += 1
            if n_scatter < n_adjacent and adj_tries < 150:
                adj_tries += 1
                sgn = 1.0 if self._rng.random() < 0.5 else -1.0
                xy = tpos[:2] + np.array([sgn * float(self._rng.uniform(0.084, 0.096)),
                                          float(self._rng.uniform(-0.03, 0.03))])
                min_d = 0.045
            else:
                xy = np.array([self._rng.uniform(xlo, xhi), self._rng.uniform(ylo, yhi)])
                min_d = self._scatter_min_dist
                if np.linalg.norm(xy - tpos[:2]) < self._scatter_min_dist + 0.03:
                    continue
            if not (xlo <= xy[0] <= xhi and ylo <= xy[1] <= yhi):
                continue
            if any(np.linalg.norm(xy - q) < (min_d if k < n_heap else (0.046 if rubble_xy else min_d))
                   for k, q in enumerate(placed_xy)):
                continue
            near_rubble = any(np.linalg.norm(xy - q) < 0.05 for q in rubble_xy)
            _load(xy, 0.16 if near_rubble else 0.05)
            n_scatter += 1

    def _compute_base_positions(self, side_in_tote, base_grid, spacing):
        aabb_min, aabb_max = p.getAABB(self._toteA_id)
        tote_half = (np.array(aabb_max) - np.array(aabb_min)) / 2.0
        c = np.array(self._toteA_pos)[:2]
        if side_in_tote == "x-":
            start_xy = np.array([c[0] - tote_half[0] + spacing, c[1]]); side_dir = np.array([1.0, 0.0])
        elif side_in_tote == "x+":
            start_xy = np.array([c[0] + tote_half[0] - spacing, c[1]]); side_dir = np.array([-1.0, 0.0])
        elif side_in_tote == "y-":
            start_xy = np.array([c[0], c[1] - tote_half[1] + spacing]); side_dir = np.array([0.0, 1.0])
        else:
            start_xy = np.array([c[0], c[1] + tote_half[1] - spacing]); side_dir = np.array([0.0, -1.0])
        rows, cols = base_grid
        pos = []
        if side_in_tote in ["x-", "x+"]:
            for r in range(rows):
                for cc in range(cols):
                    xy = start_xy + np.array([0.0, (cc - cols / 2) * spacing]) + side_dir * (r * spacing)
                    pos.append(xy)
        else:
            for r in range(rows):
                for cc in range(cols):
                    xy = start_xy + np.array([(cc - cols / 2) * spacing, 0.0]) + side_dir * (r * spacing)
                    pos.append(xy)
        return np.array(pos, dtype=np.float64), start_xy, side_dir

    def _build_in_tote_side_pile(
        self, tote_center, side_in_tote, base_grid, layer_count, spacing, layer_h,
        lean_per_layer, pile_count, tilt_deg_min, tilt_deg_max, low_friction
    ):
        aabb_min, aabb_max = p.getAABB(self._toteA_id)
        aabb_min, aabb_max = np.array(aabb_min), np.array(aabb_max)
        tote_half = (aabb_max - aabb_min) / 2.0
        tote_center_xy = tote_center[:2]

        base_positions, start_xy, side_dir = self._compute_base_positions(
            side_in_tote, base_grid, spacing)
        base_positions = [np.asarray(b, dtype=np.float64) for b in base_positions]
        rows, cols = base_grid

        target_pos, _ = p.getBasePositionAndOrientation(self._target_body_id)
        target_pos = np.array(target_pos)
        dir_to_target = target_pos[:2] - tote_center_xy
        norm = np.linalg.norm(dir_to_target)
        if norm < 1e-6:
            dir_to_target = side_dir
        else:
            dir_to_target = dir_to_target / norm

        cubes_created = 0
        z0 = target_pos[2] + 0.02

        sample_spec = None
        build_from_spec = None
        diverse_colors = None
        tote_floor_z = None
        if self._heterogeneous:
            from rubble_generator import (
                sample_rubble_spec, build_rubble_from_spec, get_diverse_palette,
            )
            sample_spec = sample_rubble_spec
            build_from_spec = build_rubble_from_spec
            diverse_colors = get_diverse_palette(32)
            try:
                tote_aabb_min, _ = p.getAABB(self._toteA_id)
                tote_floor_z = float(tote_aabb_min[2]) + 0.005
            except Exception:
                tote_floor_z = float(tote_center[2]) + 0.01
            het_max_horizontal = float(spacing) * 0.90

        for layer in range(layer_count):
            if cubes_created >= pile_count:
                break

            layer_offset_xy = ((dir_to_target * 0.0) if self._regular_stack
                               else dir_to_target * lean_per_layer * layer)
            z = z0 + layer * layer_h

            for base_xy in base_positions:
                if cubes_created >= pile_count:
                    break

                if getattr(self, '_regular_stack', False) and self._rng.random() < 0.18:
                    continue
                pos_xy = base_xy + layer_offset_xy

                if self._regular_stack:
                    _t = getattr(self, '_semi_tilt_max', 1.5)
                    yaw_deg = float(self._rng.uniform(-_t, _t))
                    tilt_deg = float(self._rng.uniform(0.0, _t))
                    if self._rng.random() < 0.5:
                        euler = [math.radians(tilt_deg), 0.0, math.radians(yaw_deg)]
                    else:
                        euler = [0.0, math.radians(tilt_deg), math.radians(yaw_deg)]
                else:
                    yaw_deg = float(self._rng.uniform(-10.0, 10.0))
                    tilt_deg = float(self._rng.uniform(tilt_deg_min, tilt_deg_max))
                    if self._rng.random() < 0.5:
                        euler = [math.radians(tilt_deg), 0.0, math.radians(yaw_deg)]
                    else:
                        euler = [0.0, math.radians(tilt_deg), math.radians(yaw_deg)]
                orn = p.getQuaternionFromEuler(euler)

                if self._heterogeneous:
                    spec = sample_spec(
                        self._rng,
                        shape_weights=self._heterogeneous_shape_weights,
                        max_horizontal=het_max_horizontal,
                    )
                    obj_height = float(spec['dims'][2])
                    min_center_z = tote_floor_z + obj_height / 2.0 + 0.002
                    layer_base_z = z0 + layer * layer_h
                    z_for_this_obj = max(min_center_z, layer_base_z)
                    pos = [float(pos_xy[0]), float(pos_xy[1]), float(z_for_this_obj)]

                    color = diverse_colors[cubes_created % len(diverse_colors)]
                    rgba = (*color, 1.0)
                    bid = build_from_spec(spec, pos, orn, color=rgba)
                    self._objects_body_ids.append(bid)
                else:
                    pos = [float(pos_xy[0]), float(pos_xy[1]), float(z)]
                    bid = p.loadURDF(self._cube_urdf, basePosition=pos,
                                     baseOrientation=orn, useFixedBase=False)
                    self._objects_body_ids.append(bid)
                    color = self._object_colors[cubes_created % len(self._object_colors)]
                    try:
                        p.changeVisualShape(bid, -1, rgbaColor=(*color, 1.0))
                    except Exception:
                        pass

                    p.changeDynamics(bid, -1,
                                     lateralFriction=low_friction,
                                     spinningFriction=0.005,
                                     rollingFriction=0.005,
                                     linearDamping=0.05,
                                     angularDamping=0.1,
                                     frictionAnchor=True)
                cubes_created += 1

    def set_joint_positions(self, values):
        assert len(self._robot_joint_indices) == len(values)
        for j, v in zip(self._robot_joint_indices, values):
            p.resetJointState(self.robot_body_id, j, v)

    def _target_exposure_oracle(self, grid_n: int = 9, ray_h: float = 0.6) -> float:
        tid = int(self._target_body_id)
        try:
            amin, amax = p.getAABB(tid)
        except Exception:
            return 0.0
        xs = np.linspace(amin[0], amax[0], grid_n)
        ys = np.linspace(amin[1], amax[1], grid_n)
        top = float(amax[2])
        rf = [[float(x), float(y), top + 1e-3] for x in xs for y in ys]
        rt = [[float(x), float(y), top + ray_h] for x in xs for y in ys]
        try:
            res = p.rayTestBatch(rf, rt)
        except Exception:
            return 0.0
        return sum(1 for r in res if int(r[0]) == -1) / max(1, len(res))

    def get_ids(self):
        ids = {
            "plane": self._plane_id,
            "ur5": self.robot_body_id,
            "mount": self._mount_body_id,
            "toteA": self._toteA_id,
            "toteB": self._toteB_id,
            "obstacles": list(self.obstacles),
            "cubes": list(self._objects_body_ids),
            "gripper": self._gripper_body_id
        }
        if not getattr(self, "_hidden_target", False):
            ids["target"] = self._target_body_id
        return ids

    def get_oracle_ids(self):
        return {"target": self._target_body_id}

    def get_target_xy_oracle(self):
        pos, _ = p.getBasePositionAndOrientation(self._target_body_id)
        return (float(pos[0]), float(pos[1]))

    def _get_ids_legacy(self):
        return {
            "plane": self._plane_id,
            "ur5": self.robot_body_id,
            "mount": self._mount_body_id,
            "toteA": self._toteA_id,
            "toteB": self._toteB_id,
            "target": self._target_body_id,
            "obstacles": list(self.obstacles),
            "cubes": list(self._objects_body_ids),
            "gripper": self._gripper_body_id
        }


def get_tableau_palette():
    palette = np.array(
        [
            [78, 121, 167],
            [242, 142, 43],
            [225, 87, 89],
            [118, 183, 178],
            [89, 161, 79],
            [237, 201, 72],
            [176, 122, 161],
            [255, 157, 167],
            [118, 183, 178],
            [186, 176, 172],
        ],
        dtype=np.float64
    )
    return palette / 255.0


class SphereMarker:
    def __init__(self, position, radius=0.02, rgba_color=(1, 0, 0, 0.6)):
        self.radius = float(radius)
        self.rgba = tuple(rgba_color)
        self._visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=self.rgba
        )
        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self._visual_shape,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

    def set_position(self, position):
        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            p.getQuaternionFromEuler([0, 0, 0])
        )


if __name__ == "__main__":
    sim = PyBulletSim(
        gui=True,
        pile_count=300,
        settle_steps=2000,
        target_urdf=None,
        side_in_tote="x-",
        base_grid=(4, 3),
        layer_count=12,
        spacing=0.035,
        layer_height=0.03,
        lean_per_layer=0.015,
        tilt_deg_min=4.0,
        tilt_deg_max=10.0,
        low_friction=0.25
    )
    camera_distance = 1
    camera_yaw = 135
    camera_pitch = 45
    camera_target_position = [0.5, 0, 0.2]


    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target_position
    )
    ids = sim.get_ids()
    print({k: (len(v) if isinstance(v, list) else v) for k, v in ids.items()})
    for _ in range(600):
        p.stepSimulation()
        time.sleep(1/240.0)
