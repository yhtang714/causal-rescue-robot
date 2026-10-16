"""Arm controller: grasp pipeline with gates C0-C6, the blade push with gate C7 and the compliant descent, motion planning."""
import os
import pybullet as p
import time
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set
import logging

from robot_geometry import RobotGeometry
from sim import MotionResult

try:
    from ompl import base as _ompl_base
    from ompl import geometric as _ompl_geometric
    from ompl import util as _ompl_util
    _ompl_util.RNG.setSeed(20260907)
    _OMPL_AVAILABLE = True
except Exception:
    _ompl_base = None
    _ompl_geometric = None
    _OMPL_AVAILABLE = False


class GraspStatus(Enum):
    SUCCESS = "success"
    FAILED_IK = "failed_ik"
    FAILED_MOTION = "failed_motion"
    FAILED_GRASP = "failed_grasp"
    FAILED_PLACE = "failed_place"
    TIMEOUT = "timeout"
    INVALID_TARGET = "invalid_target"
    OUT_OF_WORKSPACE = "out_of_workspace"


class ControllerState(Enum):
    IDLE = "IDLE"
    PRECHECK = "PRECHECK"
    PREGRASP = "PREGRASP"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    VERIFY_CONTACT = "VERIFY_CONTACT"
    OPTIONAL_MICRO_LIFT = "OPTIONAL_MICRO_LIFT"
    ATTACH = "ATTACH"
    LIFT = "LIFT"
    TRANSIT = "TRANSIT"
    LOWER = "LOWER"
    DETACH = "DETACH"
    OPEN = "OPEN"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETREAT = "RETREAT"
    DONE = "DONE"


class FailureCode(Enum):
    INVALID_TARGET = "INVALID_TARGET"
    NO_FEASIBLE_CANDIDATE = "NO_FEASIBLE_CANDIDATE"
    OUT_OF_WORKSPACE = "OUT_OF_WORKSPACE"
    IK_FAILED = "IK_FAILED"
    FK_RESIDUAL_TOO_HIGH = "FK_RESIDUAL_TOO_HIGH"
    MOTION_TIMEOUT = "MOTION_TIMEOUT"
    PATH_COLLISION = "PATH_COLLISION"
    NO_BILATERAL_CONTACT = "NO_BILATERAL_CONTACT"
    MICRO_LIFT_FAILED = "MICRO_LIFT_FAILED"
    ATTACH_FAILED = "ATTACH_FAILED"
    LIFT_FAILED = "LIFT_FAILED"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    PLACE_FAILED = "PLACE_FAILED"
    RECOVERY_FAILED = "RECOVERY_FAILED"


@dataclass
class StepResult:
    state: ControllerState
    ok: bool
    failure_code: Optional[FailureCode] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    position: Tuple[float, float, float]
    orientation: Optional[Tuple[float, float, float, float]] = None


@dataclass
class GraspRequest:
    target_id: int
    timeout: float = 30.0
    approach_height: float = 0.12
    grasp_height: float = 0.02
    candidate_yaws: Optional[List[float]] = None
    enable_micro_lift: bool = True
    place_layout: str = "opposite"


@dataclass
class GraspResult:
    status: GraspStatus
    error_code: str = ""
    error_message: str = ""
    execution_time: float = 0.0
    final_pose: Optional[Pose] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    failure_code: Optional[FailureCode] = None
    transitions: List[StepResult] = field(default_factory=list)
    bilateral_contact_success: bool = False
    friction_micro_lift_success: Optional[bool] = None
    constraint_assisted_transport_success: bool = False


@dataclass
class PushRequest:
    target_id: int
    dir_index: int = 0
    dist: float = 0.07
    depth: float = 0.015
    standoff: float = 0.022
    timeout: float = 30.0


@dataclass
class PushResult:
    status: GraspStatus
    failure_code: Optional[FailureCode] = None
    error_message: str = ""
    execution_time: float = 0.0
    target_displacement_m: float = 0.0
    max_other_displacement_m: float = 0.0
    n_others_moved_over_1cm: int = 0
    transitions: List[StepResult] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class ProductionRobotController:

    def __init__(self, sim, enable_logging: bool = True,
                 geometry: Optional[RobotGeometry] = None):
        self.sim = sim
        self.robot_id = sim.robot_body_id
        self.end_effector_index = sim.robot_end_effector_link_index
        self.joint_indices = sim._robot_joint_indices

        tip_off = float(getattr(sim, "_tool_tip_to_ee_joint",
                                np.array([0, 0, 0.15]))[2])
        self.geom = geometry or RobotGeometry(tool_tip_offset_z=tip_off)

        self._down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        self._joint_lower, self._joint_upper = [], []
        for j in self.joint_indices:
            info = p.getJointInfo(self.robot_id, j)
            jl, ju = float(info[8]), float(info[9])
            if jl >= ju:
                jl, ju = -math.pi, math.pi
            self._joint_lower.append(jl)
            self._joint_upper.append(ju)
        self._joint_ranges = [u - l for l, u in zip(self._joint_lower, self._joint_upper)]
        self._rest_pose = list(sim.robot_home_joint_config)

        ws1 = sim._workspace1_bounds
        try:
            ws2 = sim._workspace2_bounds
            x_lo = min(float(ws1[0, 0]), float(ws2[0, 0])) - 0.05
            x_hi = max(float(ws1[0, 1]), float(ws2[0, 1])) + 0.05
            y_lo = min(float(ws1[1, 0]), float(ws2[1, 0])) - 0.05
            y_hi = max(float(ws1[1, 1]), float(ws2[1, 1])) + 0.05
        except Exception:
            x_lo, x_hi = float(ws1[0, 0]) - 0.05, float(ws1[0, 1]) + 0.05
            y_lo, y_hi = float(ws1[1, 0]) - 0.05, float(ws1[1, 1]) + 0.05

        self.workspace = {'x': [x_lo, x_hi], 'y': [y_lo, y_hi], 'z': [0.01, 0.55]}
        self.pick_workspace = {
            'x': [float(ws1[0, 0]) - 0.05, float(ws1[0, 1]) + 0.05],
            'y': [float(ws1[1, 0]) - 0.05, float(ws1[1, 1]) + 0.05],
            'z': [0.01, 0.55],
        }

        try:
            tB = sim._toteB_pos
            self.place_tote_center = (float(tB[0]), float(tB[1]), float(tB[2]))
            self.place_tote_id = int(sim._toteB_id)
        except Exception:
            self.place_tote_center = (0.35, 0.0, 0.01)
            self.place_tote_id = None

        self._same_side_tote_id = None
        self._same_side_tote_center = None

        self.logger = logging.getLogger("RobotCtrl")
        if enable_logging and not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s', '%H:%M:%S'))
            self.logger.addHandler(h)

        self._is_grasping = False
        self._grasp_constraint_id = None
        self._last_goal_invalid = False
        self._stats = {
            'total_attempts': 0, 'successful_grasps': 0,
            'failed_grasps': 0, 'failed_places': 0,
        }

        self.logger.info("ProductionRobotController (state-machine) initialized")
        self.logger.info(f"  Tool tip offset z: {self.geom.tool_tip_offset_z:.3f}m")
        self.logger.info(f"  Pick workspace x={self.pick_workspace['x']}")
        self.logger.info(f"  Place tote center (ToteB): {self.place_tote_center}")

    def _move_tip(self, tip_pos, orn=None, kp=0.1, timeout=5.0, pos_tol=0.01):
        if orn is None:
            orn = self._down_orn
        ee_target = self.geom.tip_to_ee(tip_pos)
        return self.sim.move_tool(list(ee_target), orn, speed=kp, timeout=timeout, pos_tol=pos_tol)

    _PLAN_IF_JOINT_DELTA_ABOVE = 1.0

    def _move_to_cfg_planned_if_far(self, goal_cfg, goal_tip, orn, kp, timeout, pos_tol=0.01, clearance=None):
        start_cfg = self._current_joint_state()
        delta = max(abs(a - b) for a, b in zip(goal_cfg, start_cfg))
        if delta > self._PLAN_IF_JOINT_DELTA_ABOVE and _OMPL_AVAILABLE:
            held_id = self._get_held_object_id()
            held_rel = self._held_relative_transform(held_id) if held_id is not None else None
            reason = {}
            waypoints = self._plan_joint_path(
                goal_cfg, held_id=held_id, held_rel=held_rel,
                allowed_ids=[self.sim._plane_id],
                planning_time=min(6.0, max(1.5, float(timeout) * 0.4)),
                start_tip=self._get_tip_pos(), goal_tip=tuple(goal_tip), orn=orn,
                reason=reason, clearance=clearance)
            if waypoints:
                r = self._execute_planned_path(waypoints, reason.get("goal_cfg", goal_cfg), kp=kp,
                                               pos_tol=pos_tol, timeout=timeout)
                if r.reached:
                    return r
            elif reason.get("goal_invalid"):
                self.logger.warning("planned-move goal is in collision at the target pose; "
                                    "refusing to execute it directly")
                self._last_goal_invalid = True
                return MotionResult(reached=False, elapsed=0.0, max_joint_error=float(delta),
                                    final_joints=list(start_cfg), timed_out=False)
        return self.sim.move_joints(goal_cfg, speed=kp, timeout=timeout, pos_tol=pos_tol)

    def ready_tip(self):
        tA = self.sim._toteA_pos
        try:
            lo, hi = self._tray_interior_bounds(int(self.sim.get_ids()["toteA"]))
            y_edge = float(lo[1])
        except Exception:
            y_edge = float(tA[1]) - 0.21
        return (float(tA[0]) + 0.05, y_edge - 0.14, self.geom.safe_z_tip)

    def ready(self, timeout: float = 10.0) -> bool:
        self.logger.info("Moving to ready pose (beside pick tote)")
        r = self._move_tip_validated(list(self.ready_tip()), orn=self._current_ee_orn(),
                                     kp=0.15, timeout=timeout, seed_from_current=True)
        self._settle(150)
        return bool(r.reached)

    def _candidate_yaws_for(self, target_id: int, sx: float, sy: float) -> List[float]:
        primary_yaw = 0.0 if sx >= sy else math.pi / 2
        try:
            from causal_core.cube_pose import grasp_yaws
            yaws = list(grasp_yaws(target_id))
        except Exception:
            yaws = []
        return yaws or [primary_yaw, primary_yaw + math.pi / 2]

    _PRESHAPE_SIDE_MARGIN = 0.007
    neighbour_nudge_allow: float = 0.0

    @staticmethod
    def grasp_tip_for(obj_info: dict, grasp_height: float) -> Tuple[float, float, float]:
        cx, cy = float(obj_info['center'][0]), float(obj_info['center'][1])
        top_z = float(obj_info['aabb_max'][2])
        centre_z = float(obj_info['center'][2])
        return (cx, cy, max(centre_z + 0.006, top_z - float(grasp_height) - 0.015))

    @staticmethod
    def descent_tips_for(obj_info: dict, grasp_height: float):
        cx, cy = float(obj_info['center'][0]), float(obj_info['center'][1])
        top_z = float(obj_info['aabb_max'][2])
        return [(cx, cy, top_z + 0.04), (cx, cy, top_z + 0.015),
                ProductionRobotController.grasp_tip_for(obj_info, grasp_height)]

    def preshape_gap_for(self, target_id: int, yaw: float) -> float:
        from causal_core.cube_pose import extent_along
        return float(extent_along(int(target_id), float(yaw)) + 2.0 * self._PRESHAPE_SIDE_MARGIN)

    def _preshape_gripper_for(self, target_id: int, yaw: float) -> float:
        if not getattr(self, "preshape_enabled", True):
            self.sim.open_gripper()
            return 0.0808
        gap = self.preshape_gap_for(target_id, yaw)
        angle = self.sim.set_gripper_opening(gap)
        self.logger.info(f"Gripper pre-shaped to inner gap {gap*1000:.1f}mm "
                         f"(driver angle {angle if angle is None else round(angle, 3)})")
        return gap

    _C6_MARGIN = 0.004

    def _grasp_descent_clear(self, target_id: int, tip_points, orn, margin: Optional[float] = None,
                             preshape_gap: Optional[float] = None):
        margin = self._C6_MARGIN if margin is None else float(margin)
        attached = self._ompl_attached_bodies(None, None)
        obstacle_ids = self._ompl_nearby_obstacle_ids([self.sim._plane_id, int(target_id)], None)
        mount_id = getattr(self.sim, "_mount_body_id", None)
        saved = self._current_joint_state()
        saved_poses = [(bid, p.getBasePositionAndOrientation(bid)) for bid, _ in attached]
        saved_gripper = self.sim.gripper_joint_positions()
        nudge = float(getattr(self, "neighbour_nudge_allow", 0.0) or 0.0)
        gripper_id = self.sim._gripper_body_id
        finger_links = {3, 4, 8, 9}
        cube_ids = set(int(b) for b in (getattr(self.sim, "_objects_body_ids", []) or []))
        try:
            if preshape_gap is not None:
                self.sim.set_gripper_joints_kinematic(float(preshape_gap))
            cur = list(saved)
            for tip in tip_points:
                ok, _, data = self._solve_and_validate_ik(list(tip), orn, extra_seed=cur)
                if not ok or "joint_solution" not in data:
                    return False, {"reason": "ik_unreachable", "tip": tuple(tip)}
                cfg = self._wrap_joint_targets(data["joint_solution"])
                self._reset_joint_state(cfg)
                self._ompl_place_attached(attached)
                for key, dist, _req in self._ompl_close_pairs(attached, obstacle_ids, margin):
                    a_id, a_link, b_id, _b_link = key
                    if b_id == self.robot_id or a_id == b_id or b_id == mount_id:
                        continue
                    allowed = margin
                    if (nudge > 0.0 and a_id == gripper_id and a_link in finger_links
                            and b_id in cube_ids):
                        allowed = margin - nudge
                    if dist < allowed:
                        return False, {"reason": "descent_collision", "body": int(b_id),
                                       "distance": float(dist), "tip": tuple(tip),
                                       "link": int(a_link)}
                cur = cfg
            return True, {}
        finally:
            self._reset_joint_state(saved)
            for bid, (pos, o) in saved_poses:
                p.resetBasePositionAndOrientation(bid, pos, o)
            if preshape_gap is not None:
                self.sim.restore_gripper_joints(saved_gripper)

    def descent_clear_for_any_yaw(self, target_id: int, grasp_height: float = 0.02):
        info = self.get_object_info(target_id)
        if info is None:
            return False, {"reason": "no_object"}
        sx, sy = info['size'][0], info['size'][1]
        tips = self.descent_tips_for(info, grasp_height)
        last = {}
        for yaw in self._candidate_yaws_for(target_id, sx, sy):
            orn = p.getQuaternionFromEuler([math.pi, 0, yaw])
            gap = self.preshape_gap_for(target_id, yaw)
            ok, last = self._grasp_descent_clear(target_id, tips, orn, preshape_gap=gap)
            if ok:
                return True, {"yaw": yaw, "preshape_gap": gap}
        return False, last

    def _recover_to_ready(self) -> None:
        try:
            cur = self._get_tip_pos()
            if cur[2] < self.geom.safe_z_tip - 0.03:
                self._move_tip_validated([cur[0], cur[1], self.geom.safe_z_tip],
                                         orn=self._current_ee_orn(), kp=0.1, timeout=4.0,
                                         seed_from_current=True)
            self.ready(timeout=8.0)
        except Exception as e:
            self.logger.warning(f"recover_to_ready failed ({e}); falling back to home()")
            self.home()

    def home(self, timeout: float = 10.0):
        self.logger.info("Moving to home")
        home_cfg = list(self.sim.robot_home_joint_config)
        saved = self._current_joint_state()
        self._reset_joint_state(home_cfg)
        home_tip = self._get_tip_pos()
        self._reset_joint_state(saved)
        self._move_to_cfg_planned_if_far(home_cfg, home_tip, self._down_orn,
                                         kp=0.2, timeout=timeout)
        self._settle(300)
        return True

    def _current_joint_state(self) -> List[float]:
        return [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]

    def _current_ee_orn(self):
        ls = p.getLinkState(self.robot_id, self.end_effector_index, computeForwardKinematics=1)
        return tuple(ls[1])

    def _rotate_base_only(self, target_base_yaw: float, timeout: float, speed: float) -> bool:
        current = self._current_joint_state()
        current[0] = float(target_base_yaw)
        r = self.sim.move_joints(current, speed=speed, timeout=timeout)
        self._settle(150)
        return bool(r.reached)

    def _rotate_tip_toward(self, target_xy, timeout: float, speed: float) -> bool:
        bx, by = p.getBasePositionAndOrientation(self.robot_id)[0][:2]
        tip = self._get_tip_pos()
        a_tip = math.atan2(tip[1] - by, tip[0] - bx)
        a_goal = math.atan2(target_xy[1] - by, target_xy[0] - bx)
        da = (a_goal - a_tip + math.pi) % (2 * math.pi) - math.pi
        cur = self._current_joint_state()
        pan = cur[0]
        lo, hi = self._joint_lower[0], self._joint_upper[0]
        target = pan + da
        if not (lo <= target <= hi):
            da = da - math.copysign(2 * math.pi, da)
            target = pan + da
            if not (lo <= target <= hi):
                return False
        goal_cfg = list(cur)
        goal_cfg[0] = float(target)
        tip = self._get_tip_pos()
        r = math.hypot(tip[0] - bx, tip[1] - by)
        goal_tip = (bx + r * math.cos(a_tip + da), by + r * math.sin(a_tip + da), tip[2])
        res = self._move_to_cfg_planned_if_far(goal_cfg, goal_tip, self._current_ee_orn(),
                                               kp=speed, timeout=timeout)
        self._settle(150)
        return bool(res.reached)

    def rotate_to_place_side(self, timeout=5.0) -> bool:
        self.logger.info("Rotating base to place-side (toward ToteB)")
        try:
            tB = self.sim._toteB_pos
            return self._rotate_tip_toward((float(tB[0]), float(tB[1])),
                                           timeout=timeout, speed=0.28)
        except Exception as e:
            self.logger.error(f"rotate_to_place_side failed: {e}")
            return False

    def rotate_to_pick_side(self, timeout=5.0) -> bool:
        self.logger.info("Rotating base to pick-side (toward ToteA)")
        try:
            tA = self.sim._toteA_pos
            return self._rotate_tip_toward((float(tA[0]), float(tA[1])),
                                           timeout=timeout, speed=0.4)
        except Exception as e:
            self.logger.error(f"rotate_to_pick_side failed: {e}")
            return False

    def _settle(self, steps=500):
        self.sim.step_simulation(int(steps))

    def _get_tip_pos(self):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = ee_state[0]
        return (ee_pos[0], ee_pos[1], ee_pos[2] - self.geom.tool_tip_offset_z)

    def _move_safe(self, tip_target, orn=None, kp=0.1, timeout=5.0) -> bool:
        if orn is None:
            orn = self._down_orn
        cur_tip = self._get_tip_pos()
        tx, ty, tz = tip_target
        ok = True

        if cur_tip[2] < self.geom.safe_z_tip - 0.03:
            r = self._move_tip_validated([cur_tip[0], cur_tip[1], self.geom.safe_z_tip], orn, kp=kp, timeout=timeout)
            ok = ok and r.reached

        if tz < self.geom.safe_z_tip - 0.03:
            r = self._move_tip_validated([tx, ty, self.geom.safe_z_tip], orn, kp=kp, timeout=timeout,
                                         plan_if_far=True)
            ok = ok and r.reached
            if not ok:
                return False
            r = self._move_tip_validated([tx, ty, tz], orn, kp=max(kp * 0.7, 0.06), timeout=timeout,
                                         clearance=0.01)
            ok = ok and r.reached
        else:
            r = self._move_tip_validated([tx, ty, tz], orn, kp=kp, timeout=timeout, plan_if_far=True,
                                         clearance=0.01)
            ok = ok and r.reached
        return ok

    def _ik_seed_configs(self) -> List[List[float]]:
        home = list(self._rest_pose)
        flipped_elbow = list(home)
        flipped_elbow[2] = -home[2]
        shoulder_rot = list(home)
        shoulder_rot[0] = home[0] + math.pi
        return [home, flipped_elbow, shoulder_rot]

    def _solve_and_validate_ik(self, tip_pos, orn, pos_tol=0.03, extra_seed=None, all_solutions=None) -> Tuple[bool, Optional[FailureCode], Dict[str, Any]]:
        ee_target = self.geom.tip_to_ee(tip_pos)
        saved = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        limit_tol = 0.08
        attempts = []
        best_in_limit = None

        seeds = ([list(extra_seed)] if extra_seed is not None else []) + self._ik_seed_configs()
        try:
            for seed in seeds:
                for j, v in zip(self.joint_indices, seed):
                    p.resetJointState(self.robot_id, j, v)
                try:
                    sol = p.calculateInverseKinematics(
                        self.robot_id, self.end_effector_index, list(ee_target), orn,
                        maxNumIterations=200, residualThreshold=1e-4)
                except Exception as e:
                    attempts.append({"seed": seed, "error": str(e)})
                    continue
                sol = list(sol)[:len(self.joint_indices)]
                if not all(np.isfinite(v) for v in sol):
                    attempts.append({"seed": seed, "reason": "non-finite"})
                    continue
                in_limit = all(lo - limit_tol <= v <= hi + limit_tol
                              for v, lo, hi in zip(sol, self._joint_lower, self._joint_upper))
                for j, v in zip(self.joint_indices, sol):
                    p.resetJointState(self.robot_id, j, v)
                ls = p.getLinkState(self.robot_id, self.end_effector_index, computeForwardKinematics=True)
                achieved_pos, achieved_orn = ls[4], ls[5]
                pos_residual = math.sqrt(sum((achieved_pos[i] - ee_target[i]) ** 2 for i in range(3)))
                try:
                    diff = p.getDifferenceQuaternion(achieved_orn, orn)
                    orn_residual = 2.0 * math.acos(min(1.0, abs(float(diff[3]))))
                except Exception:
                    orn_residual = float("nan")
                data = {"joint_solution": sol, "pos_residual": pos_residual,
                        "orn_residual": orn_residual, "in_limit": in_limit}
                attempts.append(data)
                if in_limit:
                    if best_in_limit is None or pos_residual < best_in_limit[0]:
                        best_in_limit = (pos_residual, data)
                    if pos_residual <= pos_tol:
                        if all_solutions is None:
                            return True, None, data
                        all_solutions.append(data)
        finally:
            for j, v in zip(self.joint_indices, saved):
                p.resetJointState(self.robot_id, j, v)

        if all_solutions:
            return True, None, all_solutions[0]
        if best_in_limit is not None:
            return False, FailureCode.FK_RESIDUAL_TOO_HIGH, best_in_limit[1]
        return False, FailureCode.IK_FAILED, {"attempts": attempts}

    def _wrap_joint_targets(self, target: List[float]) -> List[float]:
        current = self._current_joint_state()
        wrapped = list(target)
        for i, (cur, tgt, lo, hi) in enumerate(zip(current, target, self._joint_lower, self._joint_upper)):
            if (hi - lo) < (2 * math.pi - 1e-3):
                continue
            diff = tgt - cur
            wrapped_diff = diff - 2 * math.pi * round(diff / (2 * math.pi))
            candidate = cur + wrapped_diff
            if lo - 0.08 <= candidate <= hi + 0.08:
                wrapped[i] = candidate
        return wrapped

    def _get_held_object_id(self) -> Optional[int]:
        if self._grasp_constraint_id is None:
            return None
        try:
            return int(p.getConstraintInfo(self._grasp_constraint_id)[2])
        except Exception:
            return None

    def _held_relative_transform(self, held_id: int):
        try:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index,
                                      computeForwardKinematics=1)
            obj_pos, obj_orn = p.getBasePositionAndOrientation(held_id)
            inv_pos, inv_orn = p.invertTransform(ee_state[0], ee_state[1])
            return p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
        except Exception:
            return None

    def _reset_joint_state(self, cfg) -> None:
        for j, v in zip(self.joint_indices, cfg):
            p.resetJointState(self.robot_id, j, v)

    def _ompl_wrist_z(self, cfg) -> float:
        saved = self._current_joint_state()
        try:
            self._reset_joint_state(cfg)
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index,
                                      computeForwardKinematics=1)
            return float(ee_state[0][2])
        finally:
            self._reset_joint_state(saved)

    _OMPL_SAFETY_MARGIN = 0.03

    def _ompl_attached_bodies(self, held_id, held_rel):
        attached = []
        gid = getattr(self.sim, "_gripper_body_id", None)
        if gid is not None:
            rel = self._held_relative_transform(int(gid))
            if rel is not None:
                attached.append((int(gid), rel))
        if held_id is not None and held_rel is not None:
            attached.append((int(held_id), held_rel))
        return attached

    def _ompl_nearby_obstacle_ids(self, allowed_ids, held_id):
        allowed = set(int(a) for a in allowed_ids if a is not None)
        allowed.add(self.robot_id)
        if held_id is not None:
            allowed.add(int(held_id))
        gid = getattr(self.sim, "_gripper_body_id", None)
        if gid is not None:
            allowed.add(int(gid))
        obstacle_ids = []
        for i in range(p.getNumBodies()):
            bid = p.getBodyUniqueId(i)
            if bid in allowed:
                continue
            obstacle_ids.append(bid)
        return obstacle_ids

    def _ompl_place_attached(self, attached):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index,
                                  computeForwardKinematics=1)
        for bid, rel in attached:
            pos, orn = p.multiplyTransforms(ee_state[0], ee_state[1], rel[0], rel[1])
            p.resetBasePositionAndOrientation(bid, pos, orn)
        return ee_state

    _OMPL_SELF_MARGIN = -0.002
    _OMPL_ATTACHED_SELF_MARGIN = 0.01
    _OMPL_BOX_MARGIN = 0.08
    _OMPL_SELF_QUERY = 0.015

    def _ompl_close_pairs(self, attached, obstacle_ids, margin, nearby=None):
        obs = obstacle_ids if nearby is None else nearby
        moving = [self.robot_id] + [bid for bid, _ in attached]
        mount_id = getattr(self.sim, "_mount_body_id", None)
        boxes = set(int(b) for b in (getattr(self.sim, "obstacles", []) or []))
        box_margin = max(margin, self._OMPL_BOX_MARGIN)
        for a_id in moving:
            for b in obs:
                if b == mount_id:
                    req = self._OMPL_SELF_MARGIN
                elif b in boxes:
                    req = box_margin
                else:
                    req = margin
                for cp in p.getClosestPoints(bodyA=a_id, bodyB=b, distance=req if req > margin else margin):
                    yield (int(cp[1]), int(cp[3]), int(cp[2]), int(cp[4])), float(cp[8]), req
        sm, sq = self._OMPL_SELF_MARGIN, self._OMPL_SELF_QUERY
        am = self._OMPL_ATTACHED_SELF_MARGIN
        for bid, _ in attached:
            for cp in p.getClosestPoints(bodyA=bid, bodyB=self.robot_id, distance=sq):
                yield (int(cp[1]), int(cp[3]), int(cp[2]), int(cp[4])), float(cp[8]), am
        for cp in p.getClosestPoints(bodyA=self.robot_id, bodyB=self.robot_id, distance=sq):
            yield (int(cp[1]), int(cp[3]), int(cp[2]), int(cp[4])), float(cp[8]), sm

    def _ompl_baseline_pairs(self, reference_cfg, attached, obstacle_ids, margin):
        saved = self._current_joint_state()
        saved_poses = [(bid, p.getBasePositionAndOrientation(bid)) for bid, _ in attached]
        try:
            self._reset_joint_state(reference_cfg)
            self._ompl_place_attached(attached)
            return set(key for key, _, _ in self._ompl_close_pairs(attached, obstacle_ids, margin))
        finally:
            self._reset_joint_state(saved)
            for bid, (pos, orn) in saved_poses:
                p.resetBasePositionAndOrientation(bid, pos, orn)

    def _ompl_state_validity_checker(self, allowed_ids, baseline_pairs,
                                     wrist_z_ceiling, attached,
                                     obstacle_ids, obstacle_aabbs, margin):
        allowed = set(int(a) for a in allowed_ids if a is not None)
        n_joints = len(self.joint_indices)
        n_links = p.getNumJoints(self.robot_id)
        attached_ids = [bid for bid, _ in attached]

        def _padded_moving_aabb():
            lo, hi = None, None
            bodies = [(self.robot_id, link) for link in range(-1, n_links)]
            for bid in attached_ids:
                bodies += [(bid, link) for link in range(-1, p.getNumJoints(bid))]
            for bid, link in bodies:
                alo, ahi = p.getAABB(bid, link)
                if lo is None:
                    lo, hi = list(alo), list(ahi)
                else:
                    lo = [min(lo[k], alo[k]) for k in range(3)]
                    hi = [max(hi[k], ahi[k]) for k in range(3)]
            pad = max(margin, self._OMPL_BOX_MARGIN)
            return ([lo[k] - pad for k in range(3)],
                    [hi[k] + pad for k in range(3)])

        def is_valid(state) -> bool:
            cfg = [state[i] for i in range(n_joints)]
            for v, lo, hi in zip(cfg, self._joint_lower, self._joint_upper):
                if not (lo - 1e-3 <= v <= hi + 1e-3):
                    return False
            saved = self._current_joint_state()
            saved_poses = [(bid, p.getBasePositionAndOrientation(bid)) for bid in attached_ids]
            try:
                self._reset_joint_state(cfg)
                ee_state = self._ompl_place_attached(attached)
                if ee_state[0][2] > wrist_z_ceiling:
                    return False
                q_lo, q_hi = _padded_moving_aabb()
                nearby = [
                    b for b in obstacle_ids if b not in allowed and (
                        (a := obstacle_aabbs[b])[0][0] <= q_hi[0] and a[1][0] >= q_lo[0] and
                        a[0][1] <= q_hi[1] and a[1][1] >= q_lo[1] and
                        a[0][2] <= q_hi[2] and a[1][2] >= q_lo[2]
                    )
                ]
                for key, dist, required in self._ompl_close_pairs(attached, obstacle_ids, margin, nearby=nearby):
                    if key in baseline_pairs:
                        continue
                    if dist < required:
                        return False
                return True
            finally:
                self._reset_joint_state(saved)
                for bid, (pos, orn) in saved_poses:
                    p.resetBasePositionAndOrientation(bid, pos, orn)
        return is_valid

    class _FakeState:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, i):
            return self._values[i]

    def _guide_samplers(self, start_tip, goal_tip):
        try:
            bx, by = p.getBasePositionAndOrientation(self.robot_id)[0][:2]
        except Exception:
            bx, by = 0.0, 0.0
        sx, sy = start_tip[0] - bx, start_tip[1] - by
        gx, gy = goal_tip[0] - bx, goal_tip[1] - by
        dx, dy = gx - sx, gy - sy
        seg2 = dx * dx + dy * dy
        u = 0.0 if seg2 < 1e-9 else max(0.0, min(1.0, -(sx * dx + sy * dy) / seg2))
        cx, cy = sx + u * dx, sy + u * dy
        samplers = []
        if math.hypot(cx, cy) > 0.25:
            samplers.append(lambda t: [start_tip[k] + t * (goal_tip[k] - start_tip[k]) for k in range(3)])
        r0, r1 = math.hypot(sx, sy), math.hypot(gx, gy)
        a0, a1 = math.atan2(sy, sx), math.atan2(gy, gx)
        da_short = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        da_long = da_short - math.copysign(2 * math.pi, da_short) if abs(da_short) > 1e-6 else 2 * math.pi

        z0, z1 = start_tip[2], goal_tip[2]
        split = 0.7

        def _arc(da, sweep_first):
            def pos(t):
                if sweep_first:
                    s = min(1.0, t / split)
                    r, a = r0 + s * (r1 - r0), a0 + s * da
                    z = z0 if t <= split else z0 + ((t - split) / (1 - split)) * (z1 - z0)
                else:
                    s = max(0.0, (t - (1 - split)) / split)
                    r, a = r0 + s * (r1 - r0), a0 + s * da
                    z = z1 if t >= (1 - split) else z0 + (t / (1 - split)) * (z1 - z0)
                return [bx + r * math.cos(a), by + r * math.sin(a), z]
            return pos
        for da in (da_short, da_long):
            samplers.append(_arc(da, sweep_first=(z1 <= z0)))
            samplers.append(_arc(da, sweep_first=(z1 > z0)))
        return samplers

    def _try_cartesian_guide_path(self, start_cfg, goal_cfg, start_tip, goal_tip,
                                  orn, checker, n_samples=16):
        for sample_pos in self._guide_samplers(start_tip, goal_tip):
            path = self._guide_path_along(sample_pos, start_cfg, goal_cfg, orn, checker, n_samples)
            if path is not None:
                return path
        return None

    def _guide_path_along(self, sample_pos, start_cfg, goal_cfg, orn, checker, n_samples):
        saved = self._current_joint_state()
        try:
            self._reset_joint_state(start_cfg)
            configs = [list(start_cfg)]
            for i in range(1, n_samples + 1):
                t = i / n_samples
                pos = sample_pos(t)
                ok, _, data = self._solve_and_validate_ik(list(pos), orn, extra_seed=configs[-1])
                if not ok or "joint_solution" not in data:
                    return None
                cand = self._wrap_joint_targets(data["joint_solution"])
                if max(abs(a - b) for a, b in zip(cand[:-1], configs[-1][:-1])) > 1.2:
                    return None
                configs.append(cand)
                self._reset_joint_state(cand)
            if max(abs(a - b) for a, b in zip(configs[-1][:-1], goal_cfg[:-1])) > 0.2:
                goal_cfg = list(configs[-1])
            else:
                configs[-1] = list(goal_cfg)
            start_roll, goal_roll = start_cfg[-1], goal_cfg[-1]
            roll_diff = goal_roll - start_roll
            roll_diff -= 2 * math.pi * round(roll_diff / (2 * math.pi))
            n = len(configs) - 1
            for idx, cfg in enumerate(configs):
                if 0 < idx < n:
                    cfg[-1] = start_roll + (idx / n) * roll_diff
            near_roll = start_roll + roll_diff
            if self._joint_lower[-1] - 0.08 <= near_roll <= self._joint_upper[-1] + 0.08:
                configs[-1][-1] = near_roll
            for cfg in configs:
                if not checker(self._FakeState(cfg)):
                    return None
            return configs
        finally:
            self._reset_joint_state(saved)

    def _plan_joint_path(self, goal_cfg, held_id=None, held_rel=None,
                         allowed_ids=(), planning_time=2.0,
                         start_tip=None, goal_tip=None, orn=None,
                         reason=None, clearance=None) -> Optional[List[List[float]]]:
        if not _OMPL_AVAILABLE:
            return None
        try:
            n_joints = len(self.joint_indices)
            start_cfg = self._current_joint_state()
            space = _ompl_base.RealVectorStateSpace(n_joints)
            bounds = _ompl_base.RealVectorBounds(n_joints)
            for i in range(n_joints):
                bounds.setLow(i, float(self._joint_lower[i]))
                bounds.setHigh(i, float(self._joint_upper[i]))
            space.setBounds(bounds)
            ss = _ompl_geometric.SimpleSetup(space)

            clearance_margin = self._OMPL_SAFETY_MARGIN if clearance is None else float(clearance)
            attached = self._ompl_attached_bodies(held_id, held_rel)
            obstacle_ids = self._ompl_nearby_obstacle_ids(allowed_ids, held_id)
            obstacle_aabbs = {b: p.getAABB(b) for b in obstacle_ids}
            baseline_pairs = self._ompl_baseline_pairs(start_cfg, attached,
                                                       obstacle_ids, clearance_margin)
            height_margin = 0.08
            ceiling = max(self._ompl_wrist_z(start_cfg), self._ompl_wrist_z(goal_cfg)) + height_margin
            checker = self._ompl_state_validity_checker(allowed_ids, baseline_pairs,
                                                        ceiling, attached,
                                                        obstacle_ids, obstacle_aabbs,
                                                        clearance_margin)
            ss.setStateValidityChecker(checker)

            si = ss.getSpaceInformation()
            start_state = si.allocState()
            goal_state = si.allocState()
            for i in range(n_joints):
                start_state[i] = float(start_cfg[i])
                goal_state[i] = float(goal_cfg[i])
            if not checker(start_state) or not checker(goal_state):
                if reason is not None:
                    reason["start_invalid"] = not checker(start_state)
                    reason["goal_invalid"] = not checker(goal_state)
                return None

            if start_tip is not None and goal_tip is not None and orn is not None:
                guide = self._try_cartesian_guide_path(start_cfg, goal_cfg, start_tip,
                                                       goal_tip, orn, checker)
                if guide is not None:
                    self._reset_joint_state(start_cfg)
                    if reason is not None:
                        reason["goal_cfg"] = list(guide[-1])
                    return guide

            ss.setStartAndGoalStates(start_state, goal_state)
            ss.setPlanner(_ompl_geometric.RRTConnect(si))

            budgets = [int(1000 * float(planning_time)), int(2000 * float(planning_time))]
            solved = False
            for i, n_iter in enumerate(budgets):
                counter = {"n": 0}

                def _ptc_fn(counter=counter, limit=n_iter):
                    counter["n"] += 1
                    return counter["n"] >= limit
                solved = ss.solve(_ompl_base.PlannerTerminationCondition(_ptc_fn))
                if solved and ss.haveExactSolutionPath():
                    break
                if i < len(budgets) - 1:
                    ss.getProblemDefinition().clearSolutionPaths()
            if not solved or not ss.haveExactSolutionPath():
                return None
            path = ss.getSolutionPath()
            _ompl_geometric.PathSimplifier(si).simplifyMax(path)
            path.interpolate(60)
            waypoints = [[path.getState(i)[j] for j in range(n_joints)]
                        for i in range(path.getStateCount())]
            if any(abs(a - b) > 1e-2 for a, b in zip(waypoints[-1], goal_cfg)):
                self.logger.warning("OMPL path endpoint drifted from goal_cfg after "
                                    "simplify/interpolate; discarding planned path")
                self._reset_joint_state(start_cfg)
                return None
            self._reset_joint_state(start_cfg)
            return waypoints
        except Exception as e:
            self.logger.warning(f"OMPL planning raised, falling back to direct move: {e}")
            return None

    def _execute_planned_path(self, waypoints, goal_cfg, kp, pos_tol, timeout) -> MotionResult:
        intermediate_kp = max(kp, 0.25)
        for wp in waypoints[1:]:
            self.sim.move_joints(wp, speed=intermediate_kp, timeout=1.0, pos_tol=0.03)
        return self.sim.move_joints(goal_cfg, speed=max(kp, 0.2),
                                    timeout=max(float(timeout), 2.0), pos_tol=pos_tol)

    def _move_tip_validated(self, tip_pos, orn=None, kp=0.1, timeout=5.0, pos_tol=0.01,
                            use_planning=False, seed_from_current=False, plan_if_far=True,
                            clearance=None):
        if orn is None:
            orn = self._down_orn
        extra_seed = self._current_joint_state() if (use_planning or seed_from_current) else None
        ok, code, data = self._solve_and_validate_ik(list(tip_pos), orn, extra_seed=extra_seed)
        if not ok or "joint_solution" not in data:
            return MotionResult(reached=False, elapsed=0.0, max_joint_error=float("inf"),
                                final_joints=[], timed_out=False)
        goal_cfg = self._wrap_joint_targets(data["joint_solution"])

        if use_planning:
            held_id = self._get_held_object_id()
            held_rel = self._held_relative_transform(held_id) if held_id is not None else None
            planning_time = min(6.0, max(1.5, float(timeout) * 0.4))
            reason = {}
            waypoints = self._plan_joint_path(
                goal_cfg, held_id=held_id, held_rel=held_rel,
                allowed_ids=[self.sim._plane_id], planning_time=planning_time,
                start_tip=self._get_tip_pos(), goal_tip=tuple(tip_pos), orn=orn,
                reason=reason)

            if waypoints:
                result = self._execute_planned_path(waypoints, reason.get("goal_cfg", goal_cfg), kp=kp,
                                                    pos_tol=pos_tol, timeout=timeout)
                if result.reached:
                    return result

        if plan_if_far:
            cur = self._current_joint_state()
            thr = self._PLAN_IF_JOINT_DELTA_ABOVE

            def _delta(cfg):
                return max(abs(a - b) for a, b in zip(cfg, cur))

            candidates = [goal_cfg]
            goal_self_collides = self._segment_self_collides(cur, goal_cfg)
            if _delta(goal_cfg) > thr or goal_self_collides:
                sols = []
                self._solve_and_validate_ik(list(tip_pos), orn, extra_seed=cur, all_solutions=sols)
                branches = []
                for data_n in sols:
                    cfg_n = self._wrap_joint_targets(data_n["joint_solution"])
                    if all(max(abs(a - b) for a, b in zip(cfg_n, other)) > 0.05 for other in branches):
                        branches.append(cfg_n)
                branches.sort(key=_delta)
                if branches:
                    candidates = branches
            if goal_self_collides or len(candidates) > 1:
                clean = [cfg for cfg in candidates
                         if not (self._segment_self_collides(cur, cfg) if _delta(cfg) <= thr
                                 else self._goal_self_collides(cfg))]
                if clean:
                    candidates = clean
                elif goal_self_collides:
                    return MotionResult(reached=False, elapsed=0.0, max_joint_error=float("inf"),
                                        final_joints=list(cur), timed_out=False)
            last = None
            for cfg in candidates:
                self._last_goal_invalid = False
                r = self._move_to_cfg_planned_if_far(cfg, tuple(tip_pos), orn,
                                                     kp=kp, timeout=timeout, pos_tol=pos_tol,
                                                     clearance=clearance)
                if r.reached or not self._last_goal_invalid:
                    return r
                last = r
            return last if last is not None else MotionResult(
                reached=False, elapsed=0.0, max_joint_error=float("inf"),
                final_joints=list(cur), timed_out=False)
        return self.sim.move_joints(goal_cfg, speed=kp, timeout=timeout, pos_tol=pos_tol)

    def _goal_self_collides(self, cfg) -> bool:
        try:
            held_id = self._get_held_object_id()
            held_rel = self._held_relative_transform(held_id) if held_id is not None else None
            attached = self._ompl_attached_bodies(held_id, held_rel)
            cur = self._current_joint_state()
            baseline = self._ompl_baseline_pairs(cur, attached, [], self._OMPL_SAFETY_MARGIN)
            saved_poses = [(bid, p.getBasePositionAndOrientation(bid)) for bid, _ in attached]
            try:
                self._reset_joint_state(list(cfg))
                self._ompl_place_attached(attached)
                for key, dist, _req in self._ompl_close_pairs(attached, [], self._OMPL_SAFETY_MARGIN):
                    if key in baseline:
                        continue
                    if dist < max(_req, 0.002):
                        return True
                return False
            finally:
                self._reset_joint_state(cur)
                for bid, (pos, o) in saved_poses:
                    p.resetBasePositionAndOrientation(bid, pos, o)
        except Exception:
            return False

    def _segment_self_collides(self, start_cfg, goal_cfg, n: int = 8) -> bool:
        for k in range(1, n + 1):
            t = k / float(n)
            cfg = [a + t * (b - a) for a, b in zip(start_cfg, goal_cfg)]
            if self._goal_self_collides(cfg):
                return True
        return False

    def _unexpected_collision(self, allowed_ids) -> Tuple[bool, Optional[Dict[str, Any]]]:
        try:
            contacts = p.getContactPoints(bodyA=self.robot_id)
        except Exception:
            return False, None
        allowed = set(int(a) for a in allowed_ids if a is not None)
        for c in contacts:
            other = int(c[2])
            if other in allowed:
                continue
            normal_force = float(c[9]) if len(c) > 9 else 0.0
            if normal_force > 1.0:
                return True, {"other_body": other, "normal_force": normal_force,
                             "link_index": int(c[3])}
        return False, None

    def get_object_info(self, object_id):
        try:
            pos, orn = p.getBasePositionAndOrientation(object_id)
            aabb_min, aabb_max = p.getAABB(object_id)
            return {
                'position': pos, 'orientation': orn,
                'aabb_min': aabb_min, 'aabb_max': aabb_max,
                'size': [aabb_max[i] - aabb_min[i] for i in range(3)],
                'center': [(aabb_max[i] + aabb_min[i]) / 2 for i in range(3)],
            }
        except Exception:
            return None

    def is_position_in_workspace(self, tip_pos):
        x, y, z = tip_pos
        return (self.workspace['x'][0] <= x <= self.workspace['x'][1] and
                self.workspace['y'][0] <= y <= self.workspace['y'][1] and
                self.workspace['z'][0] <= z <= self.workspace['z'][1])

    def is_position_in_pick_area(self, tip_pos):
        x, y, z = tip_pos
        return (self.pick_workspace['x'][0] <= x <= self.pick_workspace['x'][1] and
                self.pick_workspace['y'][0] <= y <= self.pick_workspace['y'][1] and
                self.pick_workspace['z'][0] <= z <= self.pick_workspace['z'][1])

    def _ensure_same_side_tote(self):
        if self._same_side_tote_id is not None:
            return
        ws1 = self.sim._workspace1_bounds
        cx = float((ws1[0, 0] + ws1[0, 1]) / 2.0)
        y_extent = float(ws1[1, 1] - ws1[1, 0])
        cy = float(ws1[1, 1]) - 0.18 * y_extent
        import os
        tote_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "assets", "tote", "toteA_tall.urdf")
        self._same_side_tote_id = p.loadURDF(
            tote_urdf, [cx, cy, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self._same_side_tote_center = (cx, cy, 0.01)
        self.logger.info(f"Created same-side destination tray at ({cx:.2f}, {cy:.2f})")

    def _place_target(self, layout: str) -> Tuple[Tuple[float, float, float], Optional[int]]:
        if layout == "same-side":
            self._ensure_same_side_tote()
            return self._same_side_tote_center, self._same_side_tote_id
        return self.place_tote_center, self.place_tote_id

    def _tray_interior_bounds(self, tote_id: Optional[int]) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        if tote_id is None:
            return None
        try:
            amin, amax = p.getAABB(tote_id)
        except Exception:
            return None
        inset = 0.02
        lo = (amin[0] + inset, amin[1] + inset, amin[2])
        hi = (amax[0] - inset, amax[1] - inset, amax[2] + 0.15)
        return lo, hi

    def _get_finger_contacts(self, target_id) -> Tuple[Set[int], Set[int]]:
        gid = self.sim._gripper_body_id
        left, right = set(), set()
        if gid is None:
            return left, right
        try:
            cps = p.getContactPoints(bodyA=gid, bodyB=int(target_id))
        except Exception:
            return left, right
        for cp in cps:
            link_a = int(cp[3])
            side = self.geom.classify_finger_link(link_a)
            if side == 'left':
                left.add(link_a)
            elif side == 'right':
                right.add(link_a)
        return left, right

    def _has_proper_two_finger_grasp(self, target_id) -> Tuple[bool, Dict[str, Any]]:
        consistent_frames = 0
        last_left, last_right = set(), set()
        for _ in range(5):
            self.sim.step_simulation(3)
            left, right = self._get_finger_contacts(target_id)
            if len(left) >= 1 and len(right) >= 1:
                consistent_frames += 1
                last_left, last_right = left, right
        info = {"consistent_frames": consistent_frames,
                "left_links": sorted(last_left), "right_links": sorted(last_right)}
        if consistent_frames < 4:
            self.logger.info(f"Contact check FAIL: only {consistent_frames}/5 frames had bilateral contact")
            info["reason"] = "insufficient_consistent_frames"
            return False, info

        try:
            obj_pos, _ = p.getBasePositionAndOrientation(int(target_id))
            obj_xy = (obj_pos[0], obj_pos[1])
        except Exception:
            info["reason"] = "object_pose_unavailable"
            return False, info

        try:
            right_pad_state = p.getLinkState(self.sim._gripper_body_id, 4)
            left_pad_state = p.getLinkState(self.sim._gripper_body_id, 9)
            right_pad_xy = (right_pad_state[0][0], right_pad_state[0][1])
            left_pad_xy = (left_pad_state[0][0], left_pad_state[0][1])
        except Exception:
            self.logger.warning("Couldn't read pad link states; skipping geometry gate")
            right_pad_xy = left_pad_xy = obj_xy

        mid_xy = ((right_pad_xy[0] + left_pad_xy[0]) / 2, (right_pad_xy[1] + left_pad_xy[1]) / 2)
        center_dist = math.sqrt((obj_xy[0] - mid_xy[0]) ** 2 + (obj_xy[1] - mid_xy[1]) ** 2)
        info["pad_midpoint"] = mid_xy
        info["center_dist_from_pad_midpoint"] = center_dist
        if center_dist > 0.025:
            self.logger.info(f"Contact check FAIL: cube center is {center_dist:.3f}m from pad midpoint")
            info["reason"] = "object_not_between_pads"
            return False, info

        try:
            pad_z = 0.5 * (right_pad_state[0][2] + left_pad_state[0][2])
            pad_half_h = 0.0205
            dz = float(obj_pos[2]) - pad_z
            info["object_dz_from_pad_centre"] = dz
            if abs(dz) > pad_half_h:
                self.logger.info(f"Contact check FAIL: cube centre {dz:+.3f}m from pad centre -- "
                                 f"outside the pads' vertical span, not a pinch")
                info["reason"] = "object_outside_pad_span"
                return False, info
        except Exception:
            pass

        try:
            cps = p.getContactPoints(bodyA=self.sim._gripper_body_id, bodyB=int(target_id))
        except Exception:
            cps = []
        max_normal_force = 0.0
        pad_normals = {"left": [], "right": []}
        for cp in cps:
            if len(cp) >= 10:
                max_normal_force = max(max_normal_force, abs(float(cp[9])))
            side = self.geom.classify_finger_link(int(cp[3]))
            if side in pad_normals and abs(float(cp[9])) > 0.05:
                pad_normals[side].append(tuple(float(v) for v in cp[7]))
        info["max_normal_force"] = max_normal_force
        if max_normal_force < 0.5:
            self.logger.info(f"Contact check FAIL: max contact normal force = {max_normal_force:.2f}N")
            info["reason"] = "insufficient_normal_force"
            return False, info
        horiz = {}
        for side, ns in pad_normals.items():
            horiz[side] = [n for n in ns if abs(n[2]) < 0.6]
        info["horizontal_contacts"] = {k: len(v) for k, v in horiz.items()}
        if not horiz["left"] or not horiz["right"]:
            self.logger.info("Contact check FAIL: finger contacts are not horizontal on both sides "
                             "(fingers resting on the cube, not around it)")
            info["reason"] = "contacts_not_horizontal"
            return False, info
        nl = horiz["left"][0]; nr = horiz["right"][0]
        opposed = nl[0] * nr[0] + nl[1] * nr[1]
        info["normal_opposition"] = opposed
        if opposed > -0.3:
            self.logger.info(f"Contact check FAIL: left/right contact normals not opposed (dot={opposed:+.2f})")
            info["reason"] = "contacts_not_opposed"
            return False, info

        self.logger.info(f"Contact check PASS: left_links={last_left}, right_links={last_right}, "
                         f"center_dist={center_dist:.3f}m, max_force={max_normal_force:.1f}N")
        return True, info

    def execute_grasp_request(self, request: GraspRequest) -> GraspResult:
        start_time = self.sim.now()
        deadline = start_time + float(request.timeout)
        self._stats['total_attempts'] += 1
        self.logger.info(f"=== Pick & Place (state machine): object {request.target_id} ===")

        trace: List[StepResult] = []

        def emit(state, ok, code=None, msg="", **data):
            step = StepResult(state=state, ok=ok, failure_code=code, message=msg, data=data)
            trace.append(step)
            return step

        def time_left(cap: float = 4.0):
            return max(0.1, min(float(cap), deadline - self.sim.now()))

        def finalize(status, failure_code=None, msg="", **extra_debug):
            elapsed = self.sim.now() - start_time
            if status != GraspStatus.SUCCESS:
                self._stats['failed_grasps'] += 1
            result = GraspResult(
                status=status, error_message=msg, execution_time=elapsed,
                failure_code=failure_code, transitions=trace,
                bilateral_contact_success=bilateral_contact_success,
                friction_micro_lift_success=friction_micro_lift_success,
                constraint_assisted_transport_success=constraint_assisted_transport_success,
                debug_info=extra_debug,
            )
            return result

        bilateral_contact_success = False
        friction_micro_lift_success: Optional[bool] = None
        constraint_assisted_transport_success = False

        if self._grasp_constraint_id is not None:
            self.logger.warning(
                f"Clearing stale constraint id={self._grasp_constraint_id} left over "
                f"from a previous request before starting a new one")
            self._release_constraint()

        try:
            obj_info = self.get_object_info(request.target_id)
            if obj_info is None:
                emit(ControllerState.PRECHECK, False, FailureCode.INVALID_TARGET,
                     f"Object {request.target_id} not found")
                return finalize(GraspStatus.INVALID_TARGET, FailureCode.INVALID_TARGET,
                                f"Object {request.target_id} not found")

            cx, cy = obj_info['center'][0], obj_info['center'][1]
            top_z = obj_info['aabb_max'][2]
            tip_grasp = self.grasp_tip_for(obj_info, request.grasp_height)
            tip_approach = (cx, cy, top_z + request.approach_height)

            if not self.is_position_in_pick_area(tip_grasp):
                emit(ControllerState.PRECHECK, False, FailureCode.OUT_OF_WORKSPACE,
                     f"Pick target {tip_grasp} outside pick area")
                return finalize(GraspStatus.OUT_OF_WORKSPACE, FailureCode.OUT_OF_WORKSPACE,
                                f"Pick target {tip_grasp} outside pick area")

            sx, sy = obj_info['size'][0], obj_info['size'][1]
            primary_yaw = 0.0 if sx >= sy else math.pi / 2
            yaws = request.candidate_yaws
            if yaws is None:
                yaws = self._candidate_yaws_for(request.target_id, sx, sy)
            emit(ControllerState.PRECHECK, True, msg="precheck ok",
                 top_z=top_z, size=[sx, sy], candidate_yaws=yaws)

            self.sim.open_gripper()

            contact_ok = False
            contact_info = {}
            chosen_yaw = yaws[0]

            for yaw_idx, yaw in enumerate(yaws):
                if self.sim.now() > deadline:
                    emit(ControllerState.PREGRASP, False, FailureCode.MOTION_TIMEOUT,
                         "request deadline exceeded before trying next yaw")
                    break
                orn = p.getQuaternionFromEuler([math.pi, 0, yaw])

                ik_ok, ik_code, ik_data = self._solve_and_validate_ik(tip_approach, orn)
                if not ik_ok:
                    emit(ControllerState.PREGRASP, False, ik_code,
                         f"IK/FK validation failed for yaw={yaw:.2f}", **ik_data)
                    continue
                emit(ControllerState.PREGRASP, True, msg=f"IK validated for yaw={yaw:.2f}", **ik_data)

                preshape_gap = self._preshape_gripper_for(request.target_id, yaw)
                emit(ControllerState.PREGRASP, True,
                     msg=f"gripper pre-shaped to {preshape_gap*1000:.1f}mm for yaw={yaw:.2f}",
                     preshape_gap=preshape_gap)
                desc_tips = self.descent_tips_for(obj_info, request.grasp_height)
                if getattr(self, "descent_check_enabled", True):
                    clear, clear_info = self._grasp_descent_clear(request.target_id, desc_tips, orn)
                else:
                    clear, clear_info = True, {}
                if not clear:
                    emit(ControllerState.PREGRASP, False, FailureCode.PATH_COLLISION,
                         f"descent for yaw={yaw:.2f} would contact body "
                         f"{clear_info.get('body', '?')} ({clear_info.get('reason')}); skipping",
                         **clear_info)
                    continue

                reached = self._move_safe(tip_approach, orn=orn, kp=0.1, timeout=time_left())
                if not reached:
                    emit(ControllerState.PREGRASP, False, FailureCode.MOTION_TIMEOUT,
                         "approach motion did not reach target")
                    continue
                cur_tip = self._get_tip_pos()
                err_xy = math.sqrt((cur_tip[0] - cx) ** 2 + (cur_tip[1] - cy) ** 2)
                emit(ControllerState.PREGRASP, True, msg="approach reached", xy_error=err_xy)

                collided, coll_info = self._unexpected_collision(allowed_ids=[self.sim._plane_id])
                if collided:
                    emit(ControllerState.PREGRASP, False, FailureCode.PATH_COLLISION,
                         f"unexpected contact with body {coll_info['other_body']} "
                         f"during approach (force={coll_info['normal_force']:.1f}N)",
                         **coll_info)
                    self._move_safe([cx, cy, self.geom.safe_z_tip], orn=orn, timeout=time_left())
                    continue

                r = self._move_tip(tip_grasp, orn=orn, kp=0.08, timeout=time_left(), pos_tol=0.01)
                self._settle(80)
                achieved_tip = self._get_tip_pos()
                descent_err = math.sqrt(sum((achieved_tip[i] - tip_grasp[i]) ** 2 for i in range(3)))
                if descent_err > 0.05:
                    emit(ControllerState.DESCEND, False, FailureCode.MOTION_TIMEOUT,
                         f"descent stalled {descent_err:.3f}m from target depth",
                         motion_reached=r.reached, descent_err=descent_err)
                    self._move_safe([cx, cy, self.geom.safe_z_tip], orn=orn, timeout=time_left())
                    continue
                emit(ControllerState.DESCEND, True,
                     msg=f"descended to grasp depth (err={descent_err:.4f}m, "
                         f"motion_reached={r.reached})",
                     motion_reached=r.reached, descent_err=descent_err)

                self.sim.close_gripper()
                self._settle(150)
                emit(ControllerState.CLOSE, True, msg="gripper closed")

                contact_ok, contact_info = self._has_proper_two_finger_grasp(request.target_id)
                emit(ControllerState.VERIFY_CONTACT, contact_ok,
                     None if contact_ok else FailureCode.NO_BILATERAL_CONTACT,
                     "bilateral contact verified" if contact_ok else "no proper bilateral contact",
                     **contact_info)
                if contact_ok:
                    chosen_yaw = yaw
                    break

                self.sim.open_gripper()
                self._move_safe([cx, cy, self.geom.safe_z_tip], orn=orn, timeout=time_left())

            bilateral_contact_success = bool(contact_ok)
            if not contact_ok:
                self.sim.open_gripper()
                self._recover_to_ready()
                return finalize(GraspStatus.FAILED_GRASP, FailureCode.NO_BILATERAL_CONTACT,
                                "No two-sided finger contact after trying all candidate yaws")

            orn = p.getQuaternionFromEuler([math.pi, 0, chosen_yaw])

            if request.enable_micro_lift:
                z_before = p.getBasePositionAndOrientation(request.target_id)[0][2]
                micro_target = (cx, cy, tip_grasp[2] + 0.02)
                r = self._move_tip(micro_target, orn=orn, kp=0.05, timeout=min(2.0, time_left()))
                self._settle(60)
                z_after = p.getBasePositionAndOrientation(request.target_id)[0][2]
                lift_dz = z_after - z_before
                friction_micro_lift_success = bool(lift_dz > 0.01)
                emit(ControllerState.OPTIONAL_MICRO_LIFT, friction_micro_lift_success,
                     None if friction_micro_lift_success else FailureCode.MICRO_LIFT_FAILED,
                     f"micro-lift dz={lift_dz:.4f}m (friction-only, no constraint yet)",
                     lift_dz=lift_dz)
                if not friction_micro_lift_success:
                    self.sim.open_gripper()
                    self._settle(60)
                    self._move_safe([cx, cy, self.geom.safe_z_tip], orn=orn, timeout=time_left())
                    self._recover_to_ready()
                    return finalize(GraspStatus.FAILED_GRASP, FailureCode.MICRO_LIFT_FAILED,
                                    f"Cube did not follow a friction-only 2cm lift (dz={lift_dz:.4f}m); "
                                    f"not a real grasp, constraint not created")
                self._move_tip(tip_grasp, orn=orn, kp=0.05, timeout=min(2.0, time_left()))
                self._settle(40)

            try:
                ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
                ee_pos, ee_orn = ee_state[0], ee_state[1]
                obj_pos, obj_orn = p.getBasePositionAndOrientation(request.target_id)
                ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos, ee_orn)
                child_pos, child_orn = p.multiplyTransforms(ee_inv_pos, ee_inv_orn, obj_pos, obj_orn)
                self._grasp_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.robot_id, parentLinkIndex=self.end_effector_index,
                    childBodyUniqueId=request.target_id, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                    parentFramePosition=child_pos, childFramePosition=[0, 0, 0],
                    parentFrameOrientation=child_orn, childFrameOrientation=[0, 0, 0, 1],
                )
                self._is_grasping = True
                self.sim.hold_gripper_gently()
                emit(ControllerState.ATTACH, True, msg=f"constraint id={self._grasp_constraint_id}")
            except Exception as e:
                emit(ControllerState.ATTACH, False, FailureCode.ATTACH_FAILED, str(e))
                self.sim.open_gripper()
                self._recover_to_ready()
                return finalize(GraspStatus.FAILED_GRASP, FailureCode.ATTACH_FAILED, str(e))

            z_before_lift = p.getBasePositionAndOrientation(request.target_id)[0][2]
            r = self._move_tip([cx, cy, self.geom.safe_z_tip], orn=orn, kp=0.08, timeout=time_left(6.0))
            self._settle(200)
            z_after_lift = p.getBasePositionAndOrientation(request.target_id)[0][2]
            actual_lift = z_after_lift - z_before_lift
            lift_ok = actual_lift >= 0.05
            emit(ControllerState.LIFT, lift_ok,
                 None if lift_ok else FailureCode.LIFT_FAILED,
                 f"lifted {actual_lift:.3f}m", lifted_m=actual_lift, motion_reached=r.reached)
            if not lift_ok:
                self._release_constraint()
                self.sim.open_gripper()
                self._recover_to_ready()
                return finalize(GraspStatus.FAILED_GRASP, FailureCode.LIFT_FAILED,
                                f"Object only lifted {actual_lift:.3f}m")

            place_center, place_id = self._place_target(request.place_layout)
            place_x, place_y, place_z_floor = place_center
            tip_above_place = (place_x, place_y, self.geom.safe_z_tip)
            tip_drop = (place_x, place_y, place_z_floor + 0.18)
            if request.place_layout == "opposite":
                rotated = self.rotate_to_place_side(timeout=time_left(10.0))
                if not rotated:
                    emit(ControllerState.TRANSIT, False, FailureCode.MOTION_TIMEOUT,
                         "base rotation to place-side failed")
                if p.getBasePositionAndOrientation(request.target_id)[0][2] < z_after_lift - 0.08:
                    emit(ControllerState.TRANSIT, False, FailureCode.OBJECT_DROPPED,
                         "object height dropped unexpectedly during transit")
                    self._release_constraint()
                    self.sim.open_gripper()
                    self._recover_to_ready()
                    return finalize(GraspStatus.FAILED_PLACE, FailureCode.OBJECT_DROPPED,
                                    "Object dropped during transit")
            place_orn = self._current_ee_orn()

            r = self._move_tip_validated(tip_above_place, orn=place_orn, kp=0.15,
                                         timeout=time_left(20.0), use_planning=True)
            self._settle(120)
            emit(ControllerState.TRANSIT, r.reached,
                 None if r.reached else FailureCode.MOTION_TIMEOUT,
                 "moved above destination tray", place_layout=request.place_layout)

            collided, coll_info = self._unexpected_collision(
                allowed_ids=[self.sim._plane_id, request.target_id, place_id])
            if collided:
                emit(ControllerState.TRANSIT, False, FailureCode.PATH_COLLISION,
                     f"unexpected contact with body {coll_info['other_body']} "
                     f"while transiting to the destination tray "
                     f"(force={coll_info['normal_force']:.1f}N)", **coll_info)
                self._release_constraint()
                self.sim.open_gripper()
                self._recover_to_ready()
                return finalize(GraspStatus.FAILED_PLACE, FailureCode.PATH_COLLISION,
                                "Unexpected collision during transit")

            r = self._move_tip_validated(tip_drop, orn=place_orn, kp=0.1, timeout=time_left(6.0),
                                         seed_from_current=True)
            self._settle(100)
            emit(ControllerState.LOWER, r.reached,
                 None if r.reached else FailureCode.MOTION_TIMEOUT,
                 "lowered to release height")

            self.logger.info("[PLACE] Releasing object")
            self._release_constraint()
            self._settle(20)
            emit(ControllerState.DETACH, True, msg="constraint released")

            self.sim.open_gripper()
            self._settle(180)
            emit(ControllerState.OPEN, True, msg="gripper opened")

            after_release_pos = p.getBasePositionAndOrientation(request.target_id)[0]
            dist_to_tote = math.sqrt((after_release_pos[0] - place_x) ** 2 +
                                     (after_release_pos[1] - place_y) ** 2)
            bounds = self._tray_interior_bounds(place_id)
            if bounds is not None:
                lo, hi = bounds
                placed_ok = (lo[0] <= after_release_pos[0] <= hi[0] and
                            lo[1] <= after_release_pos[1] <= hi[1] and
                            lo[2] <= after_release_pos[2] <= hi[2])
            else:
                placed_ok = dist_to_tote < 0.2 and after_release_pos[2] < 0.30
            constraint_clear = self._grasp_constraint_id is None
            placed_ok = placed_ok and constraint_clear
            constraint_assisted_transport_success = bool(placed_ok)
            emit(ControllerState.VERIFY_PLACE, placed_ok,
                 None if placed_ok else FailureCode.PLACE_FAILED,
                 f"placed_at={tuple(round(v, 3) for v in after_release_pos)} "
                 f"dist_to_tote={dist_to_tote:.3f} constraint_clear={constraint_clear}",
                 placed_pose=list(after_release_pos), dist_to_tote_center=dist_to_tote,
                 tray_bounds=bounds)

            self._move_tip_validated(tip_above_place, orn=place_orn, kp=0.15, timeout=6.0,
                                     seed_from_current=True)
            if not self.ready(timeout=8.0):
                self.logger.warning("[PLACE] Return to ready pose failed; subsequent pick may fail")
            emit(ControllerState.RETREAT, True, msg="retreated to ready pose")

            if placed_ok:
                self._stats['successful_grasps'] += 1
                emit(ControllerState.DONE, True, msg="pick-and-place complete")
                return finalize(GraspStatus.SUCCESS, None,
                                "contact-validated, constraint-assisted pick-and-place complete",
                                lifted_m=actual_lift, placed_at=tuple(after_release_pos),
                                dist_to_tote_center=dist_to_tote)
            else:
                self._stats['failed_places'] += 1
                emit(ControllerState.DONE, False, FailureCode.PLACE_FAILED, "placement outside tray bounds")
                return finalize(GraspStatus.FAILED_PLACE, FailureCode.PLACE_FAILED,
                                f"Object landed outside tray interior bounds (dist={dist_to_tote:.2f}m)")

        except Exception as e:
            self.logger.error(f"Exception in pick-place state machine: {e}", exc_info=True)
            recovered = True
            try:
                self._release_constraint()
                self.sim.open_gripper()
                self._recover_to_ready()
                self._settle(300)
            except Exception:
                recovered = False
            emit(ControllerState.DONE, False,
                 FailureCode.RECOVERY_FAILED if not recovered else None, str(e))
            return finalize(GraspStatus.FAILED_MOTION,
                            None if recovered else FailureCode.RECOVERY_FAILED, str(e))

    _PUSH_BLADE_HALF = 0.020
    _PUSH_CLOSED_GAP = 0.004
    _PUSH_INCREMENTS = 10
    _PUSH_WALL_MARGIN = 0.03
    _PUSH_DESCENT_SEGMENTS = 4
    _PUSH_COMPLIANT = os.environ.get("PUSH_COMPLIANT", "1") != "0"
    _PUSH_STOP_FORCE = 15.0
    _PUSH_DESCENT_STEP = 0.010
    _PUSH_LAG_STOP = 0.008
    _PUSH_SKID_MAX = 0.030

    def _unwind_periodic_joints(self, margin: float = math.pi) -> List[float]:
        cfg = self._current_joint_state()
        new_cfg = list(cfg)
        changed = False
        for k, q in enumerate(cfg):
            if abs(q) > margin:
                w = (q + math.pi) % (2.0 * math.pi) - math.pi
                new_cfg[k] = w
                changed = True
        if changed:
            for k, j in enumerate(self.joint_indices):
                if new_cfg[k] != cfg[k]:
                    p.resetJointState(self.robot_id, j, new_cfg[k], targetVelocity=0.0)
            p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL,
                                        targetPositions=new_cfg, positionGains=[0.1] * len(new_cfg))
            self.logger.info(f"Unwound periodic joints: {[round(v, 2) for v in cfg]} -> {[round(v, 2) for v in new_cfg]}")
        return new_cfg

    def push_start_pose(self, request: "PushRequest"):
        from causal_core.actions import PUSH_DIRS
        from causal_core.cube_pose import extent_along
        info = self.get_object_info(int(request.target_id))
        if info is None:
            return None
        d = PUSH_DIRS[int(request.dir_index) % len(PUSH_DIRS)]
        yaw = math.atan2(d[1], d[0])
        half = 0.5 * float(extent_along(int(request.target_id), yaw))
        cx, cy = float(info['center'][0]), float(info['center'][1])
        top_z = float(info['aabb_max'][2])
        back = half + float(request.standoff) + self._PUSH_BLADE_HALF
        start = (cx - d[0] * back, cy - d[1] * back, top_z - float(request.depth))
        end = (start[0] + d[0] * float(request.dist), start[1] + d[1] * float(request.dist), start[2])
        return dict(start=start, end=end, yaw=yaw, top_z=top_z, dir=d, half=half)

    def push_descent_clear(self, request: "PushRequest"):
        pose = self.push_start_pose(request)
        if pose is None:
            return False, {"reason": "no_object"}
        sx, sy, sz = pose["start"]
        try:
            lo, hi = self._tray_interior_bounds(int(self.sim.get_ids()["toteA"]))
            m = self._PUSH_WALL_MARGIN
            if not (lo[0] + m <= sx <= hi[0] - m and lo[1] + m <= sy <= hi[1] - m):
                return False, {"reason": "start_too_close_to_wall", "start": pose["start"],
                               "margin": m, "bounds": [list(map(float, lo[:2])), list(map(float, hi[:2]))]}
        except Exception:
            pass
        orn = p.getQuaternionFromEuler([math.pi, 0, pose["yaw"]])
        tips = [(sx, sy, pose["top_z"] + 0.04), (sx, sy, pose["top_z"] + 0.015), (sx, sy, sz)]
        ok, info = self._grasp_descent_clear(-1, tips, orn, preshape_gap=self._PUSH_CLOSED_GAP)
        info = dict(info)
        info.update({"start": pose["start"], "end": pose["end"], "yaw": pose["yaw"], "column_free": bool(ok)})
        if ok or not self._PUSH_COMPLIANT:
            return ok, info
        if info.get("reason") != "descent_collision":
            return False, info
        cube_ids = set(int(b) for b in (getattr(self.sim, "_objects_body_ids", []) or []))
        if int(info.get("body", -1)) not in cube_ids:
            info["reason"] = "descent_collision_tote"
            return False, info
        try:
            from causal_core.actions import column_top
            short = float(column_top(int(request.target_id), int(request.dir_index))) - float(sz)
            if short > self._PUSH_SKID_MAX:
                info.update({"reason": "would_skid", "column_short_m": short})
                return False, info
        except Exception:
            pass
        ok2, info2 = self._blade_descent_clear_ignoring_pieces(tips, orn)
        info2 = dict(info2)
        info2.update({"start": pose["start"], "end": pose["end"], "yaw": pose["yaw"], "column_free": False,
                      "column_blocked_by": int(info.get("body", -1)), "column_block_z": float(info.get("tip", (0, 0, 0))[2])})
        return ok2, info2

    def _blade_descent_clear_ignoring_pieces(self, tip_points, orn):
        margin = self._C6_MARGIN
        attached = self._ompl_attached_bodies(None, None)
        cube_ids = set(int(b) for b in (getattr(self.sim, "_objects_body_ids", []) or []))
        obstacle_ids = [b for b in self._ompl_nearby_obstacle_ids([self.sim._plane_id, -1], None) if int(b) not in cube_ids]
        mount_id = getattr(self.sim, "_mount_body_id", None)
        saved = self._current_joint_state()
        saved_poses = [(bid, p.getBasePositionAndOrientation(bid)) for bid, _ in attached]
        saved_gripper = self.sim.gripper_joint_positions()
        try:
            self.sim.set_gripper_joints_kinematic(float(self._PUSH_CLOSED_GAP))
            cur = list(saved)
            for tip in tip_points:
                ok, _, data = self._solve_and_validate_ik(list(tip), orn, extra_seed=cur)
                if not ok or "joint_solution" not in data:
                    return False, {"reason": "ik_unreachable", "tip": tuple(tip)}
                cfg = self._wrap_joint_targets(data["joint_solution"])
                self._reset_joint_state(cfg)
                self._ompl_place_attached(attached)
                for key, dist, _req in self._ompl_close_pairs(attached, obstacle_ids, margin):
                    a_id, a_link, b_id, _b_link = key
                    if b_id == self.robot_id or a_id == b_id or b_id == mount_id or int(b_id) in cube_ids:
                        continue
                    if dist < margin:
                        return False, {"reason": "descent_collision_tote", "body": int(b_id),
                                       "distance": float(dist), "tip": tuple(tip), "link": int(a_link)}
                cur = cfg
            return True, {}
        finally:
            self._reset_joint_state(saved)
            for bid, (pos, o) in saved_poses:
                p.resetBasePositionAndOrientation(bid, pos, o)
            self.sim.restore_gripper_joints(saved_gripper)

    def _servo_tip_monitored(self, tip_pos, orn, kp: float, timeout: float, tote_id: int,
                             stop_force: float, pos_tol: float = 0.006):
        ee_target = self.geom.tip_to_ee(tip_pos)
        target = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, list(ee_target), orn,
                                              maxNumIterations=200, residualThreshold=1e-4)
        target = list(target)[:len(self.joint_indices)]
        p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL, target,
                                    positionGains=[float(kp)] * len(self.joint_indices))
        g = self.sim._gripper_body_id
        cube_ids = set(int(b) for b in (getattr(self.sim, "_objects_body_ids", []) or []))
        t0 = self.sim.now()
        while True:
            self.sim.step_simulation(1)
            if tote_id >= 0:
                f = sum(float(c[9]) for c in p.getContactPoints(bodyA=g, bodyB=tote_id))
                if f > 5.0:
                    return "tote", f
            f = sum(float(c[9]) for c in p.getContactPoints(bodyA=g) if int(c[2]) in cube_ids)
            if f > float(stop_force):
                return "piece", f
            cur = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
            if max(abs(a - b) for a, b in zip(cur, target)) <= float(pos_tol):
                return "reached", f
            if self.sim.now() - t0 > float(timeout):
                return "timeout", f

    def execute_push_request(self, request: "PushRequest") -> "PushResult":
        t_start = self.sim.now()
        transitions: List[StepResult] = []

        def rec(state, ok, msg, data=None, code=None):
            transitions.append(StepResult(state=state, ok=ok, failure_code=code, message=msg, data=data or {}))

        def done(status, code, msg, **kw):
            self._recover_to_ready()
            return PushResult(status=status, failure_code=code, error_message=msg,
                              execution_time=float(self.sim.now() - t_start),
                              transitions=transitions, **kw)

        cid = int(request.target_id)
        pose = self.push_start_pose(request)
        if pose is None:
            rec(ControllerState.PRECHECK, False, "object not found", code=FailureCode.INVALID_TARGET)
            return done(GraspStatus.INVALID_TARGET, FailureCode.INVALID_TARGET, "object not found")
        ok, info = self.push_descent_clear(request)
        if not ok:
            rec(ControllerState.PRECHECK, False, f"blade descent blocked: {info.get('reason')}",
                data=info, code=FailureCode.PATH_COLLISION)
            return done(GraspStatus.FAILED_MOTION, FailureCode.PATH_COLLISION, "blade descent blocked")
        rec(ControllerState.PRECHECK, True, "push precheck ok",
            data={"start": pose["start"], "end": pose["end"], "yaw": round(pose["yaw"], 3),
                  "dir": pose["dir"], "dist": float(request.dist)})

        others_before = {int(b): np.array(p.getBasePositionAndOrientation(int(b))[0])
                         for b in (getattr(self.sim, "_objects_body_ids", []) or []) if int(b) != cid}
        pos_before = np.array(p.getBasePositionAndOrientation(cid)[0])

        self.sim.set_gripper_opening(self._PUSH_CLOSED_GAP)
        rec(ControllerState.CLOSE, True, "gripper closed to blade", data={"gap": self._PUSH_CLOSED_GAP})

        orn = p.getQuaternionFromEuler([math.pi, 0, pose["yaw"]])
        sx, sy, sz = pose["start"]
        ex, ey, ez = pose["end"]
        self._unwind_periodic_joints()
        if not self._move_safe([sx, sy, max(sz + 0.12, self.geom.safe_z_tip)], orn=orn, kp=0.1, timeout=8.0):
            rec(ControllerState.PREGRASP, False, "approach failed", code=FailureCode.MOTION_TIMEOUT)
            return done(GraspStatus.FAILED_MOTION, FailureCode.MOTION_TIMEOUT, "approach failed")
        rec(ControllerState.PREGRASP, True, "above stroke start")
        self._unwind_periodic_joints()
        tote_id = int(self.sim.get_ids().get("toteA", -1))
        z0 = float(self._get_tip_pos()[2])
        nseg = max(1, int(self._PUSH_DESCENT_SEGMENTS))
        wall_hit = None
        stopped_on = None
        if self._PUSH_COMPLIANT:
            zi = z0
            i = 0
            while zi > sz + 1e-6:
                i += 1
                zi = max(sz, zi - float(self._PUSH_DESCENT_STEP))
                why, f = self._servo_tip_monitored([sx, sy, zi], orn, kp=0.08, timeout=1.2, tote_id=tote_id,
                                                   stop_force=self._PUSH_STOP_FORCE)
                if why == "tote":
                    wall_hit = (i, f)
                    break
                tip_now = self._get_tip_pos()
                if why == "piece":
                    stopped_on = ("contact", float(f), float(tip_now[2]))
                    break
                if float(tip_now[2]) - zi > float(self._PUSH_LAG_STOP):
                    stopped_on = ("lag", float(f), float(tip_now[2]))
                    break
        else:
            for i in range(1, nseg + 1):
                zi = z0 + (sz - z0) * i / nseg
                self._move_tip([sx, sy, zi], orn=orn, kp=0.08, timeout=3.0, pos_tol=0.008)
                if tote_id >= 0:
                    f = sum(float(cpt[9]) for cpt in p.getContactPoints(bodyA=self.sim._gripper_body_id, bodyB=tote_id))
                    if f > 5.0:
                        wall_hit = (i, f)
                        break
        if wall_hit is not None:
            rec(ControllerState.DESCEND, False, f"blade touched the tote wall during descent (segment {wall_hit[0]}, {wall_hit[1]:.0f} N) -- aborting",
                data={"segment": wall_hit[0], "force": wall_hit[1]}, code=FailureCode.PATH_COLLISION)
            tip = self._get_tip_pos()
            self._move_tip([tip[0], tip[1], max(pose["top_z"] + 0.12, self.geom.safe_z_tip)], orn=orn, kp=0.1, timeout=6.0, pos_tol=0.02)
            self.sim.open_gripper()
            return done(GraspStatus.FAILED_MOTION, FailureCode.PATH_COLLISION, "descent touched the tote wall")
        self._settle(60)
        got = self._get_tip_pos()
        derr = float(np.linalg.norm(np.array(got) - np.array([sx, sy, sz])))
        if self._PUSH_COMPLIANT:
            short = max(0.0, float(got[2]) - float(sz))
            sz = float(got[2])
            rec(ControllerState.DESCEND, True,
                (f"blade at stroke start (err={derr:.4f}m)" if stopped_on is None and short < 0.005 else
                 f"blade stopped {short*1000:.0f} mm short of nominal depth "
                 f"({stopped_on[0] if stopped_on else 'setpoint'}, {stopped_on[1] if stopped_on else 0.0:.0f} N on the pile)"),
                data={"tip": [float(v) for v in got], "descent_err": derr, "descent_short_m": short,
                      "stopped_on": stopped_on[0] if stopped_on else None,
                      "stop_force": float(stopped_on[1]) if stopped_on else 0.0})
        else:
            if derr > 0.05:
                rec(ControllerState.DESCEND, False, f"blade descent stalled {derr:.3f}m short",
                    data={"tip": [float(v) for v in got], "descent_err": derr}, code=FailureCode.MOTION_TIMEOUT)
                return done(GraspStatus.FAILED_MOTION, FailureCode.MOTION_TIMEOUT, "descent failed")
            rec(ControllerState.DESCEND, True, f"blade at stroke start (err={derr:.4f}m)",
                data={"tip": [sx, sy, sz], "descent_err": derr})
        n = max(1, int(self._PUSH_INCREMENTS))
        for i in range(1, n + 1):
            tx = sx + (ex - sx) * i / n
            ty = sy + (ey - sy) * i / n
            self._move_tip([tx, ty, sz], orn=orn, kp=0.08, timeout=2.5, pos_tol=0.008)
        tip = self._get_tip_pos()
        travelled = float(np.dot(np.array(tip[:2]) - np.array([sx, sy]), np.array(pose["dir"][:2])))
        stroke_fraction = max(0.0, min(1.0, travelled / max(1e-6, float(request.dist))))
        rec(ControllerState.TRANSIT, True,
            f"stroke complete ({travelled*1000:.0f} of {request.dist*1000:.0f} mm)" if stroke_fraction >= 0.5
            else f"stroke STALLED at {travelled*1000:.0f} of {request.dist*1000:.0f} mm",
            data={"tip": [float(v) for v in tip], "target_end": [ex, ey, sz],
                  "travelled_m": travelled, "stroke_fraction": stroke_fraction})
        self._move_tip([tip[0], tip[1], max(pose["top_z"] + 0.12, self.geom.safe_z_tip)],
                       orn=orn, kp=0.1, timeout=6.0, pos_tol=0.02)
        rec(ControllerState.RETREAT, True, "risen clear")
        self.sim.open_gripper()

        pos_after = np.array(p.getBasePositionAndOrientation(cid)[0])
        moved = float(np.linalg.norm(pos_after - pos_before))
        others = {b: float(np.linalg.norm(np.array(p.getBasePositionAndOrientation(b)[0]) - q))
                  for b, q in others_before.items()}
        rec(ControllerState.DONE, True, f"push complete ({moved*1000:.0f}mm)")
        return done(GraspStatus.SUCCESS, None, "push complete",
                    target_displacement_m=round(moved, 4),
                    max_other_displacement_m=round(max(others.values(), default=0.0), 4),
                    n_others_moved_over_1cm=int(sum(1 for v in others.values() if v > 0.01)))

    def _release_constraint(self):
        if self._grasp_constraint_id is not None:
            try:
                p.removeConstraint(self._grasp_constraint_id)
            except Exception:
                pass
            self._grasp_constraint_id = None
            self._is_grasping = False

    def release_object(self):
        self._release_constraint()

    def get_statistics(self):
        t = self._stats['total_attempts']
        return {**self._stats, 'success_rate': self._stats['successful_grasps'] / max(1, t)}

    def cleanup(self):
        self._release_constraint()


def create_controller_from_sim(sim, enable_logging=True):
    return ProductionRobotController(sim=sim, enable_logging=enable_logging)
