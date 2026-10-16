"""Geometric feasibility of a pinch grasp: gates C1-C4 on the pile state."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


@dataclass
class FeasibilityConfig:
    top_grid_n: int = 5
    top_visible_frac: float = 0.55
    top_ray_height: float = 0.40
    overlap_frac_thresh: float = 0.20
    above_z_eps: float = 5e-3
    pregrasp_height: float = 0.12
    ik_pos_tol: float = 0.03
    ik_max_iters: int = 80
    tool_tip_offset_z: float = 0.15
    finger_thickness: float = 0.018
    safety_margin: float = 0.008
    side_probe_depth: float = 0.030
    n_side_samples: int = 3
    gripper_max_width: float = 0.085


@dataclass
class CandidateRecord:
    cube_id: int
    grasp_xy: Tuple[float, float]
    top_z: float
    grasp_yaw: float
    top_visible: bool = False
    no_overlap_above: bool = False
    ik_feasible: bool = False
    side_clearance_ok: bool = False
    top_visible_frac: float = 0.0
    contact_count: int = 0
    approach_clearance: float = 0.0
    support_load: float = 0.0
    min_side_clearance: float = 0.0
    ik_residual: float = 0.0

    def feasible(self, gates: Tuple[str, ...]) -> bool:
        ok = True
        if "C1" in gates:
            ok = ok and self.top_visible
        if "C2" in gates:
            ok = ok and self.no_overlap_above
        if "C3" in gates:
            ok = ok and self.ik_feasible
        if "C4" in gates:
            ok = ok and self.side_clearance_ok
        return ok

    def soft_features(self) -> Dict[str, float]:
        return {
            "F_top_visible_frac": float(self.top_visible_frac),
            "F_contact_count": float(self.contact_count),
            "F_approach_clearance": float(self.approach_clearance),
            "F_support_load": float(self.support_load),
            "F_min_side_clearance": float(self.min_side_clearance),
            "F_ik_residual": float(self.ik_residual),
            "F_ik_feasible": 1.0 if self.ik_feasible else 0.0,
            "F_no_overlap_above": 1.0 if self.no_overlap_above else 0.0,
        }


class FeasibleCandidateSet:
    def __init__(self, sim, cfg: Optional[FeasibilityConfig] = None):
        self.cfg = cfg or FeasibilityConfig()
        self.robot = int(sim.robot_body_id)
        self.ee = int(sim.robot_end_effector_link_index)
        self.joints = [int(j) for j in sim._robot_joint_indices]
        self.rest = list(sim.robot_home_joint_config)
        lo, hi = [], []
        for j in self.joints:
            info = p.getJointInfo(self.robot, j)
            jl, ju = float(info[8]), float(info[9])
            if jl >= ju:
                jl, ju = -math.pi, math.pi
            lo.append(jl); hi.append(ju)
        self.lower, self.upper = lo, hi
        self.ranges = [u - l for l, u in zip(lo, hi)]
        self._down = p.getQuaternionFromEuler([math.pi, 0, 0])

    def build(self, cube_ids: List[int], gates: Tuple[str, ...] = ("C1", "C2")) -> List[CandidateRecord]:
        recs = self.evaluate_all(cube_ids)
        return [r for r in recs if r.feasible(gates)]

    def evaluate_all(self, cube_ids: List[int]) -> List[CandidateRecord]:
        cube_set = set(int(c) for c in cube_ids)
        out = []
        for cid in cube_ids:
            r = self._evaluate(int(cid), cube_set)
            if r is not None:
                out.append(r)
        return out

    def _evaluate(self, cid: int, cube_set) -> Optional[CandidateRecord]:
        try:
            pos, orn = p.getBasePositionAndOrientation(cid)
            amin, amax = p.getAABB(cid)
        except Exception:
            return None
        cx, cy = float(pos[0]), float(pos[1])
        top_z = float(amax[2])
        sx, sy = float(amax[0] - amin[0]), float(amax[1] - amin[1])
        yaw = float(p.getEulerFromQuaternion(orn)[2])
        grasp_yaw = yaw if sx <= sy else yaw + math.pi / 2.0
        rec = CandidateRecord(cid, (cx, cy), top_z, grasp_yaw)

        rec.top_visible_frac = self._top_vis(cid, amin, amax)
        rec.top_visible = rec.top_visible_frac >= self.cfg.top_visible_frac

        ov = self._overlap_above(cid, amin, amax, cube_set)
        rec.no_overlap_above = ov <= self.cfg.overlap_frac_thresh

        rec.contact_count = self._contacts(cid)
        rec.support_load = self._support_load(cid, amin, amax, cube_set)
        rec.approach_clearance = self._approach_clear(cid, cx, cy, top_z)

        rec.ik_feasible, rec.ik_residual = self._ik(cx, cy, top_z)

        rec.min_side_clearance = self._side_clear(cid, cx, cy, top_z, grasp_yaw, sx, sy)
        rec.side_clearance_ok = rec.min_side_clearance >= self.cfg.finger_thickness + self.cfg.safety_margin
        return rec

    def _top_vis(self, cid, amin, amax) -> float:
        n = self.cfg.top_grid_n
        xs = np.linspace(amin[0], amax[0], n)
        ys = np.linspace(amin[1], amax[1], n)
        z0 = float(amax[2]) + self.cfg.top_ray_height
        z1 = float(amax[2]) + 1e-3
        rf = [[float(x), float(y), z0] for x in xs for y in ys]
        rt = [[float(x), float(y), z1] for x in xs for y in ys]
        try:
            res = p.rayTestBatch(rf, rt)
        except Exception:
            return 0.0
        hit = sum(1 for r in res if int(r[0]) == cid)
        return hit / max(1, len(res))

    def _overlap_above(self, cid, amin, amax, cube_set) -> float:
        base_z = float(amin[2])
        ax0, ay0, ax1, ay1 = amin[0], amin[1], amax[0], amax[1]
        area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
        mx = 0.0
        for o in cube_set:
            if o == cid:
                continue
            try:
                omin, omax = p.getAABB(o)
            except Exception:
                continue
            if float(omin[2]) <= base_z + self.cfg.above_z_eps:
                continue
            ox0 = max(ax0, omin[0]); oy0 = max(ay0, omin[1])
            ox1 = min(ax1, omax[0]); oy1 = min(ay1, omax[1])
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            mx = max(mx, (ox1 - ox0) * (oy1 - oy0) / area)
        return mx

    def _ik(self, cx, cy, top_z) -> Tuple[bool, float]:
        tip = (cx, cy, top_z + self.cfg.pregrasp_height)
        ee_t = (tip[0], tip[1], tip[2] + self.cfg.tool_tip_offset_z)
        try:
            sol = p.calculateInverseKinematics(
                self.robot, self.ee, ee_t, self._down,
                lowerLimits=self.lower, upperLimits=self.upper,
                jointRanges=self.ranges, restPoses=self.rest,
                maxNumIterations=self.cfg.ik_max_iters)
        except Exception:
            try:
                sol = p.calculateInverseKinematics(
                    self.robot, self.ee, ee_t, self._down,
                    maxNumIterations=self.cfg.ik_max_iters)
            except Exception:
                return False, 1e9
        saved = [p.getJointState(self.robot, j)[0] for j in self.joints]
        try:
            for k, j in enumerate(self.joints):
                if k < len(sol):
                    p.resetJointState(self.robot, j, sol[k])
            ls = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True)
            ep = ls[4]
            res = math.sqrt((ep[0] - ee_t[0]) ** 2 + (ep[1] - ee_t[1]) ** 2 + (ep[2] - ee_t[2]) ** 2)
        finally:
            for k, j in enumerate(self.joints):
                p.resetJointState(self.robot, j, saved[k])
        return (res <= self.cfg.ik_pos_tol), float(res)

    def _side_clear(self, cid, cx, cy, top_z, yaw, sx, sy) -> float:
        half = 0.5 * (sy if sx <= sy else sx)
        nx, ny = math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)
        zs = np.linspace(top_z - self.cfg.side_probe_depth, top_z - 2e-3, self.cfg.n_side_samples)
        reach = self.cfg.gripper_max_width
        clears = []
        for s in (+1.0, -1.0):
            ex, ey = cx + s * nx * half, cy + s * ny * half
            rf = [[ex, ey, float(z)] for z in zs]
            rt = [[ex + s * nx * reach, ey + s * ny * reach, float(z)] for z in zs]
            try:
                res = p.rayTestBatch(rf, rt)
            except Exception:
                res = []
            d = reach
            for r in res:
                if int(r[0]) >= 0 and int(r[0]) != cid:
                    d = min(d, float(r[2]) * reach)
            clears.append(d)
        return min(clears) if clears else 0.0

    def _contacts(self, cid) -> int:
        try:
            cps = p.getContactPoints(bodyA=cid)
        except Exception:
            return 0
        bodies = set()
        for cp in cps:
            other = cp[2] if cp[1] == cid else cp[1]
            if other != cid:
                bodies.add(other)
        return len(bodies)

    def _support_load(self, cid, amin, amax, cube_set) -> float:
        base_z = float(amin[2])
        ax0, ay0, ax1, ay1 = amin[0], amin[1], amax[0], amax[1]
        load = 0.0
        for o in cube_set:
            if o == cid:
                continue
            try:
                omin, omax = p.getAABB(o)
            except Exception:
                continue
            if float(omin[2]) <= base_z + self.cfg.above_z_eps:
                continue
            if min(ax1, omax[0]) <= max(ax0, omin[0]) or min(ay1, omax[1]) <= max(ay0, omin[1]):
                continue
            try:
                load += float(p.getDynamicsInfo(o, -1)[0])
            except Exception:
                load += 0.05
        return load

    def _approach_clear(self, cid, cx, cy, top_z) -> float:
        try:
            res = p.rayTest([cx, cy, top_z + 1e-3], [cx, cy, top_z + 0.6])
        except Exception:
            return 0.6
        if res and int(res[0][0]) >= 0 and int(res[0][0]) != cid:
            return float(res[0][3][2]) - top_z
        return 0.6
