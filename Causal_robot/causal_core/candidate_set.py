"""Candidate records with the corrected top-visibility and clearance gates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from causal_core.feasibility import FeasibilityConfig, FeasibleCandidateSet


@dataclass
class TopVisConfig:
    grid_n: int = 5
    ray_height: float = 0.40
    inset_frac: float = 0.15
    depth_frac: float = 0.5
    visible_frac_thresh: float = 0.55


def top_visibility(cid: int, amin: Sequence[float], amax: Sequence[float],
                    cfg: Optional[TopVisConfig] = None) -> Tuple[float, List[int]]:
    cfg = cfg or TopVisConfig()
    n = max(1, int(cfg.grid_n))
    amin = [float(v) for v in amin]
    amax = [float(v) for v in amax]

    ex, ey = amax[0] - amin[0], amax[1] - amin[1]
    ix, iy = cfg.inset_frac * ex, cfg.inset_frac * ey
    xs = (np.linspace(amin[0] + ix, amax[0] - ix, n) if n > 1
          else np.array([0.5 * (amin[0] + amax[0])]))
    ys = (np.linspace(amin[1] + iy, amax[1] - iy, n) if n > 1
          else np.array([0.5 * (amin[1] + amax[1])]))

    z0 = amax[2] + cfg.ray_height
    z1 = amax[2] - cfg.depth_frac * (amax[2] - amin[2])
    rf = [[float(x), float(y), z0] for x in xs for y in ys]
    rt = [[float(x), float(y), z1] for x in xs for y in ys]
    try:
        res = p.rayTestBatch(rf, rt)
    except Exception:
        return 0.0, []
    ids = [int(r[0]) for r in res]
    hit = sum(1 for h in ids if h == int(cid))
    return hit / max(1, len(ids)), ids


class _CorrectedGate(FeasibleCandidateSet):
    def __init__(self, sim, cfg: Optional[FeasibilityConfig] = None,
                 top_cfg: Optional[TopVisConfig] = None):
        super().__init__(sim, cfg)
        self.top_cfg = top_cfg or TopVisConfig(
            grid_n=self.cfg.top_grid_n,
            ray_height=self.cfg.top_ray_height,
            visible_frac_thresh=self.cfg.top_visible_frac,
        )

    def _top_vis(self, cid, amin, amax) -> float:
        frac, _ids = top_visibility(cid, amin, amax, self.top_cfg)
        return frac

    def _side_clear(self, cid, cx, cy, top_z, yaw, sx, sy) -> float:
        try:
            from causal_core.cube_pose import grasp_yaws
            yaws = grasp_yaws(cid)
        except Exception:
            yaws = [yaw]
        return max(self._side_clear_band(cid, cx, cy, top_z, y, sx, sy) for y in yaws)

    def _side_clear_band(self, cid, cx, cy, top_z, yaw, sx, sy) -> float:
        try:
            centre_z = float(p.getBasePositionAndOrientation(cid)[0][2])
        except Exception:
            centre_z = float(top_z) - 0.023
        tip_z = max(centre_z + 0.006, float(top_z) - 0.02 - 0.015)
        finger_bottom = tip_z + 0.005
        z_lo = min(finger_bottom - 0.003, float(top_z) - 0.004)
        half = 0.5 * (sy if sx <= sy else sx)
        nx, ny = math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)
        zs = np.linspace(z_lo, float(top_z) - 2e-3, self.cfg.n_side_samples)
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

    def _ik_seeds(self):
        cur = [p.getJointState(self.robot, j)[0] for j in self.joints]
        home = list(getattr(self, "rest", cur))[:len(self.joints)]
        if len(home) < len(self.joints):
            home = cur
        flipped = list(home); flipped[2] = -flipped[2]
        shoulder = list(home); shoulder[0] = home[0] + math.pi
        return [cur, home, flipped, shoulder]

    def _ik(self, cx, cy, top_z) -> Tuple[bool, float]:
        ee_t = (cx, cy, top_z + 0.25)
        saved = [p.getJointState(self.robot, j)[0] for j in self.joints]
        if not hasattr(self, "_jlimits"):
            self._jlimits = [(p.getJointInfo(self.robot, j)[8], p.getJointInfo(self.robot, j)[9])
                             for j in self.joints]
        best = 1e9
        try:
            for seed in self._ik_seeds():
                for k, j in enumerate(self.joints):
                    p.resetJointState(self.robot, j, seed[k])
                try:
                    sol = p.calculateInverseKinematics(
                        self.robot, self.ee, ee_t, self._down,
                        maxNumIterations=200, residualThreshold=1e-4)
                except Exception:
                    continue
                sol = list(sol)[:len(self.joints)]
                if not all(math.isfinite(v) for v in sol):
                    continue
                if not all(lo - 0.08 <= v <= hi + 0.08 for v, (lo, hi) in zip(sol, self._jlimits)):
                    continue
                for k, j in enumerate(self.joints):
                    p.resetJointState(self.robot, j, sol[k])
                ls = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True)
                ep = ls[4]
                res = math.sqrt((ep[0] - ee_t[0]) ** 2 + (ep[1] - ee_t[1]) ** 2
                                 + (ep[2] - ee_t[2]) ** 2)
                best = min(best, res)
                if res <= self.cfg.ik_pos_tol:
                    return True, float(res)
        finally:
            for k, j in enumerate(self.joints):
                p.resetJointState(self.robot, j, saved[k])
        return False, float(best)


@dataclass
class CandidateRecord:
    cube_id: int
    top_visible_frac: float
    top_visible: bool
    overlap_above_frac: float
    no_overlap_above: bool
    ik_feasible: bool
    ik_residual: float
    min_side_clearance: float
    side_clearance_ok: bool
    grasp_xy: Tuple[float, float] = (0.0, 0.0)
    top_z: float = 0.0
    grasp_yaw: float = 0.0
    first_hit_ids: Optional[List[int]] = None

    def passes(self, gates: Sequence[str]) -> bool:
        ok = {"C1": self.top_visible, "C2": self.no_overlap_above,
              "C3": self.ik_feasible, "C4": self.side_clearance_ok}
        return all(ok[g] for g in gates)

    def failed(self, gates: Sequence[str]) -> List[str]:
        ok = {"C1": self.top_visible, "C2": self.no_overlap_above,
              "C3": self.ik_feasible, "C4": self.side_clearance_ok}
        return [g for g in gates if not ok[g]]


class CorrectedCandidateSet:

    def __init__(self, sim, top_cfg: Optional[TopVisConfig] = None,
                 feas_cfg: Optional[FeasibilityConfig] = None):
        self.top_cfg = top_cfg or TopVisConfig()
        self.feas_cfg = feas_cfg or FeasibilityConfig(finger_thickness=0.030, safety_margin=0.005)
        self._gate = _CorrectedGate(sim, self.feas_cfg, self.top_cfg)

    def evaluate_c1c2(self, cube_ids: List[int]) -> List[CandidateRecord]:
        out: List[CandidateRecord] = []
        cube_set = set(int(c) for c in cube_ids)
        for cid in (int(c) for c in cube_ids):
            try:
                amin, amax = p.getAABB(cid)
            except Exception:
                continue
            frac, ids = top_visibility(cid, amin, amax, self.top_cfg)
            ov = self._gate._overlap_above(cid, amin, amax, cube_set)
            out.append(CandidateRecord(
                cube_id=cid, top_visible_frac=float(frac),
                top_visible=bool(frac >= self.top_cfg.visible_frac_thresh),
                overlap_above_frac=float(ov),
                no_overlap_above=bool(ov <= self.feas_cfg.overlap_frac_thresh),
                ik_feasible=False, ik_residual=float("nan"),
                min_side_clearance=float("nan"), side_clearance_ok=False,
                first_hit_ids=ids))
        return out

    def evaluate_all(self, cube_ids: List[int]) -> List[CandidateRecord]:
        cube_ids = [int(c) for c in cube_ids]
        recs = self._gate.evaluate_all(cube_ids)
        out: List[CandidateRecord] = []
        for r in recs:
            cid = int(r.cube_id)
            try:
                amin, amax = p.getAABB(cid)
            except Exception:
                continue
            frac, ids = top_visibility(cid, amin, amax, self.top_cfg)
            out.append(CandidateRecord(
                cube_id=cid,
                top_visible_frac=float(frac),
                top_visible=bool(frac >= self.top_cfg.visible_frac_thresh),
                overlap_above_frac=float(1.0 - float(r.no_overlap_above)),
                no_overlap_above=bool(r.no_overlap_above),
                ik_feasible=bool(r.ik_feasible),
                ik_residual=float(r.ik_residual),
                min_side_clearance=float(r.min_side_clearance),
                side_clearance_ok=bool(r.side_clearance_ok),
                grasp_xy=r.grasp_xy, top_z=r.top_z, grasp_yaw=r.grasp_yaw,
                first_hit_ids=ids))
        return out

    def candidates(self, cube_ids: List[int], gates: Sequence[str] = ("C1", "C2"),
                    target_id: Optional[int] = None) -> List[CandidateRecord]:
        ids = [c for c in cube_ids if target_id is None or int(c) != int(target_id)]
        recs = (self.evaluate_c1c2(ids) if set(gates) <= {"C1", "C2"}
                else self.evaluate_all(ids))
        return [r for r in recs if r.passes(gates)]
