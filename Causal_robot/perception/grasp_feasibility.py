"""Grasp feasibility checks on the simulated gripper."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


@dataclass
class GraspInfo:
    block_id: int
    grasp_position: Tuple[float, float, float]
    grasp_yaw: float
    clearance_above: float
    approach_clear: bool
    grasp_quality: float
    feasible: bool


class GraspFeasibilityChecker:

    def __init__(
        self,
        gripper_width: float = 0.085,
        gripper_height: float = 0.12,
        approach_height: float = 0.25,
        min_clearance: float = 0.08,
        ray_check_radius: float = 0.04,
    ):
        self.gripper_width = gripper_width
        self.gripper_height = gripper_height
        self.approach_height = approach_height
        self.min_clearance = min_clearance
        self.ray_check_radius = ray_check_radius

    def check_block(
        self,
        block_id: int,
        other_block_ids: List[int],
        target_id: int,
    ) -> GraspInfo:
        try:
            pos, orn = p.getBasePositionAndOrientation(int(block_id))
            aabb_min, aabb_max = p.getAABB(int(block_id))
        except Exception:
            return GraspInfo(
                block_id=block_id,
                grasp_position=(0, 0, 0),
                grasp_yaw=0,
                clearance_above=0,
                approach_clear=False,
                grasp_quality=0,
                feasible=False,
            )

        grasp_x = float(pos[0])
        grasp_y = float(pos[1])
        grasp_z = float(aabb_max[2])
        grasp_position = (grasp_x, grasp_y, grasp_z)

        euler = p.getEulerFromQuaternion(orn)
        grasp_yaw = float(euler[2])

        clearance = self._check_clearance_above(
            block_id, grasp_position, other_block_ids, target_id
        )

        approach_clear = self._check_approach_path(
            grasp_position, other_block_ids, target_id, block_id
        )

        grasp_quality = self._compute_grasp_quality(
            clearance, approach_clear, grasp_z
        )

        feasible = (
            clearance >= self.min_clearance and
            approach_clear and
            grasp_quality > 0.3
        )

        return GraspInfo(
            block_id=block_id,
            grasp_position=grasp_position,
            grasp_yaw=grasp_yaw,
            clearance_above=clearance,
            approach_clear=approach_clear,
            grasp_quality=grasp_quality,
            feasible=feasible,
        )

    def check_all_blocks(
        self,
        graspable_blocks: List[int],
        all_blocks: List[int],
        target_id: int,
    ) -> Dict[int, GraspInfo]:
        results = {}
        for block_id in graspable_blocks:
            other_blocks = [b for b in all_blocks if b != block_id]
            results[block_id] = self.check_block(block_id, other_blocks, target_id)
        return results

    def _check_clearance_above(
        self,
        block_id: int,
        grasp_position: Tuple[float, float, float],
        other_block_ids: List[int],
        target_id: int,
    ) -> float:
        x, y, z = grasp_position

        ray_from = [x, y, z + 0.01]
        ray_to = [x, y, z + self.approach_height + self.gripper_height]

        try:
            result = p.rayTest(ray_from, ray_to)
            if result and result[0][0] >= 0:
                hit_id = result[0][0]
                if hit_id != block_id and hit_id != target_id:
                    hit_pos = result[0][3]
                    clearance = float(hit_pos[2]) - z
                    return max(0.0, clearance)
        except Exception:
            pass

        return self.approach_height + self.gripper_height

    def _check_approach_path(
        self,
        grasp_position: Tuple[float, float, float],
        other_block_ids: List[int],
        target_id: int,
        block_id: int,
    ) -> bool:
        x, y, z = grasp_position
        approach_z = z + self.approach_height

        offsets = [
            (0, 0),
            (self.ray_check_radius, 0),
            (-self.ray_check_radius, 0),
            (0, self.ray_check_radius),
            (0, -self.ray_check_radius),
        ]

        for dx, dy in offsets:
            ray_from = [x + dx, y + dy, approach_z]
            ray_to = [x + dx, y + dy, z + 0.02]

            try:
                result = p.rayTest(ray_from, ray_to)
                if result and result[0][0] >= 0:
                    hit_id = result[0][0]
                    if hit_id != block_id and hit_id != target_id:
                        return False
            except Exception:
                pass

        return True

    def _compute_grasp_quality(
        self,
        clearance: float,
        approach_clear: bool,
        grasp_z: float,
    ) -> float:
        if not approach_clear:
            return 0.0

        clearance_score = min(1.0, clearance / (self.approach_height + 0.1))

        if 0.1 <= grasp_z <= 0.4:
            height_score = 1.0
        elif grasp_z < 0.1:
            height_score = grasp_z / 0.1
        else:
            height_score = max(0.5, 1.0 - (grasp_z - 0.4) / 0.3)

        return float(clearance_score * 0.6 + height_score * 0.4)

    def find_best_grasp_angle(
        self,
        block_id: int,
        other_block_ids: List[int],
        target_id: int,
        angles_to_try: List[float] = None,
    ) -> Tuple[float, float]:
        if angles_to_try is None:
            angles_to_try = [0, math.pi/4, math.pi/2, 3*math.pi/4]

        best_yaw = 0.0
        best_quality = 0.0

        for yaw in angles_to_try:
            info = self.check_block(block_id, other_block_ids, target_id)
            if info.grasp_quality > best_quality:
                best_quality = info.grasp_quality
                best_yaw = yaw

        return best_yaw, best_quality


def filter_feasible_blocks(
    graspable_blocks: List[int],
    all_blocks: List[int],
    target_id: int,
    checker: GraspFeasibilityChecker = None,
) -> List[int]:
    if checker is None:
        checker = GraspFeasibilityChecker()

    grasp_infos = checker.check_all_blocks(graspable_blocks, all_blocks, target_id)

    feasible_blocks = [
        block_id for block_id, info in grasp_infos.items()
        if info.feasible
    ]

    return feasible_blocks
