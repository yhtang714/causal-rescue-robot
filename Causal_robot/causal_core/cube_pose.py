"""Grasp yaw candidates from a piece pose."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pybullet as p


def up_axis_and_face_yaw(cube_id: int) -> Tuple[int, float, float]:
    _, orn = p.getBasePositionAndOrientation(int(cube_id))
    R = np.asarray(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
    up = int(np.argmax(np.abs(R[2, :])))
    flat = float(abs(R[2, up]))
    other = [i for i in range(3) if i != up][0]
    ax = R[:, other]
    return up, flat, float(math.atan2(ax[1], ax[0]))


def face_yaw(cube_id: int) -> float:
    return up_axis_and_face_yaw(cube_id)[2]


def grasp_yaws(cube_id: int):
    y = face_yaw(cube_id)
    y = (y + math.pi / 2) % math.pi - math.pi / 2
    return [y, y + math.pi / 2]


def extent_along(cube_id: int, yaw: float, half_side: float = 0.023) -> float:
    _, orn = p.getBasePositionAndOrientation(int(cube_id))
    R = np.asarray(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return float(2.0 * half_side * np.sum(np.abs(d @ R)))
