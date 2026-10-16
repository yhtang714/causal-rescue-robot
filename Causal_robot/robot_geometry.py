"""Measured geometry of the gripper and the tool tip."""
from dataclasses import dataclass, field
from typing import Set


@dataclass
class RobotGeometry:

    tool_tip_offset_z: float = 0.15
    """ee_link 到指尖在世界坐标 z 方向的距离 (正数, 米)."""

    safe_z_tip: float = 0.35
    """指尖巡航高度 — 高于堆叠最大高度, 避免横向飞越时撞到墙."""

    right_finger_links: Set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5})
    left_finger_links: Set[int] = field(default_factory=lambda: {6, 7, 8, 9, 10})

    def tip_to_ee(self, tip_pos):
        return (float(tip_pos[0]),
                float(tip_pos[1]),
                float(tip_pos[2]) + self.tool_tip_offset_z)

    def ee_to_tip(self, ee_pos):
        return (float(ee_pos[0]),
                float(ee_pos[1]),
                float(ee_pos[2]) - self.tool_tip_offset_z)

    def classify_finger_link(self, link_idx: int) -> str:
        if link_idx in self.left_finger_links:
            return 'left'
        if link_idx in self.right_finger_links:
            return 'right'
        return 'other'
