"""Scene presets and the scene builder."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pybullet as p

from sim import PyBulletSim


@dataclass
class Scene:
    sim: PyBulletSim
    target_id: Optional[int]
    cube_ids: List[int]
    place_layout: str = "opposite"


def _cube_urdf_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "objects", "cube.urdf")


def _spawn_flat_cube(pos, rng=None) -> int:
    orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    bid = p.loadURDF(_cube_urdf_path(), basePosition=list(pos), baseOrientation=orn,
                     useFixedBase=False)
    p.changeDynamics(bid, -1, lateralFriction=1.0, spinningFriction=0.02,
                     rollingFriction=0.01, linearDamping=0.05, angularDamping=0.1)
    return bid


def build_isolated_scene(seed: int = 0, gui: bool = False) -> Scene:
    sim = PyBulletSim(gui=gui, seed=seed, pile_count=0, layer_count=0,
                      randomize_target=False, require_buried=False,
                      settle_steps=200)
    ws1 = sim._workspace1_bounds
    cx = float((ws1[0, 0] + ws1[0, 1]) / 2.0)
    cy = float((ws1[1, 0] + ws1[1, 1]) / 2.0)
    rng = np.random.default_rng(seed)
    jx, jy = rng.uniform(-0.03, 0.03, size=2)
    z0 = 0.021 + 0.005
    cube_id = _spawn_flat_cube([cx + jx, cy + jy, z0])
    sim._objects_body_ids = [cube_id]
    for _ in range(150):
        sim.step_simulation(1)
    return Scene(sim=sim, target_id=None, cube_ids=[cube_id])


def build_stack_scene(seed: int = 0, gui: bool = False, n: int = 3) -> Scene:
    sim = PyBulletSim(gui=gui, seed=seed, pile_count=0, layer_count=0,
                      randomize_target=False, require_buried=False,
                      settle_steps=200)
    ws1 = sim._workspace1_bounds
    cx = float((ws1[0, 0] + ws1[0, 1]) / 2.0)
    cy = float((ws1[1, 0] + ws1[1, 1]) / 2.0)
    cube_ids = []
    cube_h = 0.042
    for i in range(n):
        z = 0.021 + i * cube_h + 0.004
        cube_ids.append(_spawn_flat_cube([cx, cy, z]))
        for _ in range(120):
            sim.step_simulation(1)
    sim._objects_body_ids = list(cube_ids)
    for _ in range(150):
        sim.step_simulation(1)
    return Scene(sim=sim, target_id=None, cube_ids=cube_ids)


PILE_PRESETS = {
    "aval_het": dict(pile_style="avalanche", heterogeneous=True, aval_count=50, aval_batch=5,
                     aval_radius=0.075, aval_bury_batches=5, aval_height=0.26, aval_tilt_deg=35.0),
    "aval_hom30": dict(pile_style="avalanche", heterogeneous=False, aval_count=30, aval_batch=5,
                       aval_radius=0.075, aval_bury_batches=5, aval_height=0.26, aval_tilt_deg=35.0),
    "aval_hom40": dict(pile_style="avalanche", heterogeneous=False, aval_count=40, aval_batch=5,
                       aval_radius=0.075, aval_bury_batches=5, aval_height=0.26, aval_tilt_deg=35.0),
    "aval_hom50": dict(pile_style="avalanche", heterogeneous=False, aval_count=50, aval_batch=5,
                       aval_radius=0.075, aval_bury_batches=5, aval_height=0.26, aval_tilt_deg=35.0),
    "aval_het_tall": dict(pile_style="avalanche", heterogeneous=True, aval_count=50, aval_batch=3,
                          aval_radius=0.065, aval_bury_batches=17, aval_height=0.10, aval_tilt_deg=35.0),
    "aval_het_slide": dict(pile_style="avalanche", heterogeneous=True, aval_count=50, aval_batch=3,
                           aval_radius=0.10, aval_bury_batches=9, aval_height=0.12, aval_tilt_deg=35.0,
                           aval_slide_speed=0.7, aval_slide_from=0.16),
    "aval_het80": dict(pile_style="avalanche", heterogeneous=True, aval_count=80, aval_batch=3,
                       aval_radius=0.085, aval_bury_batches=27, aval_height=0.10, aval_tilt_deg=35.0),
    "aval_het80_tall": dict(pile_style="avalanche", heterogeneous=True, aval_count=80, aval_batch=3,
                            aval_radius=0.065, aval_bury_batches=27, aval_height=0.10, aval_tilt_deg=35.0),
    "aval_het80_slide": dict(pile_style="avalanche", heterogeneous=True, aval_count=80, aval_batch=3,
                             aval_radius=0.10, aval_bury_batches=18, aval_height=0.12, aval_tilt_deg=35.0,
                             aval_slide_speed=0.7, aval_slide_from=0.16),
}


def build_rescue_scene(seed: int = 42, gui: bool = False,
                       place_layout: str = "opposite", pile: str = "aval_het") -> Scene:
    if pile not in PILE_PRESETS:
        raise ValueError(f"unknown pile preset {pile!r}; choose from {sorted(PILE_PRESETS)}")
    params = dict(PILE_PRESETS[pile])
    if os.environ.get("PILE_OVERRIDE"):
        import json
        params.update(json.loads(os.environ["PILE_OVERRIDE"]))
    cache_dir = os.environ.get("PILE_CACHE")
    cache_path = None
    if cache_dir and not os.environ.get("PILE_OVERRIDE"):
        cache_path = os.path.join(cache_dir, f"{pile}_{seed}.json")
        if os.path.exists(cache_path):
            params["restore_poses"] = cache_path
    try:
        sim = PyBulletSim(gui=gui, seed=seed, require_buried=True, **params)
    except RuntimeError as e:
        if "scene cache" not in str(e) or not cache_path:
            raise
        print(f"[scenes] {e} -- rebuilding in full", flush=True)
        try:
            p.disconnect()
        except Exception:
            pass
        params.pop("restore_poses", None)
        os.remove(cache_path)
        sim = PyBulletSim(gui=gui, seed=seed, require_buried=True, **params)
    if cache_path and not os.path.exists(cache_path):
        sim.save_poses(cache_path)
    ids = sim.get_ids()
    target_id = int(ids["target"])
    cube_ids = [int(c) for c in ids.get("cubes", [])]
    return Scene(sim=sim, target_id=target_id, cube_ids=cube_ids, place_layout=place_layout)
