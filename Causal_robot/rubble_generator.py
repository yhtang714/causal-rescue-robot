"""Random debris pieces (cubes, cuboids, cylinders, plates) with random size, mass and friction."""
import pybullet as p
import numpy as np
import math


DEFAULT_SHAPE_WEIGHTS = {
    "cube":     0.20,
    "cuboid":   0.40,
    "cylinder": 0.25,
    "plate":    0.15,
}


def _sample_shape_class(rng, weights=None):
    if weights is None:
        weights = DEFAULT_SHAPE_WEIGHTS
    shapes = list(weights.keys())
    probs = np.array([weights[s] for s in shapes], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(shapes, p=probs)


def _box_inertia(mass, sx, sy, sz):
    ixx = (1.0 / 12.0) * mass * (sy ** 2 + sz ** 2)
    iyy = (1.0 / 12.0) * mass * (sx ** 2 + sz ** 2)
    izz = (1.0 / 12.0) * mass * (sx ** 2 + sy ** 2)
    return ixx, iyy, izz


def _cylinder_inertia(mass, radius, height):
    ixx = (1.0 / 12.0) * mass * (3 * radius ** 2 + height ** 2)
    iyy = ixx
    izz = 0.5 * mass * radius ** 2
    return ixx, iyy, izz


def sample_rubble_spec(rng, shape_weights=None, max_horizontal=0.034):
    shape = _sample_shape_class(rng, shape_weights)
    mass = float(rng.uniform(0.03, 0.10))
    friction = float(rng.uniform(0.4, 0.8))

    if shape == "cube":
        s = float(rng.uniform(0.030, min(0.040, max_horizontal)))
        sx, sy, sz = s, s, s
        spec_geom = ("box", (sx, sy, sz))
        dims = (sx, sy, sz)

    elif shape == "cuboid":
        sx = float(rng.uniform(0.025, max_horizontal))
        sy = float(rng.uniform(0.025, max_horizontal))
        sz = float(rng.uniform(0.020, 0.060))
        max_dim = max(sx, sy, sz)
        min_dim = min(sx, sy, sz)
        if max_dim / min_dim > 3.0:
            min_allowed = max_dim / 3.0
            sx = max(sx, min_allowed)
            sy = max(sy, min_allowed)
            sz = max(sz, min_allowed)
        sx = min(sx, max_horizontal)
        sy = min(sy, max_horizontal)
        spec_geom = ("box", (sx, sy, sz))
        dims = (sx, sy, sz)

    elif shape == "cylinder":
        radius = float(rng.uniform(0.012, max_horizontal / 2.0))
        height = float(rng.uniform(0.030, 0.070))
        spec_geom = ("cylinder", (radius, height))
        dims = (2 * radius, 2 * radius, height)

    elif shape == "plate":
        sx = float(rng.uniform(0.025, max_horizontal))
        sy = float(rng.uniform(0.025, max_horizontal))
        sz = float(rng.uniform(0.010, 0.018))
        spec_geom = ("box", (sx, sy, sz))
        dims = (sx, sy, sz)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return {
        'shape': shape,
        'mass': mass,
        'friction': friction,
        'dims': dims,
        'geom_spec': spec_geom,
    }


def build_rubble_from_spec(spec, position, orientation, color=(0.6, 0.6, 0.6, 1.0)):
    geom_kind, geom_dims = spec['geom_spec']
    mass = spec['mass']

    if geom_kind == "box":
        sx, sy, sz = geom_dims
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2],
                                     rgbaColor=color)
        ixx, iyy, izz = _box_inertia(mass, sx, sy, sz)
    elif geom_kind == "cylinder":
        radius, height = geom_dims
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                     rgbaColor=color)
        ixx, iyy, izz = _cylinder_inertia(mass, radius, height)
    else:
        raise ValueError(f"Unknown geom kind: {geom_kind}")

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=position,
        baseOrientation=orientation,
    )

    p.changeDynamics(
        body_id, -1,
        mass=mass,
        localInertiaDiagonal=[ixx, iyy, izz],
        lateralFriction=spec['friction'],
        spinningFriction=0.005,
        rollingFriction=0.005,
        linearDamping=0.05,
        angularDamping=0.1,
        frictionAnchor=True,
    )

    return body_id


def create_rubble_object(rng, position, orientation,
                        shape_weights=None,
                        color=(0.6, 0.6, 0.6, 1.0),
                        max_horizontal=0.034):
    spec = sample_rubble_spec(rng, shape_weights=shape_weights,
                              max_horizontal=max_horizontal)
    body_id = build_rubble_from_spec(spec, position, orientation, color=color)
    return {
        'body_id': body_id,
        'shape': spec['shape'],
        'dims': spec['dims'],
        'mass': spec['mass'],
        'friction': spec['friction'],
    }


def get_diverse_palette(n_colors=16):
    base = [
        (0.55, 0.40, 0.30),
        (0.50, 0.50, 0.50),
        (0.70, 0.30, 0.25),
        (0.85, 0.55, 0.25),
        (0.35, 0.40, 0.30),
        (0.60, 0.55, 0.45),
        (0.40, 0.35, 0.30),
        (0.75, 0.70, 0.60),
        (0.80, 0.45, 0.30),
        (0.45, 0.45, 0.55),
        (0.65, 0.60, 0.50),
        (0.35, 0.30, 0.25),
        (0.55, 0.30, 0.20),
        (0.50, 0.55, 0.50),
        (0.70, 0.65, 0.55),
        (0.45, 0.40, 0.35),
    ]
    if n_colors <= len(base):
        return base[:n_colors]
    out = list(base)
    rng = np.random.default_rng(1234)
    while len(out) < n_colors:
        r, g, b = base[len(out) % len(base)]
        jitter = rng.uniform(-0.08, 0.08, 3)
        out.append((float(np.clip(r + jitter[0], 0.1, 0.95)),
                    float(np.clip(g + jitter[1], 0.1, 0.95)),
                    float(np.clip(b + jitter[2], 0.1, 0.95))))
    return out[:n_colors]
