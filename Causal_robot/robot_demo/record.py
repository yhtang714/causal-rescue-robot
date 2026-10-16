"""Runs the loop on the arm and writes the frames of the demo video; the same hooks with DEMO_RENDER=0 give a headless run."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pybullet as p
import cv2
import time as _t
_FAKE = {"t": _t.time()}
_t.time = lambda: _FAKE["t"]

from robot_demo.scenes import build_rescue_scene
from robot_demo.coordinator import RescueCoordinator

SEED = int(os.environ.get("DEMO_SEED", "42"))
PILE = os.environ.get("DEMO_PILE", "aval_het")
scene = build_rescue_scene(seed=SEED, gui=False, place_layout="opposite", pile=PILE)
sim = scene.sim
POLICY = os.environ.get("DEMO_POLICY", "heuristic")
MODEL = os.environ.get("DEMO_MODEL", None)
coord = RescueCoordinator(sim=sim, policy=POLICY, model_path=MODEL, enable_logging=True,
                          place_layout="opposite", max_steps=int(os.environ.get("DEMO_STEPS", "3")),
                          nudge_mm=float(os.environ.get("DEMO_NUDGE_MM", "0")),
                          lookahead=int(os.environ.get("DEMO_LOOKAHEAD", "0")),
                          actions=os.environ.get("DEMO_ACTIONS", "pick"),
                          push_top=int(os.environ.get("DEMO_PUSH_TOP", "4")),
                          verify_k=int(os.environ.get("DEMO_VERIFY_K", "0")))
c = coord.controller
PHASE = {"name": "INIT"}
rows = []
step = {"n": 0}
prev_wrist = {"p": None}

CUR_TARGET = {"id": None}
ALL_CUBES = [int(x) for x in sim.get_ids()["cubes"]]
PICK_START_Z = {}
_orig_egr = c.execute_grasp_request
def egr(req, *a, **kw):
    CUR_TARGET["id"] = int(req.target_id)
    PICK_START_Z.clear()
    for cid in ALL_CUBES:
        try:
            PICK_START_Z[cid] = float(p.getBasePositionAndOrientation(cid)[0][2])
        except Exception:
            pass
    try:
        return _orig_egr(req, *a, **kw)
    finally:
        CUR_TARGET["id"] = None
c.execute_grasp_request = egr

IN_PUSH = {"on": False, "cid": None, "dir": None}
_orig_push = getattr(c, "execute_push_request", None)
if _orig_push is not None:
    def epr(req, *a, **kw):
        IN_PUSH.update({"on": True, "cid": int(req.target_id), "dir": int(req.dir_index)})
        PICK_START_Z.clear()
        try:
            return _orig_push(req, *a, **kw)
        finally:
            IN_PUSH.update({"on": False, "cid": None, "dir": None})
            orig = tinted.pop("push_orig", None)
            pid = tinted.pop("push", None)
            if pid is not None and orig is not None:
                try: p.changeVisualShape(int(pid), -1, rgbaColor=list(orig))
                except Exception: pass
    c.execute_push_request = epr

def held_id():
    cid = c._grasp_constraint_id
    if cid is None:
        return None
    try:
        return int(p.getConstraintInfo(cid)[2])
    except Exception:
        return None

def sample():
    ls = p.getLinkState(c.robot_id, c.end_effector_index)
    wrist = np.array(ls[0])
    hid = held_id()
    obj = None
    if hid is not None:
        obj = np.array(p.getBasePositionAndOrientation(hid)[0])
    disp = 0.0 if prev_wrist["p"] is None else float(np.linalg.norm(wrist - prev_wrist["p"]))
    prev_wrist["p"] = wrist
    allowed = {sim._plane_id}
    if hid is not None:
        allowed.add(hid)
    allowed.add(scene.target_id)
    for tid in getattr(scene, "tote_ids", []) or []:
        allowed.add(tid)
    maxf, maxf_body = 0.0, None
    for cp in p.getContactPoints(bodyA=c.robot_id):
        other = int(cp[2])
        if other == c.robot_id or other in allowed:
            continue
        f = float(cp[9])
        if f > maxf:
            maxf, maxf_body = f, other
    selff = 0.0
    for cp in p.getContactPoints(bodyA=c.robot_id, bodyB=c.robot_id):
        selff = max(selff, float(cp[9]))
    gid = sim._gripper_body_id
    g_allowed = set(allowed) | {c.robot_id, gid}
    if CUR_TARGET["id"] is not None:
        g_allowed.add(CUR_TARGET["id"])
    if IN_PUSH["on"] and IN_PUSH["cid"] is not None:
        g_allowed.add(int(IN_PUSH["cid"]))
    gmax, gmax_body = 0.0, None
    for cp in p.getContactPoints(bodyA=gid):
        other = int(cp[2])
        if other in g_allowed:
            continue
        f = float(cp[9])
        if f > gmax:
            gmax, gmax_body = f, other
    rider = None
    if hid is not None and obj is not None and obj[2] > 0.20:
        for cid in ALL_CUBES:
            if cid == hid:
                continue
            cp_ = p.getBasePositionAndOrientation(cid)[0]
            z0 = PICK_START_Z.get(int(cid))
            if cp_[2] > 0.20 and np.linalg.norm(np.array(cp_) - obj) < 0.08 and z0 is not None and cp_[2] - z0 > 0.02:
                rider = int(cid)
                break
    rows.append(dict(step=step["n"], phase=("PUSH:" + PHASE["name"]) if IN_PUSH["on"] else PHASE["name"],
                     push=int(IN_PUSH["on"]), push_cid=IN_PUSH["cid"],
                     wx=float(wrist[0]), wy=float(wrist[1]), wz=float(wrist[2]),
                     rider=rider,
                     held=hid, ox=None if obj is None else float(obj[0]), oy=None if obj is None else float(obj[1]),
                     oz=None if obj is None else float(obj[2]),
                     sep=None if obj is None else float(np.linalg.norm(wrist - obj)),
                     disp=disp, ext_force=maxf, ext_body=maxf_body, self_force=selff,
                     grip_force=gmax, grip_body=gmax_body, target=CUR_TARGET["id"],
                     joints=[round(v, 4) for v in c._current_joint_state()]))

OUT_DIR = os.environ.get("DEMO_FRAMES", os.path.join("output", "frames"))
os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR): os.remove(os.path.join(OUT_DIR, f))
RENDER = os.environ.get("DEMO_RENDER", "1") != "0"
W, H = int(os.environ.get("DEMO_W", "1280")), int(os.environ.get("DEMO_H", "720"))
RW, RH = int(os.environ.get("DEMO_RENDER_W", "1920")), int(os.environ.get("DEMO_RENDER_H", "1080"))
LIGHT = [0.35, -0.55, 1.0]
FRAME_EXT = os.environ.get("DEMO_FRAME_EXT", "jpg")
CONTACT = {"DESCEND", "DESCEND(compliant)", "STROKE", "CLOSE", "PRESHAPE", "MICRO_LIFT", "LIFT", "RISE"}
APPROACH = {"PREGRASP_SAFE", "APPROACH", "OPEN"}
def capture_every():
    ph = PHASE["name"]
    if IN_PUSH["on"]:
        if ph in ("DESCEND", "DESCEND(compliant)", "STROKE", "RISE"):
            return 8
        if ph == "PRESHAPE":
            return 16
        if ph == "OPEN":
            return 32
    if ph in CONTACT:
        return 8
    if ph in APPROACH:
        return 16
    return 32

from PIL import Image, ImageDraw, ImageFont
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
def _font(size, bold=False):
    try:
        return ImageFont.truetype(os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()
F_BIG, F_MED, F_SMALL, F_TINY = _font(30, True), _font(24), _font(20), _font(17)
INK, SUB, WHITE = (24, 24, 24), (90, 90, 88), (255, 255, 255)
BLUE, ORANGE, GREEN = (42, 120, 214), (235, 104, 52), (40, 190, 80)
BLUE_BGR, ORANGE_BGR = (214, 120, 42), (52, 104, 235)

proj = p.computeProjectionMatrixFOV(60, W / H, 0.01, 3.0)
proj_action = p.computeProjectionMatrixFOV(50, W / H, 0.01, 3.0)
frame_i = {"n": 0}; tinted = {"id": None}; LAST_EXPO = {"v": float('nan')}
META = {"frames": [], "decisions": [], "pile": PILE, "seed": SEED, "W": W, "H": H, "fps": 30}
ACTION_NO = {"n": 0, "txt": None}

_TOTE_XY = {"c": None, "half": None}
def _tote_footprint():
    if _TOTE_XY["c"] is None:
        try:
            b = c._tray_interior_bounds(int(sim.get_ids()["toteA"]))
            lo, hi = np.array(b[0]), np.array(b[1])
            _TOTE_XY["c"] = 0.5 * (lo[:2] + hi[:2]); _TOTE_XY["half"] = 0.5 * (hi[:2] - lo[:2])
            _TOTE_XY["top"] = float(hi[2]) if len(hi) > 2 else 0.16
        except Exception:
            tA = np.array(sim._toteA_pos)
            _TOTE_XY["c"] = tA[:2]; _TOTE_XY["half"] = np.array([0.20, 0.15]); _TOTE_XY["top"] = 0.16
    return _TOTE_XY["c"], _TOTE_XY["half"], _TOTE_XY["top"]
CAM = {"mode": "wide", "hold": 0}
def _tip_pos():
    try:
        return np.array(c._get_tip_pos())
    except Exception:
        ls = p.getLinkState(c.robot_id, c.end_effector_index)
        return np.array(ls[0]) + np.array([0.0, 0.0, -0.15])

def _render(view, projm, w=None, h=None):
    w = w or W; h = h or H
    rw, rh = int(round(w * RW / W)), int(round(h * RH / H))
    img = p.getCameraImage(rw, rh, viewMatrix=view, projectionMatrix=projm, renderer=p.ER_TINY_RENDERER,
                           flags=p.ER_NO_SEGMENTATION_MASK, lightDirection=LIGHT, shadow=0)
    rgb = np.reshape(img[2], (rh, rw, 4))[:, :, :3].astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if (rw, rh) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr

def _pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
def _bgr(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def _panel(draw, x0, y0, x1, y1, alpha=190):
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, alpha))

def _exposure_bar(ovl, x, y, w, h, value, thresh):
    d = ImageDraw.Draw(ovl)
    _panel(d, x - 12, y - 34, x + w + 12, y + h + 12)
    v = 0.0 if (value is None or (isinstance(value, float) and math.isnan(value))) else max(0.0, min(1.0, float(value)))
    d.text((x, y - 31), f"Target exposure  {v:.2f}", font=F_SMALL, fill=INK)
    d.rectangle([x, y, x + w, y + h], fill=(230, 229, 225, 255))
    fill = GREEN if v >= thresh else BLUE
    d.rectangle([x, y, x + int(w * v), y + h], fill=fill + (255,))
    tx = x + int(w * thresh)
    d.line([tx, y - 5, tx, y + h + 5], fill=INK + (255,), width=2)
    d.text((tx - 16, y + h + 6), f"{thresh:.2f}", font=F_TINY, fill=INK)

def _overlay_frame(bgr, speed_x, phase_word, inset=False):
    im = _pil(bgr).convert("RGBA")
    ovl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ovl)
    if inset:
        iw, ih = W // 3, H // 3
        txt = "top view"
        tw = d.textlength(txt, font=F_TINY)
        _panel(d, W - 16 - tw - 16, 16 + ih + 6, W - 16, 16 + ih + 32, alpha=170)
        d.text((W - 16 - tw - 8, 16 + ih + 9), txt, font=F_TINY, fill=SUB)
    if ACTION_NO["txt"]:
        line1 = f"Action {ACTION_NO['n']}"
        line2 = ACTION_NO["txt"]
        tw = max(d.textlength(line1, font=F_BIG), d.textlength(line2, font=F_MED))
        _panel(d, 18, 16, 18 + tw + 28, 16 + 84)
        d.text((32, 22), line1, font=F_BIG, fill=INK)
        d.text((32, 58), line2, font=F_MED, fill=BLUE)
    txt = f"{phase_word}   {speed_x}"
    tw = d.textlength(txt, font=F_SMALL)
    _panel(d, 18, H - 52, 18 + tw + 28, H - 14)
    d.text((32, H - 46), txt, font=F_SMALL, fill=SUB)
    _exposure_bar(ovl, W - 300, H - 44, 260, 14, LAST_EXPO["v"], float(coord.exposed_thresh))
    im = Image.alpha_composite(im, ovl).convert("RGB")
    return _bgr(im)

def _phase_word():
    ph = PHASE["name"]
    if IN_PUSH["on"]:
        return {"DESCEND": "push: lowering the closed jaws", "DESCEND(compliant)": "push: compliant descent",
                "STROKE": "push: stroke", "RISE": "push: rising"}.get(ph, "push")
    return {"PREGRASP_SAFE": "approach", "APPROACH": "approach", "DESCEND": "pick: descending",
            "PRESHAPE": "pick: opening the jaws", "CLOSE": "pick: closing", "MICRO_LIFT": "pick: lift test",
            "LIFT": "pick: lifting", "TRANSIT(planned)": "transit", "ROTATE_TO_PLACE": "transit",
            "ROTATE_TO_PICK": "transit", "LOWER": "placing", "OPEN": "releasing", "RETREAT": "retreat",
            "HOME": "parking", "READY": "ready", "INIT": ""}.get(ph, ph.lower())

def _write(bgr, kind, extra=None):
    path = os.path.join(OUT_DIR, f"f{frame_i['n']:05d}.{FRAME_EXT}")
    if FRAME_EXT == "jpg":
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    else:
        cv2.imwrite(path, bgr)
    rec = dict(i=frame_i["n"], kind=kind, step=step["n"], phase=PHASE["name"], push=int(IN_PUSH["on"]),
               action=ACTION_NO["n"], exposure=(None if math.isnan(LAST_EXPO["v"]) else round(LAST_EXPO["v"], 4)))
    if extra: rec.update(extra)
    META["frames"].append(rec)
    frame_i["n"] += 1

def capture_frame(every):
    if not RENDER:
        return
    hid = held_id()
    if hid is not None and tinted["id"] != hid:
        try: p.changeVisualShape(hid, -1, rgbaColor=[0.16, 0.47, 0.84, 1.0]); tinted["id"] = hid
        except Exception: pass
    if IN_PUSH["on"] and IN_PUSH["cid"] is not None and tinted.get("push") != IN_PUSH["cid"]:
        try:
            vs = p.getVisualShapeData(int(IN_PUSH["cid"]))
            tinted["push_orig"] = tuple(vs[0][7]) if vs else None
            p.changeVisualShape(int(IN_PUSH["cid"]), -1, rgbaColor=[0.16, 0.47, 0.84, 1.0]); tinted["push"] = int(IN_PUSH["cid"])
        except Exception: pass
    ls = p.getLinkState(c.robot_id, c.end_effector_index)
    wrist = np.array(ls[0])
    tip = _tip_pos()
    obj = np.array(p.getBasePositionAndOrientation(hid)[0]) if hid is not None else None
    tc, th, ttop = _tote_footprint()
    inside = (abs(tip[0] - tc[0]) < th[0] + 0.06 and abs(tip[1] - tc[1]) < th[1] + 0.06 and tip[2] < ttop + 0.14)
    want = "action" if inside else "wide"
    if want != CAM["mode"]:
        CAM["hold"] += 1
        if CAM["hold"] >= 3:
            CAM["mode"], CAM["hold"] = want, 0
    else:
        CAM["hold"] = 0
    base_xy = np.array([0.0, 0.0])
    if CAM["mode"] == "action":
        target = tip + np.array([0.0, 0.0, -0.04])
        out = target[:2] - base_xy
        az = np.arctan2(out[1], out[0]) + np.deg2rad(80)
        pitch_rad, dist_cam = np.deg2rad(-58), 0.40
        proj_use = proj_action
    else:
        target = 0.5 * (wrist + obj) if obj is not None else wrist + np.array([0.0, 0.0, -0.12])
        out = target[:2] - base_xy
        az = np.arctan2(out[1], out[0]) + np.deg2rad(35)
        pitch_rad, dist_cam = np.deg2rad(-22), 0.9
        proj_use = proj
    eye = target + dist_cam * np.array([np.cos(pitch_rad) * np.cos(az), np.cos(pitch_rad) * np.sin(az), -np.sin(pitch_rad)])
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
    bgr = _render(view, proj_use)
    if CAM["mode"] == "action":
        iw, ih = W // 3, H // 3
        small, _tv, _tp = _render_td(iw, ih)
        cv2.rectangle(small, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 2)
        bgr[16:16 + ih, W - iw - 16:W - 16] = small
    speed = {8: "1x", 16: "2x", 32: "4x"}.get(int(every), f"{every/8:.0f}x")
    bgr = _overlay_frame(bgr, speed, _phase_word(), inset=(CAM["mode"] == "action"))
    _write(bgr, "run", dict(cam=CAM["mode"], every=int(every)))

orig_step = sim.step_simulation
def instrumented(num_steps):
    for _ in range(int(num_steps)):
        orig_step(1)
        _FAKE["t"] += 1.0 / 240.0
        step["n"] += 1
        sample()
        ev = capture_every()
        if step["n"] % ev == 0:
            capture_frame(ev)
sim.step_simulation = instrumented

def tag(name, fn):
    def w(*a, **kw):
        old = PHASE["name"]
        PHASE["name"] = name
        try:
            return fn(*a, **kw)
        finally:
            PHASE["name"] = old
    return w

c.rotate_to_place_side = tag("ROTATE_TO_PLACE", c.rotate_to_place_side)
c.rotate_to_pick_side = tag("ROTATE_TO_PICK", c.rotate_to_pick_side)
c.home = tag("HOME", c.home)
c.ready = tag("READY", c.ready)
c._move_safe = tag("PREGRASP_SAFE", c._move_safe)
_orig_mt = c._move_tip
def mt(tip, *a, **kw):
    old = PHASE["name"]
    hid = held_id()
    if IN_PUSH["on"]:
        try:
            cur = np.array(c._get_tip_pos())
        except Exception:
            cur = None
        tgt = np.array(tip, dtype=float)
        if cur is not None and abs(tgt[2] - cur[2]) < 0.02 and np.linalg.norm(tgt[:2] - cur[:2]) > 0.003:
            PHASE["name"] = "STROKE"
        elif cur is not None and tgt[2] > cur[2] + 0.02:
            PHASE["name"] = "RISE"
        else:
            PHASE["name"] = "DESCEND"
    else:
        PHASE["name"] = "LIFT" if hid is not None else ("DESCEND" if tip[2] < 0.2 else "APPROACH")
    try:
        return _orig_mt(tip, *a, **kw)
    finally:
        PHASE["name"] = old
c._move_tip = mt
if hasattr(c, "_servo_tip_monitored"):
    _orig_stm = c._servo_tip_monitored
    def stm(tip, *a, **kw):
        old = PHASE["name"]
        PHASE["name"] = "DESCEND(compliant)" if IN_PUSH["on"] else old
        try:
            return _orig_stm(tip, *a, **kw)
        finally:
            PHASE["name"] = old
    c._servo_tip_monitored = stm
_orig_mtv = c._move_tip_validated
def mtv(tip, *a, **kw):
    old = PHASE["name"]
    hid = held_id()
    if kw.get("use_planning"):
        PHASE["name"] = "TRANSIT(planned)"
    elif hid is not None:
        PHASE["name"] = "LOWER"
    elif old.startswith("PREGRASP"):
        PHASE["name"] = old
    else:
        PHASE["name"] = "RETREAT"
    try:
        return _orig_mtv(tip, *a, **kw)
    finally:
        PHASE["name"] = old
c._move_tip_validated = mtv
sim.close_gripper = tag("CLOSE", sim.close_gripper)
sim.open_gripper = tag("OPEN", sim.open_gripper)
sim.set_gripper_opening = tag("PRESHAPE", sim.set_gripper_opening)
c._micro_lift_check = tag("MICRO_LIFT", c._micro_lift_check) if hasattr(c, "_micro_lift_check") else None


def current_exposure():
    ids = sim.get_ids(); alive = [int(x) for x in ids["cubes"] if int(x) not in coord._placed]
    try:
        return float(coord._observe(scene.target_id, alive)["human_exposure"])
    except Exception:
        return float("nan")

TD_EYE_Z = float(os.environ.get("DEMO_TD_EYE_Z", "0.62"))
def _td_matrices():
    tA = sim._toteA_pos
    target = [float(tA[0]), float(tA[1]), 0.05]
    eye = [float(tA[0]) + 1e-3, float(tA[1]), TD_EYE_Z]
    view = p.computeViewMatrix(eye, target, [1, 0, 0])
    proj_td = p.computeProjectionMatrixFOV(48, W / H, 0.01, 3.0)
    return view, proj_td

_ARM_COLS = []
def _arm_colours():
    if _ARM_COLS:
        return _ARM_COLS
    for bid in (c.robot_id, sim._gripper_body_id):
        if bid is None:
            continue
        try:
            vsd = p.getVisualShapeData(int(bid))
        except Exception:
            continue
        for v in vsd:
            _ARM_COLS.append((int(bid), int(v[1]), [float(x) for x in v[7]]))
    return _ARM_COLS
def _render_td(w=None, h=None):
    view, proj_td = _td_matrices()
    cols = _arm_colours()
    for bid, link, col in cols:
        try: p.changeVisualShape(bid, link, rgbaColor=[col[0], col[1], col[2], 0.0])
        except Exception: pass
    try:
        bgr = _render(view, proj_td, w, h)
    finally:
        for bid, link, col in cols:
            try: p.changeVisualShape(bid, link, rgbaColor=col)
            except Exception: pass
    return bgr, view, proj_td
def _project(view, proj_td, xyz):
    V = np.array(view).reshape(4, 4).T; P = np.array(proj_td).reshape(4, 4).T
    q = P @ (V @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))
    if abs(q[3]) < 1e-9: return None
    nx, ny = q[0] / q[3], q[1] / q[3]
    return int((nx + 1) * 0.5 * W), int((1 - ny) * 0.5 * H)

def _target_outline(im_draw, view, proj_td, width=3):
    try:
        amin, amax = p.getAABB(scene.target_id)
        corners = [_project(view, proj_td, [x, y, amax[2]]) for x, y in
                   ((amin[0], amin[1]), (amax[0], amin[1]), (amax[0], amax[1]), (amin[0], amax[1]))]
        if all(cc is not None for cc in corners):
            im_draw.polygon(corners, outline=GREEN + (255,), width=width)
    except Exception:
        pass

def capture_topdown(n_frames, label, kind, highlight_target=False):
    if not RENDER:
        return current_exposure()
    if highlight_target:
        try:
            for link in range(-1, p.getNumJoints(scene.target_id)):
                p.changeVisualShape(scene.target_id, link, rgbaColor=[0.1, 1.0, 0.1, 1.0])
        except Exception:
            pass
    expo = current_exposure()
    LAST_EXPO["v"] = expo
    bgr, view, proj_td = _render_td()
    im = _pil(bgr).convert("RGBA")
    ovl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ovl)
    _target_outline(d, view, proj_td)
    tw = d.textlength(label, font=F_BIG)
    _panel(d, 18, 16, 18 + tw + 28, 16 + 50)
    d.text((32, 22), label, font=F_BIG, fill=INK)
    _exposure_bar(ovl, W - 300, H - 44, 260, 14, expo, float(coord.exposed_thresh))
    bgr = _bgr(Image.alpha_composite(im, ovl).convert("RGB"))
    for _ in range(n_frames):
        _write(bgr, kind)
    return expo


DECISION_HOLD = int(os.environ.get("DEMO_DECISION_HOLD", "75"))
ACTION_TXT = {"s": None}
DECISION_INFO = {"actions": None, "scores": None, "chosen": -1, "verify": None}
import re as _re
DIRV = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
def _action_words(a):
    m = _re.match(r"(pick|push)\((\d+)(?:,([+-][xy]))?\)", a or "")
    if not m:
        return a or ""
    if m.group(1) == "pick":
        return f"pick piece {m.group(2)}"
    return f"push piece {m.group(2)} towards {m.group(3)}"

def capture_decision_view(step_i, exposure, feasible, selected, scores, status="executing"):
    view, proj_td = _td_matrices()
    acts = DECISION_INFO.get("actions") or [f"pick({cc})" for cc in feasible]
    chosen_i = int(DECISION_INFO.get("chosen", -1))
    v = DECISION_INFO.get("verify") or {}
    verified = {int(k): r for k, r in (v.get("verified") or {}).items()}
    ranker_top = int(v.get("ranker_top", -1)) if v else -1
    META["decisions"].append(dict(frame=frame_i["n"], step=int(step_i), exposure=float(exposure), actions=list(acts),
                                  chosen=chosen_i, verified={int(k): dict(action=r["action"], dE=r["dE"], cf=r["cf_value"], trap=r["trap"])
                                                             for k, r in verified.items()},
                                  n_actions=len(acts)))
    if not RENDER:
        return
    bgr, view, proj_td = _render_td()
    im = _pil(bgr).convert("RGBA")
    ovl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ovl)
    _target_outline(d, view, proj_td)
    labels = []
    draw_idx = sorted(verified.keys()) if verified else ([chosen_i] if chosen_i >= 0 else [])
    for i in draw_idx:
        a = acts[i] if 0 <= i < len(acts) else None
        m = _re.match(r"(pick|push)\((\d+)(?:,([+-][xy]))?\)", a or "")
        if not m:
            continue
        kind, cid, dtxt = m.group(1), int(m.group(2)), m.group(3)
        try:
            pos = p.getBasePositionAndOrientation(cid)[0]
        except Exception:
            continue
        uv = _project(view, proj_td, pos)
        if uv is None:
            continue
        r = verified.get(i)
        is_chosen = (i == chosen_i)
        is_trap = bool(r and r["trap"])
        col = BLUE if is_chosen else (ORANGE if is_trap else WHITE)
        rad = 26 if is_chosen else 20
        d.ellipse([uv[0] - rad, uv[1] - rad, uv[0] + rad, uv[1] + rad], outline=col + (255,), width=4 if is_chosen else 3)
        d.text((uv[0] + rad + 4, uv[1] - rad - 2), str(cid), font=F_SMALL, fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
        if kind == "push" and dtxt in DIRV:
            dd = DIRV[dtxt]
            uv2 = _project(view, proj_td, [pos[0] + 0.07 * dd[0], pos[1] + 0.07 * dd[1], pos[2]])
            if uv2 is not None:
                d.line([uv, uv2], fill=col + (255,), width=5 if is_chosen else 3)
                ang = math.atan2(uv2[1] - uv[1], uv2[0] - uv[0])
                for sgn in (+1, -1):
                    d.line([uv2, (uv2[0] - 14 * math.cos(ang - sgn * 0.5), uv2[1] - 14 * math.sin(ang - sgn * 0.5))],
                           fill=col + (255,), width=5 if is_chosen else 3)
        short = f"pick {cid}" if kind == "pick" else f"push {cid} {(dtxt or '').replace('-', chr(0x2212))}"
        txt = f"{short}   \u0394E {r['dE']:+.2f}" if r else short
        if is_chosen:
            verdict = "chosen" + ("  (ranker's first)" if i == ranker_top else "")
        elif is_trap:
            verdict = "trap" + ("  (ranker's first)" if i == ranker_top else "")
        else:
            verdict = "ranker's first" if i == ranker_top else ""
        labels.append((txt + ("   " + verdict if verdict else ""), col if col != WHITE else SUB))
    title = f"Decision {step_i + 1}"
    sub = f"top-{v.get('k', len(verified))} checked in the pile model" if verified else ""
    lines = [(t, F_TINY, col) for t, col in labels]
    legend = [("blue: chosen", BLUE), ("orange: trap, rejected", ORANGE), ("white: checked, not chosen", SUB)]
    tw = max([d.textlength(title, font=F_BIG), d.textlength(sub, font=F_TINY)] +
             [d.textlength(t, font=f) for t, f, _ in lines] + [d.textlength(t, font=F_TINY) for t, _ in legend])
    ph = 6 + 40 + (24 if sub else 0) + 8 + 23 * len(lines) + 10 + 21 * len(legend) + 10
    _panel(d, 18, 16, 18 + tw + 28, 16 + ph)
    y = 22
    d.text((32, y), title, font=F_BIG, fill=INK); y += 40
    if sub:
        d.text((32, y), sub, font=F_TINY, fill=SUB); y += 24
    y += 8
    for t, f, col in lines:
        d.text((32, y), t, font=f, fill=col); y += 23
    y += 10
    for t, col in legend:
        d.text((32, y), t, font=F_TINY, fill=col); y += 21
    LAST_EXPO["v"] = float(exposure)
    _exposure_bar(ovl, W - 300, H - 44, 260, 14, float(exposure), float(coord.exposed_thresh))
    bgr = _bgr(Image.alpha_composite(im, ovl).convert("RGB"))
    for _ in range(DECISION_HOLD):
        _write(bgr, "decision")

_orig_update_gui = coord._update_gui
def _update_gui_spy(step, exposure, feasible, selected, scores, status="executing", **kw):
    step_ = step
    try:
        last_rank = [sl for sl in coord.step_log if sl.state == "RANK"]
        ACTION_TXT["s"] = None
        DECISION_INFO.update(actions=None, scores=None, chosen=-1, verify=None)
        if last_rank and last_rank[-1].data.get("actions"):
            acts = last_rank[-1].data["actions"]
            i = int(last_rank[-1].data.get("chosen", -1))
            DECISION_INFO.update(actions=list(acts), scores=dict(last_rank[-1].data.get("all_scores") or {}),
                                 chosen=i, verify=last_rank[-1].data.get("lookahead"))
            if 0 <= i < len(acts):
                ACTION_TXT["s"] = acts[i]
    except Exception:
        pass
    try:
        if status == "executing":
            capture_decision_view(step_, exposure, list(feasible), selected, dict(scores), status)
            ACTION_NO["n"] = int(step_) + 1
            ACTION_NO["txt"] = _action_words(ACTION_TXT["s"]) if ACTION_TXT["s"] else f"pick piece {selected}"
    except Exception as e:
        print(f"(decision view failed: {e})", flush=True)
    return _orig_update_gui(step=step, exposure=exposure, feasible=feasible, selected=selected, scores=scores, status=status, **kw)
coord._update_gui = _update_gui_spy

_orig_eval = coord.candidate_set.evaluate_all
LAST_RECS = {}
def eval_cached(cube_ids):
    recs = _orig_eval(cube_ids)
    LAST_RECS.clear(); LAST_RECS.update({int(r.cube_id): r for r in recs})
    return recs
coord.candidate_set.evaluate_all = eval_cached
_orig_fs = coord._feasible_set
def fs_spy(state):
    LAST_EXPO["v"] = float(state["human_exposure"])
    feasible, diag = _orig_fs(state)
    occ = state.get("occluder_share", {})
    top = sorted(((int(k), float(v)) for k, v in occ.items() if v > 0), key=lambda kv: -kv[1])[:6]
    if top:
        desc = []
        for k, v in top:
            r = LAST_RECS.get(k)
            if k in feasible: tg = "FEASIBLE"
            elif k in diag.get("outside_pick_tote_excluded", []): tg = "outside"
            elif k in diag.get("flatness_excluded", []): tg = "tilted"
            elif r is None: tg = "blacklisted/placed"
            else: tg = ",".join(r.failed(("C1","C2","C3","C4"))) or "?"
            desc.append(f"{k}:{v:.2f}[{tg}]")
        print(f"BLOCKERS exposure={state['human_exposure']:.3f} " + " ".join(desc), flush=True)
    return feasible, diag
coord._feasible_set = fs_spy

e0 = capture_topdown(int(os.environ.get("DEMO_BEFORE_HOLD", "45")), "Target buried in the pile", "before")
result = coord.run(target_id=scene.target_id, seed=SEED)
ACTION_NO["txt"] = None
ok_end = getattr(result, "termination_reason", "") == "exposure_threshold_reached"
e1 = capture_topdown(int(os.environ.get("DEMO_AFTER_HOLD", "75")),
                     "Target exposed: rescue complete" if ok_end else "Run ended: " + str(getattr(result, "termination_reason", "")),
                     "after", highlight_target=True)
print(f"TOPDOWN exposure before={e0:.3f} after={e1:.3f} termination={result.termination_reason}")
print("run:", result.successful_placements, result.failed_attempts, result.termination_reason, "frames", frame_i["n"])
for sl in coord.step_log:
    if sl.state == "RANK":
        sc = sl.data["all_scores"]; top = sorted(sc.items(), key=lambda kv: -kv[1])[:4]
        print(f"RANK step {sl.step}: chosen={sl.data['chosen']} scores={[(k, round(v,3)) for k,v in top]} reason={sl.data.get('reason')}")
    if sl.state == "VERIFY":
        print(f"VERIFY step {sl.step}: target={sl.data['target']} ok={sl.data['action_success']} exposure {sl.data['exposure_before']:.3f} -> {sl.data['exposure_immediately_after']:.3f}")
json.dump(rows, open(os.environ.get("DEMO_ROWS", os.path.join("output", "audit_rows.json")), "w"))
META.update(initial_exposure=result.initial_exposure, final_exposure=result.final_exposure,
            termination=result.termination_reason, steps=result.steps_completed,
            successful=result.successful_placements, failed=result.failed_attempts,
            step_log=[dict(step=s["step"], state=s["state"], **{k: v for k, v in s.items() if k in ("action", "status", "failure_code", "exposure", "exposure_before", "exposure_immediately_after", "chosen", "actions")})
                      for s in result.step_log if s["state"] in ("OBSERVE", "RANK", "EXECUTE", "VERIFY")])
json.dump(META, open(os.environ.get("DEMO_META", os.path.join(OUT_DIR, "..", f"demo_meta_{PILE}_{SEED}.json")), "w"), indent=1, default=str)
_run_json = os.environ.get("DEMO_RUN_JSON")
if _run_json:
    os.makedirs(os.path.dirname(os.path.abspath(_run_json)), exist_ok=True)
    json.dump(dict(mode="rescue(recorded)", policy=result.policy, seed=SEED, pile=PILE,
                   initial_exposure=result.initial_exposure, final_exposure=result.final_exposure,
                   steps_completed=result.steps_completed, successful_placements=result.successful_placements,
                   failed_attempts=result.failed_attempts, termination_reason=result.termination_reason,
                   nudge_mm=float(os.environ.get("DEMO_NUDGE_MM", "0")),
                   pile_settle=getattr(sim, "_settle_report", None),
                   step_log=result.step_log), open(_run_json, "w"), indent=2, default=str)

print("\nPER-STEP OUTCOME:")
for sl in coord.step_log:
    if sl.state != "EXECUTE":
        continue
    d = sl.data
    ok = d["status"].endswith("SUCCESS")
    dist = d.get("pile_disturbance", {})
    grip_hits = sorted({r["grip_body"] for r in rows if r["target"] == d["target"] and r["grip_force"] > 2})
    print(f"  step {sl.step:2d} {d.get('action', ''):>14s} {'OK     ' if ok else 'FAILED '} {d['failure_code'] if not ok else '':28s} "
          f"others moved>1cm: {len(dist.get('moved_over_1cm', []))} (max {dist.get('max_displacement_m', 0)*1000:.0f}mm) "
          f"gripper touched: {grip_hits if grip_hits else 'none'}")
print("\nANOMALIES (ext force>5N, self force>5N, gripper-vs-other force>2N, tote contact):")
tote_ids = set()
try:
    ids_ = sim.get_ids()
    for k_ in ("toteA", "toteB"):
        if k_ in ids_: tote_ids.add(int(ids_[k_]))
except Exception:
    pass
n_an = 0
for r in rows:
    flags = []
    if r["ext_force"] > 5: flags.append(f"extF={r['ext_force']:.0f}N body={r['ext_body']}")
    if r["self_force"] > 5: flags.append(f"selfF={r['self_force']:.0f}N")
    if r["grip_force"] > 2: flags.append(f"GRIPPER-TOUCH body={r['grip_body']} F={r['grip_force']:.1f}N" + (" TOTE" if r["grip_body"] in tote_ids else ""))
    if r.get("rider") is not None: flags.append(f"DOUBLE-PICK cube {r['rider']}")
    if flags:
        n_an += 1
        if n_an <= 400:
            print(f"  step {r['step']:5d} {r['phase']:18s} " + " ".join(flags))
print(f"anomaly rows: {n_an}")
