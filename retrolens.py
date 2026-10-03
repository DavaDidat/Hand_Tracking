import atexit
import math
import os
import random
import sys
import threading
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrolens.log")
_log_fh = open(LOG_PATH, "w", buffering=1)
_log_lock = threading.Lock()


def log(msg, echo=True):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        if echo:
            print(line)
        _log_fh.write(line + "\n")


_STATE = {"cap": None, "landmarker": None, "cam": None, "stop": threading.Event()}


def _cleanup():
    _STATE["stop"].set()
    cam = _STATE["cam"]
    if cam is not None:
        cam.join(timeout=2.0)
        if cam.is_alive():
            log("WARN: thread kamera masih sibuk, cap.release dilewati", echo=False)
            cv2.destroyAllWindows()
            return
        _STATE["cam"] = None
    cap = _STATE["cap"]
    if cap is not None:
        cap.release()
        _STATE["cap"] = None
    landmarker = _STATE["landmarker"]
    if landmarker is not None:
        landmarker.close()
        _STATE["landmarker"] = None
    cv2.destroyAllWindows()


def _excepthook(exc_type, exc_value, exc_tb):
    with _log_lock:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=_log_fh)
        _log_fh.write("\n")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("ERROR: program berhenti karena exception di atas (juga tertulis di retrolens.log).")
    try:
        _cleanup()
    except Exception:
        pass
    try:
        input("Tekan ENTER untuk tutup...")
    except EOFError:
        pass


class CameraThread(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True, name="camera")
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0

    def run(self):
        while not _STATE["stop"].is_set():
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
                self.ts = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.frame, self.ts


atexit.register(_cleanup)
sys.excepthook = _excepthook

WINDOW_NAME = "RETROLENS Pake Python"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
FRAME_W = 640
FRAME_H = 480

FILTER_NAMES = [
    "dual-tone",
    "thermal",
    "sketch",
    "pixelate",
    "glitch",
    "invert",
    "red-channel",
    "edge",
    "blur",
    "cartoon",
    "rainbow-wave",
]

TIP_IDS = [4, 8, 12, 16, 20]
PALM_IDS = [0, 5, 9, 13, 17]
PORTAL_IDS = [4, 8, 12, 20]
PORTAL_2D_IDS = TIP_IDS + PALM_IDS
FINGER_PIPS = [(8, 6), (12, 10), (16, 14), (20, 18)]
PINCH_TOUCH = 0.65
PINCH_RELEASE = 0.95
BANNER_COLOR = (0, 255, 255)
BANNER_COLOR_OFF = (0, 100, 255)
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

APP_FILTER = "filter"
APP_PHOTO = "photo"
PHOTO_COLOR = (0, 255, 160)
PHOTO_PANEL_W = 300
MAX_POLAROIDS = 3
CARD_W = 150
CARD_H = 180
PHOTO_SZ = 120
SLOT_TOP = [6, 150, 294]
SLOT_CX = [170, 130, 168]
COUNTDOWN_SEC = 3
FLASH_DURATION = 0.22
PEACE_HOLD = 1.0
PEACE_COOLDOWN = 1.5
PALM_HOLD = 0.6
DEV_DURATION = 1.2

CUBE_LOCAL = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
]
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def filter_none(frame, t):
    return frame


def filter_dual_tone(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    low = np.array([150, 40, 200], np.float32)
    high = np.array([30, 180, 255], np.float32)
    out = low + (high - low) * gray[..., None]
    return out.astype(np.uint8)


def filter_thermal(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filter_sketch(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def filter_pixelate(frame, t):
    h, w = frame.shape[:2]
    block = 16
    small = cv2.resize(
        frame,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_glitch(frame, t):
    rng = random.Random(int(t * 12))
    out = frame.copy()
    shift = rng.randint(5, 14)
    out[:, :, 0] = np.roll(out[:, :, 0], shift, axis=1)
    out[:, :, 2] = np.roll(out[:, :, 2], -shift, axis=1)
    h = out.shape[0]
    for _ in range(rng.randint(3, 6)):
        y = rng.randint(0, h - 1)
        bh = rng.randint(6, 36)
        dx = rng.randint(-60, 60)
        out[y:y + bh] = np.roll(out[y:y + bh], dx, axis=1)
    return out


def filter_invert(frame, t):
    return cv2.bitwise_not(frame)


def filter_red_channel(frame, t):
    out = np.zeros_like(frame)
    out[:, :, 2] = frame[:, :, 2]
    return out


def filter_edge(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 170)
    out = (frame * 0.3).astype(np.uint8)
    out[edges > 0] = (255, 255, 255)
    return out


def filter_blur(frame, t):
    return cv2.GaussianBlur(frame, (31, 31), 0)


def filter_cartoon(frame, t):
    color = cv2.bilateralFilter(frame, 9, 80, 80)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    color[edges > 0] = (0, 0, 0)
    return color


def filter_rainbow_wave(frame, t):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    w = frame.shape[1]
    shift = (np.arange(w, dtype=np.int16) + (int(t * 80) % 180)) % 180
    hsv[:, :, 0] = (hsv[:, :, 0] + shift[None, :]) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


FILTERS = {
    "none": filter_none,
    "dual-tone": filter_dual_tone,
    "thermal": filter_thermal,
    "sketch": filter_sketch,
    "pixelate": filter_pixelate,
    "glitch": filter_glitch,
    "invert": filter_invert,
    "red-channel": filter_red_channel,
    "edge": filter_edge,
    "blur": filter_blur,
    "cartoon": filter_cartoon,
    "rainbow-wave": filter_rainbow_wave,
}


def apply_filter(name, frame, t):
    return FILTERS.get(name, filter_none)(frame, t)


def to_px(landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_fist(pts):
    wrist = pts[0]
    return all(
        dist(pts[tip], wrist) < dist(pts[pip], wrist)
        for tip, pip in FINGER_PIPS
    )


def pinch_ratio(pts):
    palm = dist(pts[0], pts[9]) + 1e-6
    return min(dist(pts[4], pts[20]), dist(pts[4], pts[8])) / palm


def finger_extended(pts, tip, pip):
    wrist = pts[0]
    return dist(pts[tip], wrist) > dist(pts[pip], wrist)


def is_peace(pts):
    return (
        finger_extended(pts, 8, 6)
        and finger_extended(pts, 12, 10)
        and not finger_extended(pts, 16, 14)
        and not finger_extended(pts, 20, 18)
    )


def is_open_palm(pts):
    if not all(finger_extended(pts, tip, pip) for tip, pip in FINGER_PIPS):
        return False
    return pinch_ratio(pts) > PINCH_RELEASE


def convex_hull(points):
    arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(arr)
    return hull.reshape(-1, 2)


def fill_mask(hull, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if hull is not None and len(hull) >= 3:
        cv2.fillPoly(mask, [hull.astype(np.int32)], 255)
    return mask


def filter_in_mask(frame, mask, filter_name, t):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return frame, 0
    x0 = max(0, int(xs.min()) - 8)
    y0 = max(0, int(ys.min()) - 8)
    x1 = min(frame.shape[1], int(xs.max()) + 9)
    y1 = min(frame.shape[0], int(ys.max()) + 9)
    roi = frame[y0:y1, x0:x1]
    filtered = apply_filter(filter_name, roi, t)
    out = frame.copy()
    region = out[y0:y1, x0:x1]
    sel = mask[y0:y1, x0:x1] > 0
    region[sel] = filtered[sel]
    return out, int(mask.sum() // 255)


def pose_cube_points(pts, t):
    palm = pts[9]
    spread = dist(pts[4], pts[20])
    size = max(70.0, spread * 1.15)
    angle = math.atan2(pts[9][1] - pts[0][1], pts[9][0] - pts[0][0]) - math.pi / 2
    tilt = (pts[20][0] - pts[4][0]) / (spread + 1e-6)

    rx = 0.35
    ry = t * 0.7 + tilt * 0.6
    rz = angle

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_mat = rot_z @ rot_y @ rot_x

    verts = np.array(CUBE_LOCAL, dtype=np.float32) * size
    rotated = verts @ rot_mat.T

    f = 750.0
    persp = f / (f + rotated[:, 2] + 1e-6)
    x = palm[0] + rotated[:, 0] * persp
    y = palm[1] + rotated[:, 1] * persp
    return np.stack([x, y], axis=1).astype(np.float32)


_smooth_corners = None


def smooth_corners(corners, alpha=0.6):
    global _smooth_corners
    if corners is None:
        _smooth_corners = None
        return None
    if _smooth_corners is None or _smooth_corners.shape != corners.shape:
        _smooth_corners = corners.copy()
    else:
        _smooth_corners = alpha * corners + (1.0 - alpha) * _smooth_corners
    return _smooth_corners


def draw_hands(frame, hands):
    for pts in hands:
        for conn in HAND_CONNECTIONS:
            a = pts[conn.start]
            b = pts[conn.end]
            cv2.line(
                frame,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_pinch_indicator(frame, hands):
    for pts in hands:
        fist = is_fist(pts)
        palm = dist(pts[0], pts[9]) + 1e-6
        for a_id, b_id in ((4, 20), (4, 8)):
            a = pts[a_id]
            b = pts[b_id]
            touching = not fist and dist(a, b) / palm < PINCH_TOUCH
            color = (0, 255, 0) if touching else (150, 150, 150)
            thickness = 3 if touching else 1
            cv2.line(
                frame,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_portal(frame, mode, portal_points):
    if portal_points is None or len(portal_points) == 0:
        return
    hull = convex_hull(portal_points)
    points = np.array(portal_points, dtype=np.int32)
    if mode == "3d":
        for a, b in CUBE_EDGES:
            cv2.line(
                frame,
                (int(points[a][0]), int(points[a][1])),
                (int(points[b][0]), int(points[b][1])),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [hull.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, mode, filter_label, fps, n_render, n_fresh, pinch_str, armed, portal_px, filter_off, app_mode, n_polaroid):
    portal_str = f"{portal_px // 1000}k" if portal_px >= 1000 else str(portal_px)
    if app_mode == APP_PHOTO:
        accent = PHOTO_COLOR
        lines = [
            "APP: FOTO [Peace tahan 1 dtk / 'm' = mode filter]",
            f"FILTER: {filter_label} [Pinch = ganti filter]",
            f"BUKA TANGAN (5 jari) = SHUTTER | Tangan: {n_fresh} | FPS: {fps:.0f}",
            f"Polaroid: {n_polaroid} | Portal: {portal_str} | 's' shot 'q' quit",
        ]
        if n_render == 0:
            lines[2] = "Angkat tangan ke depan kamera untuk buka viewfinder"
    else:
        mode_label = "3D (Full 5 Jari)" if mode == "3d" else "2D (Portal)"
        accent = BANNER_COLOR
        lines = [
            f"MODE: {mode_label} [Tekan 'c' / Kepal 2 Tangan]",
            f"FILTER: {filter_label} [Sentuh Jempol-Kelingking / 'n' / 'y']",
            f"PINCH: {pinch_str} | {'ARMED' if armed else 'RELOAD'} | Tangan: {n_fresh} | Portal: {portal_str} | FPS: {fps:.0f}",
            "'n'/'y' filter  '0' off  'c' 3D  'm' foto  's' shot  'q' quit",
        ]
        if n_render == 0:
            lines[3] = "Angkat tangan ke depan kamera untuk buka portal"
    colors = [accent, accent, accent, accent]
    if filter_off:
        colors[1] = BANNER_COLOR_OFF
    y = 32
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 1, cv2.LINE_AA)
        y += 28


def draw_banner(frame, text, color, active):
    if not active or not text:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = (frame.shape[1] - tw) // 2
    y = 105
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def save_screenshot(frame):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"retrolens_{int(time.time())}.png")
    cv2.imwrite(path, frame)
    log(f"Screenshot tersimpan: {path}")


class Polaroid:
    def __init__(self, card):
        self.card = card
        self.rotation = random.uniform(-4.0, 4.0)
        self.born = time.monotonic()
        self.target_y = float(SLOT_TOP[-1])
        self.cur_y = float(FRAME_H)
        self.target_cx = float(SLOT_CX[-1])
        self.cur_cx = float(SLOT_CX[-1])
        self._cache = None

    def is_animating(self, now):
        return (
            abs(self.cur_y - self.target_y) > 0.5
            or abs(self.cur_cx - self.target_cx) > 0.5
            or (now - self.born) < DEV_DURATION
        )

    def rotated(self, now):
        prog = (now - self.born) / DEV_DURATION
        prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)
        key = round(prog, 2)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        card = self.card.astype(np.float32)
        if prog < 1.0:
            wash = np.full_like(card, (250.0, 249.0, 245.0))
            card = card * prog + wash * (1.0 - prog)
        card = np.clip(card, 0, 255).astype(np.uint8)
        h, w = card.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self.rotation, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw = int(round(h * sin + w * cos)) + 2
        nh = int(round(h * cos + w * sin)) + 2
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        rot = cv2.warpAffine(card, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        self._cache = (key, rot, mask)
        return rot, mask


def layout_polaroids(pols):
    n = len(pols)
    for j, p in enumerate(pols):
        slot = j + MAX_POLAROIDS - n
        p.target_y = float(SLOT_TOP[slot])
        p.target_cx = float(SLOT_CX[slot])


def paste_alpha(dst, src, alpha, x, y):
    hh, ww = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + ww), min(H, y + hh)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - x, y0 - y
    a = alpha[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32)[..., None] / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    s = src[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(s * a + roi * (1.0 - a), 0, 255).astype(np.uint8)


def render_panel(pols, now, dt):
    panel = np.full((FRAME_H, PHOTO_PANEL_W, 3), (26, 26, 30), np.uint8)
    step = min(1.0, dt * 10.0)
    for p in pols:
        p.cur_y += (p.target_y - p.cur_y) * step
        p.cur_cx += (p.target_cx - p.cur_cx) * step
        if abs(p.target_y - p.cur_y) < 0.5:
            p.cur_y = p.target_y
        if abs(p.target_cx - p.cur_cx) < 0.5:
            p.cur_cx = p.target_cx
        rot, mask = p.rotated(now)
        h, w = rot.shape[:2]
        x = int(round(p.cur_cx - w / 2.0))
        y = int(round(p.cur_y))
        shadow = cv2.GaussianBlur(mask, (11, 11), 0)
        shadow = (shadow.astype(np.float32) * 0.55).astype(np.uint8)
        paste_alpha(panel, np.zeros_like(rot), shadow, x + 5, y + 7)
        paste_alpha(panel, rot, mask, x, y)
    return panel


def make_card(crop, filter_name):
    photo = cv2.resize(crop, (PHOTO_SZ, PHOTO_SZ), interpolation=cv2.INTER_AREA)
    card = np.full((CARD_H, CARD_W, 3), (247, 245, 240), np.uint8)
    px = (CARD_W - PHOTO_SZ) // 2
    py = 15
    card[py:py + PHOTO_SZ, px:px + PHOTO_SZ] = photo
    cv2.rectangle(card, (px - 1, py - 1), (px + PHOTO_SZ, py + PHOTO_SZ), (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(card, time.strftime("%H:%M:%S"), (12, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 75, 75), 1, cv2.LINE_AA)
    cv2.putText(card, filter_name[:16], (12, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (135, 135, 135), 1, cv2.LINE_AA)
    return card


def build_polaroid(frame, hull, filter_name):
    arr = np.asarray(hull, dtype=np.float32)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None, "kotak tangan terlalu kecil"
    H, W = frame.shape[:2]
    side = int(min(max(x1 - x0, y1 - y0), H, W))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    xa = max(0, min(W - side, int(round(cx - side / 2.0))))
    ya = max(0, min(H - side, int(round(cy - side / 2.0))))
    crop = frame[ya:ya + side, xa:xa + side]
    card = make_card(crop, filter_name)
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, f"polaroid_{int(time.time() * 1000)}.png")
    cv2.imwrite(path, card)
    log(f"Polaroid tersimpan: {path}")
    return Polaroid(card), None


def draw_viewfinder(frame, hull_pts, label, counting):
    arr = np.asarray(hull_pts, dtype=np.float32)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    H, W = frame.shape[:2]
    xi = max(0, int(x0))
    yi = max(0, int(y0))
    xj = min(W - 1, int(x1))
    yj = min(H - 1, int(y1))
    color = (0, 0, 255) if counting else (0, 255, 255)
    L = 24
    t = 3
    cv2.line(frame, (xi, yi), (xi + L, yi), color, t, cv2.LINE_AA)
    cv2.line(frame, (xi, yi), (xi, yi + L), color, t, cv2.LINE_AA)
    cv2.line(frame, (xj, yi), (xj - L, yi), color, t, cv2.LINE_AA)
    cv2.line(frame, (xj, yi), (xj, yi + L), color, t, cv2.LINE_AA)
    cv2.line(frame, (xi, yj), (xi + L, yj), color, t, cv2.LINE_AA)
    cv2.line(frame, (xi, yj), (xi, yj - L), color, t, cv2.LINE_AA)
    cv2.line(frame, (xj, yj), (xj - L, yj), color, t, cv2.LINE_AA)
    cv2.line(frame, (xj, yj), (xj, yj - L), color, t, cv2.LINE_AA)
    cx, cy = (xi + xj) // 2, (yi + yj) // 2
    r = 14
    cv2.line(frame, (cx - r, cy), (cx - 5, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 5, cy), (cx + r, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r), (cx, cy - 5), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 5), (cx, cy + r), color, 1, cv2.LINE_AA)
    txt = f"PHOTO | {label}"
    ly = min(H - 26, yj + 24)
    cv2.putText(frame, txt, (xi, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (xi, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if not counting:
        hint = "BUKA TANGAN = SHUTTER"
        hy = max(yi - 12, 145)
        cv2.putText(frame, hint, (xi, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hint, (xi, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)


def draw_countdown(frame, remain):
    n = max(1, int(math.ceil(remain)))
    frac = remain - math.floor(remain)
    if frac <= 0:
        frac = 1.0
    scale = 1.7 + 1.1 * frac
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(n)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 6)
    H, W = frame.shape[:2]
    org = ((W - tw) // 2, (H + th) // 2 - 20)
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, (0, 255, 255), 4, cv2.LINE_AA)
    bw = 260
    bx = (W - bw) // 2
    by = org[1] + 34
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 10), (70, 70, 70), -1)
    fill = int(bw * max(0.0, min(1.0, remain / COUNTDOWN_SEC)))
    cv2.rectangle(frame, (bx, by), (bx + fill, by + 10), (0, 255, 255), -1)


def main():
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    _STATE["landmarker"] = landmarker

    cap = cv2.VideoCapture(0)
    for _ in range(3):
        if cap.isOpened():
            break
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("ERROR: Webcam gagal dibuka. Tutup aplikasi lain yang memakai kamera (misal jendela hand tracking lama), lalu jalankan ulang.")
        _cleanup()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _STATE["cap"] = cap
    for _ in range(15):
        cap.grab()

    cam = CameraThread(cap)
    cam.start()
    _STATE["cam"] = cam
    stage = {"name": "init"}

    def _heartbeat():
        while not _STATE["stop"].is_set():
            time.sleep(1.0)
            log(f"HB stage={stage['name']}", echo=False)

    threading.Thread(target=_heartbeat, daemon=True, name="heartbeat").start()

    filter_index = FILTER_NAMES.index("pixelate")
    none_active = False
    mode = "2d"
    pinch_armed = True
    last_pinch = 0.0
    last_fist = -10.0
    fps = 0.0
    prev_t = time.monotonic()
    banner_text = ""
    banner_color = BANNER_COLOR
    banner_until = 0.0
    last_fts = 0.0
    next_stall_log = 0.0
    out_last = None
    last_hands = []
    last_seen = -10.0
    stat_frames = 0
    stat_detect = 0.0
    stat_render = 0.0
    stat_show = 0.0
    stat_last = time.monotonic()
    stat_det = 0
    prev_lms = None
    frames_since_detect = 99

    app_mode = APP_FILTER
    shot = "idle"
    countdown_t0 = 0.0
    flash_t0 = -10.0
    peace_t0 = None
    peace_armed = True
    last_peace = -10.0
    palm_t0 = None
    palm_armed = True
    want_capture = False
    polaroids = []
    panel_img = None
    was_panel_anim = False

    def toggle_app_mode():
        nonlocal app_mode, last_peace, peace_t0, panel_img
        nonlocal banner_text, banner_color, banner_until
        if shot != "idle":
            banner_text = "TUNGGU COUNTDOWN SELESAI DULU"
            banner_color = BANNER_COLOR_OFF
            banner_until = time.monotonic() + 1.0
            return
        app_mode = APP_PHOTO if app_mode == APP_FILTER else APP_FILTER
        last_peace = time.monotonic()
        peace_t0 = None
        panel_img = None
        banner_text = "MODE -> FOTO" if app_mode == APP_PHOTO else "MODE -> FILTER"
        banner_color = PHOTO_COLOR if app_mode == APP_PHOTO else BANNER_COLOR
        banner_until = time.monotonic() + 1.2
        log(f"APP MODE -> {app_mode.upper()}")

    log("RETROLENS - Hand Gesture Portal Filter")
    log("Jendela yang benar : 'RETROLENS Pake Python' (HUD kuning 4 baris)")
    log("Pinch jempol-kelingking / jempol-telunjuk : ganti filter")
    log("Kepal 2 tangan / 'c'    : toggle mode 2D / 3D")
    log("Peace sign tahan 1 dtk / 'm' : toggle mode FILTER <-> FOTO")
    log("Mode foto: buka lebar tangan (5 jari) = hitung mundur 3-2-1 -> jepret")
    log("'n'/'y' filter, '0' off, 's' screenshot, 'q' keluar")

    while cap.isOpened():
        stage["name"] = "camera"
        raw, fts = cam.snapshot()
        now = time.monotonic()
        if raw is None or fts <= last_fts:
            stall = now - last_fts if last_fts > 0 else 0.0
            if last_fts > 0 and stall >= 1.0 and stall >= next_stall_log:
                log(f"CAMERA STALL {stall:.1f}s (stage terakhir={stage['name']})")
                next_stall_log = stall + 1.0
            if out_last is not None:
                cv2.imshow(WINDOW_NAME, out_last)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if last_fts > 0 and stall > 15.0:
                log("ERROR: Webcam tidak menghasilkan frame >15 detik. Pastikan kamera tidak dipakai aplikasi lain, lalu jalankan ulang.")
                break
            continue
        last_fts = fts
        next_stall_log = 0.0

        stage["name"] = "flip"
        frame = cv2.flip(raw, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else fps * 0.9 + inst * 0.1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        do_detect = prev_lms is None or dt < 0.035 or frames_since_detect >= 1
        stage["name"] = "detect" if do_detect else "reuse"
        t_stage = time.perf_counter()
        if do_detect:
            results = landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000))
            prev_lms = results.hand_landmarks
            frames_since_detect = 0
            stat_det += 1
            lms = results.hand_landmarks
        else:
            frames_since_detect += 1
            lms = prev_lms
        t_detect = time.perf_counter() - t_stage
        if t_detect > 1.0:
            log(f"SLOW detect {t_detect * 1000:.0f}ms do_detect={do_detect}")

        stage["name"] = "render"
        t_stage = time.perf_counter()
        hands = []
        if lms:
            for lm in lms:
                hands.append(to_px(lm, w, h))
        hands.sort(key=lambda pts: pts[9][0])

        if hands:
            last_hands = hands
            last_seen = now
            render_hands = hands
        elif last_hands and now - last_seen < 0.5:
            render_hands = last_hands
        else:
            render_hands = []

        pinch_min = min((pinch_ratio(pts) for pts in hands), default=None)
        pinch_on = any(
            not is_fist(pts) and pinch_ratio(pts) < PINCH_TOUCH
            for pts in hands
        )
        pinch_off = all(pinch_ratio(pts) > PINCH_RELEASE for pts in hands)
        if pinch_armed and pinch_on and now - last_pinch > 0.4:
            filter_index = (filter_index + 1) % len(FILTER_NAMES)
            none_active = False
            last_pinch = now
            pinch_armed = False
            banner_text = f"FILTER -> {FILTER_NAMES[filter_index].upper()} ({filter_index + 1}/{len(FILTER_NAMES)})"
            banner_color = BANNER_COLOR
            banner_until = now + 1.0
        if pinch_off:
            pinch_armed = True

        if len(hands) == 2 and all(is_fist(pts) for pts in hands) and now - last_fist > 1.2:
            mode = "3d" if mode == "2d" else "2d"
            last_fist = now
            smooth_corners(None)

        peace_on = any(is_peace(pts) for pts in hands)
        if not peace_on:
            peace_t0 = None
            peace_armed = True
        elif peace_armed:
            if peace_t0 is None or shot != "idle":
                peace_t0 = now
            elif now - peace_t0 >= PEACE_HOLD and now - last_peace >= PEACE_COOLDOWN:
                toggle_app_mode()
                peace_armed = False

        want_capture = False
        palm_on = any(is_open_palm(pts) for pts in hands)
        if not palm_on:
            palm_t0 = None
            palm_armed = True
        elif palm_armed and app_mode == APP_PHOTO and shot == "idle":
            if palm_t0 is None:
                palm_t0 = now
            elif now - palm_t0 >= PALM_HOLD:
                shot = "countdown"
                countdown_t0 = now
                palm_armed = False
                palm_t0 = None
                log("COUNTDOWN 3 detik dimulai")

        if shot == "countdown" and now - countdown_t0 >= COUNTDOWN_SEC:
            shot = "flash"
            flash_t0 = now
            want_capture = True

        eff_mode = "2d" if app_mode == APP_PHOTO else mode
        portal_points = None
        if eff_mode == "3d" and render_hands:
            if len(render_hands) == 2:
                corners = np.array(
                    [pts[i] for pts in render_hands for i in PORTAL_IDS],
                    dtype=np.float32,
                )
            else:
                corners = pose_cube_points(render_hands[0], now)
            portal_points = smooth_corners(corners)
        else:
            smooth_corners(None)
            if eff_mode == "2d" and render_hands:
                portal_points = [
                    pts[i] for pts in render_hands for i in PORTAL_2D_IDS
                ]

        if none_active:
            filter_name = "none"
            filter_label = "NONE (OFF)"
        else:
            filter_name = FILTER_NAMES[filter_index]
            filter_label = f"{filter_name.upper()} ({filter_index + 1}/{len(FILTER_NAMES)})"

        out = frame
        portal_px = 0
        hull = None
        if portal_points is not None and len(portal_points) >= 3:
            hull = convex_hull(portal_points)
            mask = fill_mask(hull, frame.shape)
            out, portal_px = filter_in_mask(frame, mask, filter_name, now)

        if want_capture:
            want_capture = False
            if hull is not None and len(hull) >= 3:
                polaroid, err = build_polaroid(out, hull, filter_name)
                if polaroid is not None:
                    polaroids.append(polaroid)
                    if len(polaroids) > MAX_POLAROIDS:
                        polaroids.pop(0)
                    layout_polaroids(polaroids)
                    banner_text = "FOTO MASUK POLAROID"
                    banner_color = PHOTO_COLOR
                    banner_until = now + 1.2
                    log(f"Polaroid ditambahkan (filter={filter_name}, total={len(polaroids)})")
                else:
                    banner_text = f"FOTO GAGAL - {err}"
                    banner_color = BANNER_COLOR_OFF
                    banner_until = now + 1.5
                    log(f"Capture gagal: {err}")
            else:
                banner_text = "FOTO GAGAL - tangan ilang"
                banner_color = BANNER_COLOR_OFF
                banner_until = now + 1.5
                log("Capture gagal: tidak ada hull tangan")

        if hull is not None:
            draw_portal(out, eff_mode, portal_points)

        draw_hands(out, render_hands)
        draw_pinch_indicator(out, hands)
        if app_mode == APP_PHOTO and hull is not None:
            draw_viewfinder(out, hull, filter_label, shot == "countdown")
        if shot == "countdown":
            draw_countdown(out, COUNTDOWN_SEC - (now - countdown_t0))
        pinch_str = f"{pinch_min:.2f} < {PINCH_TOUCH:.2f}" if pinch_min is not None else "-"
        draw_hud(out, mode, filter_label, fps, len(render_hands), len(hands), pinch_str, pinch_armed, portal_px, none_active, app_mode, len(polaroids))
        draw_banner(out, banner_text, banner_color, now < banner_until)
        t_render = time.perf_counter() - t_stage
        if t_render > 1.0:
            log(f"SLOW render {t_render * 1000:.0f}ms")

        stage["name"] = "show"
        t_stage = time.perf_counter()

        photo_active = app_mode == APP_PHOTO and len(polaroids) > 0
        panel_anim = photo_active and any(p.is_animating(now) for p in polaroids)
        if photo_active and (panel_anim or panel_img is None or (was_panel_anim and not panel_anim)):
            panel_img = render_panel(polaroids, now, dt)
        was_panel_anim = panel_anim
        disp = np.hstack((out, panel_img)) if (photo_active and panel_img is not None) else out

        if shot == "flash":
            ft = now - flash_t0
            if ft >= FLASH_DURATION:
                shot = "idle"
            else:
                alpha = 1.0 - ft / FLASH_DURATION
                disp = cv2.addWeighted(disp, 1.0 - alpha, np.full_like(disp, 255), alpha, 0)

        out_last = disp
        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(1) & 0xFF
        t_show = time.perf_counter() - t_stage
        if t_show > 1.0:
            log(f"SLOW show {t_show * 1000:.0f}ms")
        stage["name"] = "keys"

        stat_frames += 1
        stat_detect += t_detect
        stat_render += t_render
        stat_show += t_show
        stat_now = time.monotonic()
        if stat_now - stat_last >= 3.0 and stat_frames:
            n = stat_frames
            log(
                f"STATS fps={n / (stat_now - stat_last):.1f} "
                f"detect={stat_detect / n * 1000:.0f}ms "
                f"render={stat_render / n * 1000:.0f}ms "
                f"show={stat_show / n * 1000:.0f}ms "
                f"det={stat_det / n * 100:.0f}% "
                f"res={w}x{h} hands={len(hands)}"
            )
            stat_frames = 0
            stat_detect = stat_render = stat_show = 0.0
            stat_det = 0
            stat_last = stat_now

        if key == ord("q"):
            break
        if key == ord("n"):
            filter_index = (filter_index + 1) % len(FILTER_NAMES)
            none_active = False
            banner_text = f"FILTER -> {FILTER_NAMES[filter_index].upper()} ({filter_index + 1}/{len(FILTER_NAMES)})"
            banner_color = BANNER_COLOR
            banner_until = now + 1.0
        elif key in (ord("y"), ord("p")):
            filter_index = (filter_index - 1) % len(FILTER_NAMES)
            none_active = False
            banner_text = f"FILTER -> {FILTER_NAMES[filter_index].upper()} ({filter_index + 1}/{len(FILTER_NAMES)})"
            banner_color = BANNER_COLOR
            banner_until = now + 1.0
        elif key == ord("0"):
            none_active = not none_active
            if none_active:
                banner_text = "FILTER -> NONE (OFF)"
                banner_color = BANNER_COLOR_OFF
            else:
                banner_text = f"FILTER -> {FILTER_NAMES[filter_index].upper()} ({filter_index + 1}/{len(FILTER_NAMES)})"
                banner_color = BANNER_COLOR
            banner_until = now + 1.0
        elif key == ord("c"):
            mode = "3d" if mode == "2d" else "2d"
            smooth_corners(None)
        elif key == ord("m"):
            toggle_app_mode()
        elif key == ord("s"):
            save_screenshot(out_last if out_last is not None else out)

    _cleanup()


if __name__ == "__main__":
    main()
