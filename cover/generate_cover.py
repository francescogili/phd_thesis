"""
Recreation of a generative book-cover design.

Structure (measured from the reference):
 1. The entire background is ONE radial gradient centered off-canvas right
    (~(1320, 545)): near-black inside r~700, then concentric arcs of
    dark red -> red -> olive seam -> blue -> orange as distance grows.
 2. A "decomposed cube": ~10 translucent parallelogram slices (vertical
    left/right edges, top/bottom edges slanting down-right), stacked
    up-and-to-the-right, colored pale blue-gray at the back to pink at
    the front, with a soft glowing core.
 3. Film grain over EVERYTHING (background, cube, before text).
 4. Right-aligned monospace author + title on the black side.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(SCRIPT_DIR)

def out_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

def get_mono_font(size):
    """Try common monospace fonts across Windows / macOS / Linux."""
    candidates = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("Warning: no monospace TTF found, using PIL default font.")
    return ImageFont.load_default(size=size)

W, H = 1536, 1024
rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# 1. Radial-gradient background (black core + concentric color arcs)
# ----------------------------------------------------------------------
CX, CY = 1320.0, 545.0          # gradient center (right of canvas middle)

xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                     np.arange(H, dtype=np.float64))
dx, dy = xx - CX, yy - CY
dist = np.sqrt(dx ** 2 + dy ** 2)

# subtle organic wobble so the arcs aren't mathematically perfect
theta = np.arctan2(dy, dx)
wobble = (
    9.0 * np.sin(2.0 * theta + 1.1)
    + 6.0 * np.sin(3.0 * theta + 4.0)
    + 4.0 * np.sin(5.0 * theta + 2.3)
)
d = dist + wobble

# color stops measured from the reference: (distance_from_center, RGB)
stops = [
    (0,    (12, 11, 13)),     # black core
    (640,  (13, 11, 14)),
    (700,  (28, 17, 22)),     # boundary starts to glow dark red
    (762,  (118, 58, 74)),    # dark red-purple
    (840,  (177, 64, 68)),    # red peak
    (960,  (124, 76, 66)),    # red -> olive
    (1022, (110, 98, 114)),   # olive/gray seam
    (1082, (70, 90, 168)),    # blue peak
    (1142, (76, 85, 146)),    # blue outer
    (1205, (158, 106, 88)),   # blue -> orange transition
    (1310, (222, 108, 58)),   # orange-red
    (1450, (228, 160, 88)),   # yellow-orange at far corners
    (2100, (232, 172, 98)),
]
positions = np.array([s[0] for s in stops], dtype=np.float64)
colors = np.array([s[1] for s in stops], dtype=np.float64)

dc = np.clip(d, positions[0], positions[-1] - 1e-3)
idx = np.searchsorted(positions, dc, side="right") - 1
idx = np.clip(idx, 0, len(stops) - 2)
p0, p1 = positions[idx], positions[idx + 1]
c0, c1 = colors[idx], colors[idx + 1]
t = ((dc - p0) / (p1 - p0))[..., None]
bg = c0 * (1 - t) + c1 * t

bg_img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")
bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=6))
canvas = bg_img.convert("RGBA")

# ----------------------------------------------------------------------
# 2. Decomposed cube: stacked translucent parallelogram slices
#    Geometry traced from the reference:
#      slice k top-left corner: (990 + 18.3k, 525 - 9.1k)
#      width 193, top/bottom edges slant down-right (slope +0.35)
#      vertical edge height: 231 + 4.9k
# ----------------------------------------------------------------------
N = 11
SLICE_W = 195.0 #193
SLANT = 0.35 #0.35                     # dy per dx of top/bottom edges

# --- 180 rotation of the cube about its own bounding-box center -------
CUBE_CX, CUBE_CY = 1168.85, 633.325

def _rot180(x, y):
    return (2 * CUBE_CX - x, 2 * CUBE_CY - y)

def slice_poly(k):
    tlx = 990.0 + 18.3 * k
    tly = 525.0 - 9.1 * k
    h = 231.0 + 4.9 * k
    rise = SLANT * SLICE_W
    verts = [
        (tlx, tly),                        # top-left
        (tlx + SLICE_W, tly + rise),       # top-right (lower)
        (tlx + SLICE_W, tly + rise + h),   # bottom-right
        (tlx, tly + h),                    # bottom-left
    ]
    return [_rot180(vx, vy) for vx, vy in verts]

# slice tint ramp reversed: warm pink at the k=0 end -> pale blue at k=9
back_rgb = np.array([255, 98, 105])
front_rgb = np.array([150, 172, 212])
back_alpha, front_alpha = 78, 32

overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for k in range(N):
    t = k / (N - 1)
    # keep pink across most of the stack; only the last few slices cool to blue
    blend = min(1.0, max(0.0, (t - 0.55) / 0.45))
    col = back_rgb * (1 - blend) + front_rgb * blend
    alpha = int(back_alpha * (1 - blend) + front_alpha * blend)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(
        slice_poly(k),
        fill=(int(col[0]), int(col[1]), int(col[2]), alpha),
    )
    overlay = Image.alpha_composite(overlay, layer)

# soft glowing core where the pink slices overlap (kept tight enough
# that the individual slice edges stay visible through it).
# rectangle corners rotated 180 about the cube center so the glow
# stays locked to the (now rotated) front face.
glow_mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(glow_mask).rectangle([1150, 512, 1290, 692], fill=165)
glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(radius=28))
glow = Image.new("RGBA", (W, H), (255, 148, 152, 0))
glow.putalpha(glow_mask)
overlay = Image.alpha_composite(overlay, glow)

canvas = Image.alpha_composite(canvas, overlay)

# ----------------------------------------------------------------------
# 3. Film grain over everything
# ----------------------------------------------------------------------
grain = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
grain_img = Image.fromarray(np.clip(grain * 14 + 128, 0, 255).astype(np.uint8), "L")
grain_img = grain_img.filter(ImageFilter.GaussianBlur(radius=0.5))
grain_arr = (np.asarray(grain_img).astype(np.float64) - 128.0)[..., None]

arr = np.asarray(canvas.convert("RGB")).astype(np.float64)
# grain scales a bit with brightness so black stays clean like the reference
lum = arr.mean(axis=2, keepdims=True) / 255.0
arr = arr + grain_arr * (0.55 + 0.85 * lum)
canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

# ----------------------------------------------------------------------
# 4. Text (crisp, applied after the grain)
# ----------------------------------------------------------------------
draw = ImageDraw.Draw(canvas)
author_font = get_mono_font(19)
title_font = get_mono_font(21)

draw.text((1248, 113), "Francesco Gili", font=author_font,
          fill=(226, 226, 226, 255), anchor="ra")

title_lines = [
    "Bayesian and Frequentist Perspectives",
    "in Three Statistical Inverse Problems",
]
ty = 245
for line in title_lines:
    draw.text((CUBE_CX, ty), line, font=title_font,
              fill=(235, 235, 235, 255), anchor="ma")
    ty += 32

canvas.convert("RGB").save(out_path("book_cover.png"))
print("done — saved to", out_path("book_cover.png"))
