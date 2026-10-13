"""
Scene 9 — Coda (SIFT scale-space video).

Render:
    manim -pql scene9_coda.py Coda     # preview (480p15)
    manim -qh  scene9_coda.py Coda     # final   (1080p60)

Beat 1: the Hook's balloon (drawn here as a grayscale disk with two stripes and a basket) at Z
and 2Z. Each image goes through the full pipeline of common.detect_and_describe; the patch is
3 sigma on each side of the keypoint, so it is twice as large up close, and the two 4x4x8
descriptors are compared by cosine similarity. Beat 2: a textured wall seen before (near) and
now (twice as far), matched with Lowe's ratio test; then a blank wall, which yields no keypoints.
Every count and similarity on screen is computed. Set NO_LATEX=1 only to preview without LaTeX.
"""
import math
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, PATCH_K, detect_and_describe, disk_image, match_ratio,  # noqa: E402
                    patch_gradients, pixel_grid, to_scene)

# ----------------------------------------------------------------- constants
NEAR_COLOR, FAR_COLOR = YELLOW, TEAL        # Z and 2Z, as in Scenes 4-8
MATCH_COLOR, MATCH_OPACITY, MATCH_WIDTH = WHITE, 0.55, 2
PX = 40

# balloon (scene units at Z; at 2Z everything is halved)
BALLOON_SHAPE = (241, 241)                  # 6 x 6 units
BALLOON_BODY = (0.0, 0.25, 1.0, 0.8)        # disk (cx, cy, r, contrast)
BALLOON_STRIPES = [(0.4, 0.25, 0.95, 0.16, 90, -0.4), (-0.4, 0.25, 0.95, 0.16, 90, -0.4)]
BALLOON_BASKET = (0.0, -1.0, 0.22, 0.13, 0, 0.55)
BALLOON_SIGMA0, BALLOON_LAYERS = 0.125, 12
BALLOON_SCALE = 0.85                        # image units -> screen units
BALLOON_X = 3.3                             # images at x = -/+ BALLOON_X
BALLOON_Y = 0.4
PATCH_SIZE = 2.6                            # both patches shown at this size
PATCH_X, PATCH_Y = 2.9, -0.15
ARROW_GRID = 8                              # gradient arrows per side (2 x 2 samples averaged per arrow)

# wall
WALL_SHAPE = (145, 241)                     # 3.6 x 6 units
WALL_SEED = 1
WALL_GRID_X, WALL_GRID_Y = np.arange(-5.25, 5.3, 1.5), np.arange(-2.7, 2.8, 1.35)
WALL_JITTER = (0.35, 0.3)
WALL_RADII = [0.25, 0.35, 0.5]
WALL_CONTRAST = (0.35, 0.8)
WALL_BASE = 0.35                            # wall is mid-gray: brightness BG + WALL_BASE
WALL_SHADE = 0.15                           # blank wall: smooth left-to-right lighting change
WALL_SIGMA0, WALL_LAYERS = 0.06, 12
WALL_SCALE = 0.95
WALL_X, WALL_Y = 3.3, 0.45

BEAT_LEN = dict(b1=18, b2=14)               # seconds, total 32
KP_DIM_WIDTH, KP_DIM_OPACITY = 1.5, 0.5     # unmatched wall keypoints
KP_WIDTH = 3                                # matched wall keypoints
RT = dict(
    imgs=1.2, rings=1.0, squares=1.0, cut=1.6, grid=0.8, arrows=1.4, roses=1.6, size=0.6, sim=0.9,
    walls=1.0, kps=0.9, matches=1.4, count=0.6, far=1.0, swap=1.0, zero=0.7, end=1.0,
)
# Cue start times (seconds into each beat), measured from the recorded narration
# (voiceover/scene 9 beat N.wav, placed by voiceover.py 0.25 s into its beat).
CUE1 = dict(rings=1.2,       # "Knowing each keypoint's size,"
            squares=2.7,     # "SIFT cuts out a patch"
            cut=3.9,         # "...proportional to it,"
            grid=6.4,        # "splits it into a four-by-four grid,"
            arrows=8.3,      # "and records which way the brightness changes"
            roses=10.2,      # "...in each cell."
            sim=13.8,        # "Seen from near or far, the same patch gives the same"
            size=15.5)       # "...128 numbers."
CUE2 = dict(kps=1.5,         # "That fingerprint is what lets a robot"
            matches=2.6,     # "recognize a place"
            count=4.2,       # "...it has seen before,"
            far=6.0,         # "even from twice as far away."
            swap=8.3,        # "But point it at a blank wall,"
            zero=9.9)        # "with no blobs to find, and it's blind."

USE_LATEX = os.environ.get("NO_LATEX") != "1"


# ---------------------------------------------------------------- precompute
def balloon_image(scale):
    s = scale
    cx, cy, r, c = BALLOON_BODY
    stripes = [(x * s, y * s, a * s, b * s, ang, con) for x, y, a, b, ang, con in BALLOON_STRIPES]
    bx, by, ba, bb, bang, bcon = BALLOON_BASKET
    return np.clip(disk_image(BALLOON_SHAPE, PX, [(cx * s, cy * s, r * s, c)],
                              stripes + [(bx * s, by * s, ba * s, bb * s, bang, bcon)]), 0, 1)


def wall_blobs():
    rng = np.random.default_rng(WALL_SEED)
    gx, gy = np.meshgrid(WALL_GRID_X, WALL_GRID_Y)
    blobs = []
    for x, y in zip(gx.ravel(), gy.ravel()):
        blobs.append((x + rng.uniform(-WALL_JITTER[0], WALL_JITTER[0]), y + rng.uniform(-WALL_JITTER[1], WALL_JITTER[1]),
                      rng.choice(WALL_RADII), rng.choice([-1, 1]) * rng.uniform(*WALL_CONTRAST)))
    return blobs


def wall_image(scale):
    blobs = [(x * scale, y * scale, r * scale, c) for x, y, r, c in wall_blobs()]
    return np.clip(disk_image(WALL_SHAPE, PX, blobs) + WALL_BASE, 0, 1)


def blank_image():
    X, _ = pixel_grid(WALL_SHAPE, PX)
    half_w = WALL_SHAPE[1] / PX / 2
    return BG + WALL_BASE + WALL_SHADE * (X + half_w) / (2 * half_w)


def body_keypoint(kps, scale):
    """The keypoint on the balloon body: the one closest to the body center."""
    cx, cy = BALLOON_BODY[0] * scale, BALLOON_BODY[1] * scale
    return min(kps, key=lambda k: math.hypot(*(np.array(to_scene(k["row"], k["col"], BALLOON_SHAPE, PX)) - (cx, cy))))


IMG_NEAR, IMG_FAR = balloon_image(1.0), balloon_image(0.5)
KP_NEAR = body_keypoint(detect_and_describe(IMG_NEAR, PX, BALLOON_SIGMA0, BALLOON_LAYERS), 1.0)
KP_FAR = body_keypoint(detect_and_describe(IMG_FAR, PX, BALLOON_SIGMA0, BALLOON_LAYERS), 0.5)
SIMILARITY = float(KP_NEAR["desc"] @ KP_FAR["desc"])     # descriptors are unit vectors -> cosine similarity

WALL_A, WALL_B, WALL_0 = wall_image(1.0), wall_image(0.5), blank_image()
KPS_A = detect_and_describe(WALL_A, PX, WALL_SIGMA0, WALL_LAYERS)
KPS_B = detect_and_describe(WALL_B, PX, WALL_SIGMA0, WALL_LAYERS)
KPS_0 = detect_and_describe(WALL_0, PX, WALL_SIGMA0, WALL_LAYERS)
MATCHES = match_ratio([k["desc"] for k in KPS_A], [k["desc"] for k in KPS_B])


def match_is_correct(i, j):
    """Ground truth: the far view is the near view shrunk by 2 about the center."""
    a = np.array(to_scene(KPS_A[i]["row"], KPS_A[i]["col"], WALL_SHAPE, PX))
    b = np.array(to_scene(KPS_B[j]["row"], KPS_B[j]["col"], WALL_SHAPE, PX))
    return float(np.linalg.norm(a * 0.5 - b)) < 2.0 / PX


def self_check():
    # balloon: the far keypoint is one octave (3 layers, half the sigma) below the near one, same place
    assert KP_NEAR["layer"] - KP_FAR["layer"] == 3, (KP_NEAR["layer"], KP_FAR["layer"])
    assert abs(KP_NEAR["sigma"] / KP_FAR["sigma"] - 2) < 1e-9
    pn = np.array(to_scene(KP_NEAR["row"], KP_NEAR["col"], BALLOON_SHAPE, PX))
    pf = np.array(to_scene(KP_FAR["row"], KP_FAR["col"], BALLOON_SHAPE, PX))
    assert np.linalg.norm(pn * 0.5 - pf) < 1.0 / PX
    # the patch is proportional to the keypoint scale
    assert abs(KP_NEAR["half"] / KP_FAR["half"] - 2) < 1e-9
    assert abs(KP_NEAR["half"] / (PATCH_K * KP_NEAR["sigma"] * PX) - 1) < 1e-9
    # same description from near and far
    assert SIMILARITY > 0.98, SIMILARITY
    # wall: several matches, every one geometrically right
    assert len(MATCHES) >= 6, len(MATCHES)
    assert all(match_is_correct(i, j) for i, j in MATCHES)
    # blank wall: nothing survives
    assert len(KPS_0) == 0
    # layout
    assert BALLOON_X + BALLOON_SHAPE[1] / PX * BALLOON_SCALE / 2 < 6.7
    assert WALL_X + WALL_SHAPE[1] / PX * WALL_SCALE / 2 < 6.7


# ------------------------------------------------------------------ helpers
def tex(*parts, fb=None):
    """MathTex, or (NO_LATEX=1) a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)


def gray_rgb(img):
    v = np.clip(img, 0, 1)
    return np.uint8(255 * np.stack([v, v, v], axis=-1))


def show_image(img, shape, scale, center, nearest=False):
    m = ImageMobject(gray_rgb(img))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.stretch_to_fit_width(shape[1] / PX * scale)
    m.stretch_to_fit_height(shape[0] / PX * scale)
    return m.move_to(center)


def screen_point(row, col, shape, scale, center):
    x, y = to_scene(row, col, shape, PX)
    return np.array(center) + np.array([x * scale, y * scale, 0.0])


def gradient_arrows(patch, center, size, color):
    """ARROW_GRID^2 arrows: 2 x 2-averaged gradients, longest arrow = 0.9 of a cell."""
    gx, gy = patch_gradients(patch)
    n = patch.shape[0] // ARROW_GRID
    ax_ = gx.reshape(ARROW_GRID, n, ARROW_GRID, n).mean(axis=(1, 3))
    ay_ = gy.reshape(ARROW_GRID, n, ARROW_GRID, n).mean(axis=(1, 3))
    mag = np.hypot(ax_, ay_)
    cell = size / ARROW_GRID
    k = 0.9 * cell / mag.max()
    arrows = VGroup()
    for a in range(ARROW_GRID):
        for b in range(ARROW_GRID):
            if mag[a, b] < 0.08 * mag.max():
                continue
            c = np.array(center) + np.array([(b + 0.5) * cell - size / 2, size / 2 - (a + 0.5) * cell, 0])
            v = np.array([ax_[a, b], ay_[a, b], 0]) * k
            arrows.add(Arrow(c - v / 2, c + v / 2, buff=0, color=color, stroke_width=2.5,
                             max_tip_length_to_length_ratio=0.35, tip_length=0.08))
    return arrows


def roses(hists, center, size, color):
    """One 8-ray star per descriptor cell; ray length = histogram bin, longest ray = 0.45 of a cell."""
    cells, _, bins = hists.shape
    cell = size / cells
    k = 0.45 * cell / hists.max()
    g = VGroup()
    for a in range(cells):
        for b in range(cells):
            c = np.array(center) + np.array([(b + 0.5) * cell - size / 2, size / 2 - (a + 0.5) * cell, 0])
            for i in range(bins):
                ln = hists[a, b, i] * k
                if ln < 1e-3:
                    continue
                ang = 2 * np.pi * i / bins
                g.add(Line(c, c + ln * np.array([math.cos(ang), math.sin(ang), 0]), color=color, stroke_width=3))
    return g


# -------------------------------------------------------------------- scene
class Coda(Scene):
    def construct(self):
        self_check()

        def pad(t_end):
            dt = t_end - self.time
            assert dt > -0.05, f"beat overran its narration slot by {-dt:.2f} s"   # keeps voiceover timings fixed
            if dt > 1e-3:
                self.wait(dt)

        def at(t_cue):
            """Wait until t_cue seconds into the current beat (raises if the previous step ran past it)."""
            pad(t0 + t_cue)

        # ================= BEAT 1 (0-18 s) =================
        # "Knowing each keypoint's size, SIFT cuts out a patch proportional to it, splits it into a
        # four-by-four grid, and records which way the brightness changes in each cell. Seen from
        # near or far, the same patch gives the same 128 numbers."
        # cues: CUE1 (timed to the recording)
        t0 = self.time
        views = []
        for img, kp, col, x, tag in ((IMG_NEAR, KP_NEAR, NEAR_COLOR, -BALLOON_X, "Z"),
                                     (IMG_FAR, KP_FAR, FAR_COLOR, BALLOON_X, "2Z")):
            ctr = [x, BALLOON_Y, 0]
            im = show_image(img, BALLOON_SHAPE, BALLOON_SCALE, ctr)
            p = screen_point(kp["row"], kp["col"], BALLOON_SHAPE, BALLOON_SCALE, ctr)
            ring = Circle(radius=math.sqrt(2) * kp["sigma"] * BALLOON_SCALE, color=col, stroke_width=4).move_to(p)
            side = 2 * kp["half"] / PX * BALLOON_SCALE
            square = Square(side_length=side, color=col, stroke_width=3).move_to(p)
            lab = tex(tag, fb=[tag]).scale(0.8).set_color(col)
            lab.next_to(im, DOWN, buff=0.3)
            views.append(dict(im=im, ring=ring, square=square, lab=lab, kp=kp, col=col))
        self.play(*[FadeIn(v["im"]) for v in views], *[FadeIn(v["lab"]) for v in views], run_time=RT["imgs"])
        at(CUE1["rings"])
        self.play(*[Create(v["ring"]) for v in views], run_time=RT["rings"])
        at(CUE1["squares"])
        self.play(*[GrowFromCenter(v["square"]) for v in views], run_time=RT["squares"])

        # cut out: each square becomes a patch of the same on-screen size (resampled to 16 x 16)
        for v, sx in zip(views, (-PATCH_X, PATCH_X)):
            v["pc"] = [sx, PATCH_Y, 0]
            pm = ImageMobject(gray_rgb(v["kp"]["patch"]))
            pm.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            pm.stretch_to_fit_width(PATCH_SIZE).stretch_to_fit_height(PATCH_SIZE).move_to(v["pc"])
            v["patch"] = pm
            v["frame"] = Square(side_length=PATCH_SIZE, color=v["col"], stroke_width=3).move_to(v["pc"])
        at(CUE1["cut"])
        self.play(*[FadeOut(v["im"]) for v in views], *[FadeOut(v["ring"]) for v in views],
                  *[ReplacementTransform(v["square"], v["frame"]) for v in views],
                  *[FadeIn(v["patch"]) for v in views],
                  *[v["lab"].animate.next_to(v["frame"], UP, buff=0.3) for v in views], run_time=RT["cut"])

        for v in views:
            v["grid"] = VGroup(*[Line(np.array(v["pc"]) + np.array([d, -PATCH_SIZE / 2, 0]),
                                      np.array(v["pc"]) + np.array([d, PATCH_SIZE / 2, 0]))
                                 for d in np.linspace(-PATCH_SIZE / 4, PATCH_SIZE / 4, 3)]
                                + [Line(np.array(v["pc"]) + np.array([-PATCH_SIZE / 2, d, 0]),
                                        np.array(v["pc"]) + np.array([PATCH_SIZE / 2, d, 0]))
                                   for d in np.linspace(-PATCH_SIZE / 4, PATCH_SIZE / 4, 3)]
                                ).set_stroke(GRAY, width=1.5, opacity=0.7)
        at(CUE1["grid"])
        self.play(*[Create(v["grid"]) for v in views], run_time=RT["grid"])

        for v in views:
            v["arrows"] = gradient_arrows(v["kp"]["patch"], v["pc"], PATCH_SIZE, WHITE)
        at(CUE1["arrows"])
        self.play(*[v["patch"].animate.set_opacity(0.45) for v in views],
                  *[LaggedStart(*[GrowArrow(a) for a in v["arrows"]], lag_ratio=0.02) for v in views],
                  run_time=RT["arrows"])

        for v in views:
            v["roses"] = roses(v["kp"]["hists"], v["pc"], PATCH_SIZE, v["col"])
        at(CUE1["roses"])
        self.play(*[FadeTransform(v["arrows"], v["roses"]) for v in views],
                  *[v["patch"].animate.set_opacity(0.15) for v in views], run_time=RT["roses"])

        # descriptor length, read off the histograms: cells x cells x directions
        for v in views:
            cells, _, bins = v["kp"]["hists"].shape
            v["size"] = Text(f"{cells} × {cells} × {bins} = {v['kp']['desc'].size} numbers", font_size=22,
                             color=GRAY_B).next_to(v["frame"], DOWN, buff=0.25)

        sim = VGroup(Text("similarity", font_size=24), Text(f"{SIMILARITY:.3f}", font_size=30))
        sim.arrange(DOWN, buff=0.15).move_to([0, PATCH_Y, 0])
        link = DoubleArrow(views[0]["frame"].get_right() + RIGHT * 0.15, views[1]["frame"].get_left() + LEFT * 0.15,
                           buff=0, color=WHITE, stroke_width=2, tip_length=0.15).next_to(sim, DOWN, buff=0.2)
        link.set_y(PATCH_Y - 0.75)
        at(CUE1["sim"])
        self.play(FadeIn(sim), GrowFromCenter(link), run_time=RT["sim"])
        at(CUE1["size"])
        self.play(*[FadeIn(v["size"]) for v in views], run_time=RT["size"])
        pad(t0 + BEAT_LEN["b1"])

        # ================= BEAT 2 (18-32 s) =================
        # "That fingerprint is what lets a robot recognize a place it has seen before, even from twice
        # as far away. But point it at a blank wall, with no blobs to find, and it's blind."
        # cues: CUE2 (timed to the recording)
        t0 = self.time
        beat1 = Group(*[m for v in views for m in (v["patch"], v["frame"], v["lab"], v["roses"], v["grid"], v["size"])],
                      sim, link)
        ca, cb = [-WALL_X, WALL_Y, 0], [WALL_X, WALL_Y, 0]
        wa = show_image(WALL_A, WALL_SHAPE, WALL_SCALE, ca)
        wb = show_image(WALL_B, WALL_SHAPE, WALL_SCALE, cb)
        tag_a = Text("seen before", font_size=22, color=GRAY_B).next_to(wa, UP, buff=0.2)
        tag_b = Text("now, twice as far", font_size=22, color=GRAY_B).next_to(wb, UP, buff=0.2)
        self.play(FadeOut(beat1), run_time=0.5)
        self.play(FadeIn(wa), FadeIn(wb), FadeIn(tag_a), FadeIn(tag_b), run_time=RT["walls"])

        def kp_rings(kps, center, color, matched):
            g = VGroup()
            for n, k in enumerate(kps):
                on = n in matched
                g.add(Circle(radius=math.sqrt(2) * k["sigma"] * WALL_SCALE, color=color,
                             stroke_width=KP_WIDTH if on else KP_DIM_WIDTH, stroke_opacity=1.0 if on else KP_DIM_OPACITY)
                      .move_to(screen_point(k["row"], k["col"], WALL_SHAPE, WALL_SCALE, center)))
            return g

        ra = kp_rings(KPS_A, ca, NEAR_COLOR, {i for i, _ in MATCHES})
        rb = kp_rings(KPS_B, cb, FAR_COLOR, {j for _, j in MATCHES})
        at(CUE2["kps"])
        self.play(FadeIn(ra), FadeIn(rb), run_time=RT["kps"])
        links = VGroup(*[Line(screen_point(KPS_A[i]["row"], KPS_A[i]["col"], WALL_SHAPE, WALL_SCALE, ca),
                              screen_point(KPS_B[j]["row"], KPS_B[j]["col"], WALL_SHAPE, WALL_SCALE, cb),
                              color=MATCH_COLOR, stroke_width=MATCH_WIDTH, stroke_opacity=MATCH_OPACITY)
                         for i, j in MATCHES])
        at(CUE2["matches"])
        self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.15), run_time=RT["matches"])
        n_ok = sum(match_is_correct(i, j) for i, j in MATCHES)
        count = Text(f"{len(MATCHES)} matches, {n_ok} correct", font_size=26).move_to([0, -2.3, 0])
        at(CUE2["count"])
        self.play(FadeIn(count), run_time=RT["count"])
        at(CUE2["far"])
        self.play(Indicate(tag_b, color=FAR_COLOR), run_time=RT["far"])

        w0 = show_image(WALL_0, WALL_SHAPE, WALL_SCALE, [0, WALL_Y, 0])
        tag_0 = Text("blank wall", font_size=22, color=GRAY_B).next_to(w0, UP, buff=0.2)
        at(CUE2["swap"])
        self.play(FadeOut(Group(wa, wb, tag_a, tag_b, ra, rb, links, count)), FadeIn(w0), FadeIn(tag_0),
                  run_time=RT["swap"])
        zero = Text(f"{len(KPS_0)} keypoints", font_size=26).move_to([0, -2.3, 0])
        at(CUE2["zero"])
        self.play(FadeIn(zero), run_time=RT["zero"])
        pad(t0 + BEAT_LEN["b2"] - RT["end"])
        self.play(FadeOut(Group(w0, tag_0, zero)), run_time=RT["end"])
        pad(t0 + BEAT_LEN["b2"])


# ---------------------------------------------------------------------------
# Docs consulted (manim==0.21.0; https://docs.manim.community/en/stable/)
#  - ImageMobject.set_resampling_algorithm, RESAMPLING_ALGORITHMS:
#      https://docs.manim.community/en/stable/reference/manim.mobject.types.image_mobject.ImageMobject.html
#  - Arrow / DoubleArrow (max_tip_length_to_length_ratio), GrowArrow, FadeTransform, LaggedStart
