"""
Scene 8 — Keypoints (SIFT scale-space video). The video's one ThreeDScene.

Render:
    manim -pql scene8_keypoints.py Keypoints     # preview (480p15)
    manim -qh  scene8_keypoints.py Keypoints     # final   (1080p60)

The DoG layers of Scene 7's circles image are stacked into an (x, y, sigma) volume. The 3x3x3
block shown is the real block of DoG samples around the middle disk's keypoint, and every
circle in Beat 2 comes from common.find_extrema / classify_extrema (Lowe's 26-neighbor test,
contrast threshold 0.03, edge ratio 10). Set NO_LATEX=1 only to preview without a LaTeX install.
"""
import math
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, CIRCLE_DISKS, CIRCLE_LAYERS, CIRCLE_PX, CIRCLE_SHAPE, CIRCLE_SIGMA0,  # noqa: E402
                    CLUTTER_DISKS, CLUTTER_ELLIPSES, LOWE_CONTRAST, PEAK, SIFT_K, classify_extrema,
                    disk_image, dog_stack, find_extrema, neighbors26, to_pixel, to_scene)

# ----------------------------------------------------------------- constants
DISK_COLORS = [PINK, TEAL, YELLOW]          # far (4Z), middle (2Z), near (Z), as in Scene 7
FOCUS = 1                                   # the keypoint we zoom into: the middle disk
PX, SHAPE, N_LAYERS = CIRCLE_PX, CIRCLE_SHAPE, CIRCLE_LAYERS
LN_K = math.log(SIFT_K)
NORM_PEAK = 2 * (PEAK - BG) / math.e        # theoretical peak of -DoG / ln k (Scene 7)
NEG_GAIN = 2.5
POS_COLOR, NEG_COLOR = WHITE, BLUE_C
AXIS_COLOR = GRAY
SAME_COLOR, OTHER_COLOR = BLUE, ORANGE      # highlight fills: same layer / layers above and below
HIGHLIGHT_OPACITY = 0.3
GHOST_OPACITY = 0.3

# 3D stack
STACK_SCALE = 0.5                           # image units -> stack units
LAYER_DZ = 0.42
CAM_STACK = dict(phi=68 * DEGREES, theta=-62 * DEGREES)
CAM_BLOCK = dict(phi=64 * DEGREES, theta=-40 * DEGREES)
CUBE_SIDE, CUBE_GAP = 0.5, 0.45          # wide gaps keep the center voxel visible
ROTATE_RATE = 0.10                          # rad/s ambient rotation while counting

# Beat 2 (flat view)
IMG_Y = 0.35
CANDIDATE_FLOOR = 0.01                      # draw candidates with |D| above this (smaller ones are flat background)

BEAT_LEN = dict(b1=18, b2=13)               # seconds, total 31
RT = dict(
    layers_in=1.0, spread=2.6, axes=1.0, marker=0.8, indicate=1.0, to_block=1.6,
    same=1.0, above=1.0, below=1.0, compare=0.9,
    flat=1.4, dots=0.6, circles=1.0, clutter=1.2, weak=1.1, edge=1.1, final=0.6,
)
# Cue start times (seconds into the beat), measured from the recorded narration
# (voiceover/scene 8 beat {1,2}.wav, placed by voiceover.py 0.25 s into each beat).
CUE1 = dict(keypoint=6.5,    # "A keypoint is a sample..."
            block=9.4,       # "...than all 26 of its neighbors:"
            same=11.7,       # "8 around it in its own layer,"
            above=13.7,      # "9 in the layer above,"
            below=15.1,      # "9 below."
            compare=16.2)
CUE2 = dict(dots=1.4,        # "Position says where the feature is;"
            circles=3.2,     # "scale says how big."
            clutter=4.9,     # "Then SIFT cleans up:"
            weak=6.9,        # "it drops weak responses, which are mostly noise,"
            edge=9.4,        # "and points on edges, which can slide along the edge."
            final=11.4)

USE_LATEX = os.environ.get("NO_LATEX") != "1"

# ---------------------------------------------------------------- precompute
IMAGE = disk_image(SHAPE, PX, CIRCLE_DISKS)
DOG, SIGMAS_PX = dog_stack(IMAGE, CIRCLE_SIGMA0 * PX, N_LAYERS)
MID_SIGMA = SIGMAS_PX * math.sqrt(SIFT_K) / PX
RESP = -DOG / LN_K                                  # sigma^2 * cooling rate (bright disks -> positive)
EXT = find_extrema(DOG)
KEYS = sorted((e for e, t in zip(EXT, classify_extrema(DOG, EXT)) if t == "keep"), key=lambda e: e[2])

CLUT = disk_image(SHAPE, PX, CIRCLE_DISKS + CLUTTER_DISKS, CLUTTER_ELLIPSES)
CDOG, _ = dog_stack(CLUT, CIRCLE_SIGMA0 * PX, N_LAYERS)
CEXT = find_extrema(CDOG)
CTAGS = classify_extrema(CDOG, CEXT)
CANDS = [(e, t) for e, t in zip(CEXT, CTAGS) if abs(CDOG[e]) > CANDIDATE_FLOOR]


def self_check():
    # one keypoint per disk, at the disk center (left to right = small to large)
    assert len(KEYS) == 3, KEYS
    for (lay, r, c), (cx, cy, rad, _) in zip(KEYS, CIRCLE_DISKS):
        x, y = to_scene(r, c, SHAPE, PX)
        assert abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9
        assert abs(math.sqrt(2) * MID_SIGMA[lay] / rad - 1) < 1e-9       # sqrt(2) sigma* = r
    # the zoomed block: center strictly beats all 26 neighbors; 8 in-layer + 9 above + 9 below
    blk = neighbors26(RESP, *KEYS[FOCUS])
    assert blk.shape == (3, 3, 3)
    mask = np.ones((3, 3, 3), bool)
    mask[1, 1, 1] = False
    assert np.all(blk[1, 1, 1] > blk[mask])
    assert mask[1].sum() == 8 and mask[0].sum() == 9 and mask[2].sum() == 9 and mask.sum() == 26
    # clutter: the same three survive; the others fail the contrast or the edge test
    kept = sorted((e for e, t in CANDS if t == "keep"), key=lambda e: e[2])
    assert kept == KEYS, kept
    tags = [t for _, t in CANDS]
    assert tags.count("weak") >= 1 and tags.count("edge") >= 1
    for e, t in CANDS:
        if t == "weak":
            assert abs(CDOG[e]) < LOWE_CONTRAST
    # every candidate outside the clean image's keypoints is caused by the clutter
    assert all(t != "keep" for e, t in CANDS if e not in KEYS)


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


def response_rgba(resp):
    """As Scene 7's colors, with alpha = response strength so the stacked layers stay see-through."""
    pos = np.clip(resp / NORM_PEAK, 0, 1)[..., None]
    neg = np.clip(-resp / NORM_PEAK * NEG_GAIN, 0, 1)[..., None]
    rgb = pos * np.array(color_to_rgb(POS_COLOR)) + neg * np.array(color_to_rgb(NEG_COLOR))
    alpha = np.clip(np.maximum(pos, neg) * 1.2, 0, 1)
    return np.uint8(255 * np.concatenate([np.clip(rgb, 0, 1), alpha], axis=-1))


def stack_point(row, col, layer):
    x, y = to_scene(row, col, SHAPE, PX)
    return np.array([x * STACK_SCALE, y * STACK_SCALE, layer_z(layer)])


def layer_z(i):
    return (i - (N_LAYERS - 1) / 2) * LAYER_DZ


def flat_image(arr):
    m = ImageMobject(arr)
    m.stretch_to_fit_width(SHAPE[1] / PX)
    m.stretch_to_fit_height(SHAPE[0] / PX)
    return m.move_to([0, IMG_Y, 0])


def flat_point(row, col):
    x, y = to_scene(row, col, SHAPE, PX)
    return np.array([x, y + IMG_Y, 0.0])


def decimals_to_separate(a, b):
    """Fewest decimals (3..6) at which a and b print differently."""
    for p in range(3, 7):
        if f"{a:.{p}f}" != f"{b:.{p}f}":
            return p
    return 6


# -------------------------------------------------------------------- scene
class Keypoints(ThreeDScene):
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

        def fixed(*mobs):
            self.add_fixed_in_frame_mobjects(*mobs)
            self.remove(*mobs)              # added for real by the animation that introduces them
            return mobs[0] if len(mobs) == 1 else mobs

        w_stack = SHAPE[1] / PX * STACK_SCALE
        h_stack = SHAPE[0] / PX * STACK_SCALE

        # ================= BEAT 1 (0-18 s) =================
        # "Stack the layers into one volume: two axes for position, a third for scale. A keypoint is a
        # sample brighter, or darker, than all 26 of its neighbors: 8 around it in its own layer,
        # 9 in the layer above, 9 below."
        # cues: stack 0-3.6, axes 3.6-4.6, then CUE1 (timed to the recording)
        t0 = self.time
        layers, frames = Group(), VGroup()
        for i in range(N_LAYERS):
            im = ImageMobject(response_rgba(RESP[i]))
            im.stretch_to_fit_width(w_stack)
            im.stretch_to_fit_height(h_stack)
            layers.add(im)
            frames.add(Rectangle(width=w_stack, height=h_stack, stroke_color=AXIS_COLOR, stroke_width=1.5))
        self.play(FadeIn(layers), FadeIn(frames), run_time=RT["layers_in"])
        self.move_camera(**CAM_STACK, run_time=RT["spread"],
                         added_anims=[m.animate.shift(OUT * layer_z(i)) for i, m in enumerate(layers)]
                         + [f.animate.shift(OUT * layer_z(i)) for i, f in enumerate(frames)])

        corner = np.array([-w_stack / 2, -h_stack / 2, layer_z(0)])
        ax_x = Line(corner, corner + RIGHT * w_stack, color=AXIS_COLOR, stroke_width=3)
        ax_y = Line(corner, corner + UP * h_stack, color=AXIS_COLOR, stroke_width=3)
        ax_s = Line(corner, corner + OUT * (layer_z(N_LAYERS - 1) - layer_z(0) + 0.3), color=AXIS_COLOR,
                    stroke_width=3)
        lab_x = tex("x", fb=["x"]).scale(0.8).move_to(ax_x.get_end() + RIGHT * 0.3)
        lab_y = tex("y", fb=["y"]).scale(0.8).move_to(ax_y.get_end() + LEFT * 0.3)
        lab_s = tex(r"\sigma", fb=["σ"]).scale(0.8).move_to(ax_s.get_end() + OUT * 0.3)
        self.add_fixed_orientation_mobjects(lab_x, lab_y, lab_s)
        self.remove(lab_x, lab_y, lab_s)
        self.play(Create(ax_x), Create(ax_y), Create(ax_s), FadeIn(lab_x), FadeIn(lab_y), FadeIn(lab_s),
                  run_time=RT["axes"])

        key = KEYS[FOCUS]
        col = DISK_COLORS[FOCUS]
        marker = Cube(side_length=0.16, fill_color=col, fill_opacity=1.0, stroke_width=0).move_to(stack_point(key[1], key[2], key[0]))
        at(CUE1["keypoint"])
        self.play(FadeIn(marker, scale=3), run_time=RT["marker"])
        self.play(Indicate(marker, scale_factor=1.8, color=col), run_time=RT["indicate"])

        # the real 3x3x3 block around the keypoint; fill opacity shows each sample relative to the center
        blk = neighbors26(RESP, *key)
        lo, hi = blk.min(), blk[1, 1, 1]
        step = CUBE_SIDE + CUBE_GAP
        cubes = {}
        block = VGroup()
        for dl in (-1, 0, 1):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    v = blk[dl + 1, dr + 1, dc + 1]
                    rel = (v - lo) / (hi - lo)
                    pos = np.array([dc * step, -dr * step, dl * step])     # rows grow downward in the image
                    if dl == dr == dc == 0:
                        cb = Cube(side_length=CUBE_SIDE, fill_color=col, fill_opacity=0.95, stroke_width=0)
                    else:
                        cb = Cube(side_length=CUBE_SIDE, fill_color=WHITE, fill_opacity=0.08 + 0.3 * rel,
                                  stroke_color=WHITE, stroke_width=1, stroke_opacity=0.5)
                    cb.move_to(pos)
                    cubes[(dl, dr, dc)] = cb
                    block.add(cb)
        center = cubes[(0, 0, 0)]
        others = VGroup(*[c for k_, c in cubes.items() if k_ != (0, 0, 0)])
        stack_all = Group(layers, frames, ax_x, ax_y, ax_s, lab_x, lab_y, lab_s)
        at(CUE1["block"])
        self.move_camera(**CAM_BLOCK, run_time=RT["to_block"],
                         added_anims=[FadeOut(stack_all), ReplacementTransform(marker, center),
                                      FadeIn(others)])
        self.begin_ambient_camera_rotation(rate=ROTATE_RATE)

        same = VGroup(*[c for (dl, dr, dc), c in cubes.items() if dl == 0 and (dr, dc) != (0, 0)])
        above = VGroup(*[c for (dl, _, _), c in cubes.items() if dl == 1])
        below = VGroup(*[c for (dl, _, _), c in cubes.items() if dl == -1])
        n_same, n_above, n_below = len(same), len(above), len(below)
        # tally, one row per group: number (in the group's color) + small tag
        rows = []
        for n, word, c in ((n_same, "around it", SAME_COLOR), (n_above, "above", OTHER_COLOR),
                           (n_below, "below", OTHER_COLOR), (n_same + n_above + n_below, "neighbors", WHITE)):
            num = tex(f"{n}", fb=[f"{n}"]).set_color(c)
            rows.append(VGroup(num, Text(word, font_size=22, color=GRAY_B)).arrange(RIGHT, buff=0.25))
        count = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UL, buff=0.6)
        rule = Line(LEFT, RIGHT, stroke_width=2, color=GRAY).set_width(count.width)
        rule.next_to(rows[2], DOWN, buff=0.12, aligned_edge=LEFT)
        rows[3].next_to(rule, DOWN, buff=0.12, aligned_edge=LEFT)
        fixed(count, rule)

        def light(group, color):
            return [c.animate.set_fill(color, opacity=HIGHLIGHT_OPACITY) for c in group]

        at(CUE1["same"])
        self.play(*light(same, SAME_COLOR), FadeIn(count[0]), run_time=RT["same"])
        at(CUE1["above"])
        self.play(*light(above, OTHER_COLOR), FadeIn(count[1]), run_time=RT["above"])
        at(CUE1["below"])
        self.play(*light(below, OTHER_COLOR), FadeIn(count[2]), Create(rule), FadeIn(count[3]),
                  run_time=RT["below"])

        best = float(np.max(np.delete(blk.ravel(), 13)))
        p = decimals_to_separate(hi, best)
        cmp_ = VGroup(
            Text(f"center   {hi:.{p}f}", font_size=24, t2c={f"{hi:.{p}f}": col}),
            Text(f"best of 26   {best:.{p}f}", font_size=24),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.2).to_corner(UR, buff=0.6)
        fixed(cmp_)
        at(CUE1["compare"])
        self.play(FadeIn(cmp_), run_time=RT["compare"])
        pad(t0 + BEAT_LEN["b1"])

        # ================= BEAT 2 (18-31 s) =================
        # "Position says where the feature is; scale says how big. Then SIFT cleans up: it drops weak
        # responses, which are mostly noise, and points on edges, which can slide along the edge."
        # cues: CUE2 (timed to the recording)
        t0 = self.time
        photo = flat_image(gray_rgb(IMAGE)).set_opacity(0.6)
        self.play(FadeOut(block), FadeOut(count), FadeOut(rule), FadeOut(cmp_), run_time=RT["flat"] / 2)
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)        # empty frame: snap back to the flat view
        self.play(FadeIn(photo), run_time=RT["flat"] / 2)

        dots = VGroup(*[Dot(flat_point(r, c), radius=0.06, color=cl) for (_, r, c), cl in zip(KEYS, DISK_COLORS)])
        at(CUE2["dots"])
        self.play(FadeIn(dots), run_time=RT["dots"])
        rings = VGroup(*[Circle(radius=math.sqrt(2) * MID_SIGMA[l], color=cl, stroke_width=4).move_to(flat_point(r, c))
                         for (l, r, c), cl in zip(KEYS, DISK_COLORS)])
        at(CUE2["circles"])
        self.play(*[GrowFromCenter(g) for g in rings], run_time=RT["circles"])

        # clutter arrives: every extremum becomes a candidate circle
        photo2 = flat_image(gray_rgb(CLUT)).set_opacity(0.6)
        cand = {"weak": VGroup(), "edge": VGroup()}
        for (l, r, c), t in CANDS:
            if t == "keep":
                continue
            cand[t].add(Circle(radius=math.sqrt(2) * MID_SIGMA[l], color=WHITE, stroke_width=2.5)
                        .move_to(flat_point(r, c)))
        at(CUE2["clutter"])
        self.play(FadeTransform(photo, photo2), FadeIn(cand["weak"]), FadeIn(cand["edge"]), run_time=RT["clutter"])

        def tag_for(group, word):
            far = max(group, key=lambda m: m.get_center()[1] if word == "edge" else -m.get_center()[1])
            return Text(word, font_size=22, color=GRAY_B).next_to(far, UP if word == "edge" else DOWN, buff=0.12)

        weak_tag = tag_for(cand["weak"], "weak")
        at(CUE2["weak"])
        self.play(cand["weak"].animate.set_stroke(GRAY, opacity=GHOST_OPACITY), FadeIn(weak_tag), run_time=RT["weak"] / 2)
        self.play(FadeOut(cand["weak"]), FadeOut(weak_tag), run_time=RT["weak"] / 2)
        at(CUE2["edge"])
        edge_tag = tag_for(cand["edge"], "edge")
        self.play(cand["edge"].animate.set_stroke(GRAY, opacity=GHOST_OPACITY), FadeIn(edge_tag), run_time=RT["edge"] / 2)
        self.play(FadeOut(cand["edge"]), FadeOut(edge_tag), run_time=RT["edge"] / 2)
        at(CUE2["final"])
        self.play(*[Indicate(g, scale_factor=1.08, color=cl) for g, cl in zip(rings, DISK_COLORS)],
                  run_time=RT["final"])
        pad(t0 + BEAT_LEN["b2"])


# ---------------------------------------------------------------------------
# Docs consulted (manim==0.21.0; https://docs.manim.community/en/stable/)
#  - ThreeDScene (move_camera, begin_ambient_camera_rotation, add_fixed_in_frame_mobjects,
#    add_fixed_orientation_mobjects):
#      https://docs.manim.community/en/stable/reference/manim.scene.three_d_scene.ThreeDScene.html
#  - Cube: https://docs.manim.community/en/stable/reference/manim.mobject.three_d.polyhedra.html
#  - ImageMobject (RGBA arrays): manim.mobject.types.image_mobject.ImageMobject
