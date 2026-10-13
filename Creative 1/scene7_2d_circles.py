"""
Scene 7 — 2D circles (SIFT scale-space video).

Render:
    manim -pql scene7_2d_circles.py TwoDCircles     # preview (480p15)
    manim -qh  scene7_2d_circles.py TwoDCircles     # final   (1080p60)

One object (a bright disk) seen from three distances: radii r, 2r, 4r. Every image and number
on screen is computed: the disk image and its SIFT DoG stack come from common.py, the curves
are the exact scale-normalized response at a disk center, and the dots are the DoG stack
measured at each center. Each disk peaks at sigma = r / sqrt(2), three layers (one octave)
after the next smaller one. Set NO_LATEX=1 only to preview without a LaTeX install.
"""
import math
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, CIRCLE_DISKS, CIRCLE_LAYERS, CIRCLE_PX, CIRCLE_R0, CIRCLE_SHAPE,  # noqa: E402
                    CIRCLE_SIGMA0, PEAK, SIFT_K, classify_extrema, disk_image, disk_norm_log, dog_stack,
                    find_extrema, to_pixel, to_scene)

# ----------------------------------------------------------------- constants
DISK_COLORS = [PINK, TEAL, YELLOW]          # far (4Z), middle (2Z), near (Z): same order as Scenes 4-6
Z_TAGS = ["4Z", "2Z", "Z"]
R0, DISKS, PX, SHAPE = CIRCLE_R0, CIRCLE_DISKS, CIRCLE_PX, CIRCLE_SHAPE    # shared with Scene 8 (common.py)
N_LAYERS, SIGMA0 = CIRCLE_LAYERS, CIRCLE_SIGMA0     # layer i spans [k^i, k^(i+1)] * SIGMA0
IMG_Y = 0.35                                # image center on screen
LN_K = math.log(SIFT_K)                     # DoG / ln k estimates sigma^2 * cooling rate (README D8)
NORM_PEAK = 2 * (PEAK - BG) / math.e        # theoretical peak height of every curve
NEG_GAIN = 2.5                              # surround (negative) response drawn this much brighter in BLUE_C
POS_COLOR, NEG_COLOR = WHITE, BLUE_C
AXIS_COLOR = GRAY
GHOST_OPACITY = 0.3

# response plot (log10 sigma on x)
LOGS_RANGE = (-1.0, 0.25)                   # sigma from 0.1 to ~1.78
PLOT_X_LEN, PLOT_Y_LEN, PLOT_CENTER = 9.0, 4.2, (0.3, -0.25)
SIGMA_TICKS = [0.1, 0.2, 0.5, 1.0]

BEAT_LEN = dict(b1=10, b2=16, b3=12, b4=9)   # seconds, total 47
RT = dict(
    img_in=1.2, radius=0.6, tags=0.8,
    eq=1.2, to_dog=1.2, readout=0.6, flash=0.8,
    out=0.8, axes=1.0, curves=2.0, dots=1.2, peaks=1.2, level=0.8,
    arrows=1.4, back=1.0, circles=0.8, key=0.5,
)
# Cue start times (seconds into each beat), measured from the recorded narration
# (voiceover/scene 7 beat N.wav, placed by voiceover.py 0.25 s into its beat).
CUE1 = dict(tags=3.9,        # "...photographed from three distances:"
            radius=(6.6, 7.8, 9.05))   # "...of radius r," / "then 2r," / "then 4r."
CUE2 = dict(dog=2.7,         # "...and we get one Difference-of-Gaussians layer:"
            readout=8.3,     # "Step sigma up,"
            sweep=9.6,       # "...and watch:"
            peaks=(10.9, 12.7, 13.7),  # "the small disk lights up first," / "then the middle one," / "then the big one."
            end=14.6)
CUE3 = dict(dots=4.4,        # "Each peaks in its own layer,"
            peaks=6.5,       # "at sigma equals r over root two,"
            level=9.4)       # "and all three peaks reach the same height."
CUE4 = dict(arrows=5.0,      # "...its blob peaks three layers apart."
            back=6.6,        # "Different sigma,"
            circles=7.6,     # "same feature."
            key=8.4)

USE_LATEX = os.environ.get("NO_LATEX") != "1"

# ---------------------------------------------------------------- precompute
IMAGE = disk_image(SHAPE, PX, DISKS)
DOG, SIGMAS_PX = dog_stack(IMAGE, SIGMA0 * PX, N_LAYERS)
MID_SIGMA = SIGMAS_PX * math.sqrt(SIFT_K) / PX          # scene units, one per layer
RESP = -DOG / LN_K                                       # sigma^2 * cooling rate, per pixel and layer
CENTER_PIX = [tuple(int(round(v)) for v in to_pixel(cx, cy, SHAPE, PX)) for cx, cy, _, _ in DISKS]
MEASURED = [RESP[:, r, c] for r, c in CENTER_PIX]        # one row of N_LAYERS samples per disk
PEAK_LAYER = [int(np.argmax(m)) for m in MEASURED]


def self_check():
    radii = [d[2] for d in DISKS]
    # same object at Z, 2Z, 4Z: radii double each time
    assert all(abs(radii[i + 1] / radii[i] - 2) < 1e-12 for i in range(2))
    # every disk center sits on a pixel center
    for cx, cy, _, _ in DISKS:
        r, c = to_pixel(cx, cy, SHAPE, PX)
        assert abs(r - round(r)) < 1e-9 and abs(c - round(c)) < 1e-9, (cx, cy)
    # the exact curve peaks at r / sqrt(2), always at height 2 (P - BG) / e
    for r in radii:
        s = np.exp(np.linspace(math.log(0.05), math.log(3), 20001))
        v = np.array([disk_norm_log(r, x) for x in s])
        assert abs(s[v.argmax()] / (r / math.sqrt(2)) - 1) < 1e-3, r
        assert abs(v.max() - NORM_PEAK) < 1e-6
    # measured DoG layers follow the exact curve, and peak on the layer whose mid-scale is r / sqrt(2)
    for r, m, lay in zip(radii, MEASURED, PEAK_LAYER):
        exact = np.array([disk_norm_log(r, s) for s in MID_SIGMA])
        assert np.max(np.abs(m - exact)) < 0.025, (r, np.max(np.abs(m - exact)))
        assert abs(MID_SIGMA[lay] / (r / math.sqrt(2)) - 1) < 1e-9, r
    # doubling the radius moves the peak exactly 3 layers (k^3 = 2)
    assert PEAK_LAYER[1] - PEAK_LAYER[0] == 3 and PEAK_LAYER[2] - PEAK_LAYER[1] == 3
    assert 0 < PEAK_LAYER[0] and PEAK_LAYER[2] < N_LAYERS - 1
    # SIFT's own detector finds exactly the three disk centers, at those layers (Scene 8 relies on this)
    ext = find_extrema(DOG)
    keep = sorted(e for e, t in zip(ext, classify_extrema(DOG, ext)) if t == "keep")
    assert keep == sorted((lay,) + rc for lay, rc in zip(PEAK_LAYER, CENTER_PIX)), keep
    # layout stays inside the frame
    assert SHAPE[1] / PX < 13.4 and IMG_Y + SHAPE[0] / PX / 2 < 3.0


# ------------------------------------------------------------------ helpers
def tex(*parts, fb=None):
    """MathTex, or (NO_LATEX=1) a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)


def number_mob(fn, color, font_size=30, places=2):
    if USE_LATEX:
        m = DecimalNumber(fn(), num_decimal_places=places, font_size=font_size, color=color)
        m.add_updater(lambda mob: mob.set_value(fn()))
    else:
        m = Text(f"{fn():.{places}f}", font_size=font_size - 6, color=color)
        m.add_updater(lambda mob: mob.become(Text(f"{fn():.{places}f}", font_size=font_size - 6, color=color)))
    return m


def gray_rgb(img):
    v = np.clip(img, 0, 1)
    return np.uint8(255 * np.stack([v, v, v], axis=-1))


def response_rgb(resp):
    """Normalized response -> RGB: positive (cooling) in POS_COLOR, negative (warming surround) in NEG_COLOR."""
    pos = np.clip(resp / NORM_PEAK, 0, 1)[..., None]
    neg = np.clip(-resp / NORM_PEAK * NEG_GAIN, 0, 1)[..., None]
    out = pos * np.array(color_to_rgb(POS_COLOR)) + neg * np.array(color_to_rgb(NEG_COLOR))
    return np.uint8(255 * np.clip(out, 0, 1))


def place_image(arr):
    m = ImageMobject(arr)
    m.stretch_to_fit_width(SHAPE[1] / PX)
    m.stretch_to_fit_height(SHAPE[0] / PX)
    m.move_to([0, IMG_Y, 0])
    return m


def on_screen(x, y):
    return np.array([x, y + IMG_Y, 0.0])


def lx(sigma):
    """Plot x-coordinate of sigma: log10 sigma shifted so the y-axis sits at the left end."""
    return math.log10(sigma) - LOGS_RANGE[0]


LAYER_RGB = [response_rgb(RESP[i]).astype(float) for i in range(N_LAYERS)]


def layer_image(ell):
    """Display image at fractional layer index ell (linear blend of the two nearest layers)."""
    i = min(int(math.floor(ell)), N_LAYERS - 2)
    w = float(np.clip(ell - i, 0, 1))
    return place_image(np.uint8((1 - w) * LAYER_RGB[i] + w * LAYER_RGB[i + 1]))


# -------------------------------------------------------------------- scene
class TwoDCircles(Scene):
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

        radii = [d[2] for d in DISKS]

        # ================= BEAT 1 (0-10 s) =================
        # "Back to two dimensions. Here's one object photographed from three distances:
        # a bright disk of radius r, then 2r, then 4r."
        t0 = self.time
        photo = place_image(gray_rgb(IMAGE))
        self.play(FadeIn(photo), run_time=RT["img_in"])

        radius_marks, radius_tags, z_tags = VGroup(), VGroup(), VGroup()
        for (cx, cy, r, _), col, zt in zip(DISKS, DISK_COLORS, Z_TAGS):
            c = on_screen(cx, cy)
            seg = Line(c, c + RIGHT * r, color=col, stroke_width=4)
            dot = Dot(c, radius=0.04, color=col)
            n = int(round(r / R0))
            lab = tex("r" if n == 1 else f"{n}r", fb=["r" if n == 1 else f"{n}r"]).scale(0.8).set_color(col)
            lab.next_to(on_screen(cx, cy + r), UP, buff=0.15)
            radius_marks.add(VGroup(seg, dot))
            radius_tags.add(lab)
            zl = tex(zt, fb=[zt]).scale(0.8).set_color(col)
            zl.move_to(on_screen(cx, -SHAPE[0] / PX / 2 - 0.35))
            z_tags.add(zl)
        at(CUE1["tags"])
        self.play(FadeIn(z_tags), run_time=RT["tags"])
        for m, t, cue in zip(radius_marks, radius_tags, CUE1["radius"]):
            at(cue)
            self.play(Create(m), FadeIn(t), run_time=RT["radius"])
        pad(t0 + BEAT_LEN["b1"])

        # ================= BEAT 2 (10-26 s) =================
        # "Blur, subtract the next blur level, and we get one Difference-of-Gaussians layer: how much
        # each point cooled in between. Step sigma up, and watch: the small disk lights up first,
        # then the middle one, then the big one."
        t0 = self.time
        eq = tex(r"D_\sigma", "=", r"G_{k\sigma} * I", "-", r"G_{\sigma} * I",
                 fb=["D_σ", "=", "G_kσ * I", "−", "G_σ * I"]).move_to([-1.3, 3.25, 0])
        self.play(Write(eq), FadeOut(radius_marks), FadeOut(radius_tags), run_time=RT["eq"])

        ell = ValueTracker(0.0)
        first = layer_image(0.0)
        at(CUE2["dog"])
        self.play(FadeTransform(photo, first), run_time=RT["to_dog"])
        self.remove(first)
        live = always_redraw(lambda: layer_image(ell.get_value()))
        self.add(live)

        lay_lab = Text("layer", font_size=24).move_to([4.9, 3.45, 0], aligned_edge=RIGHT)
        sig_lab = tex(r"\sigma =", fb=["σ ="]).scale(0.8).move_to([4.9, 2.95, 0], aligned_edge=RIGHT)
        lay_num = number_mob(lambda: math.floor(ell.get_value() + 1e-6) + 1, WHITE, 30, 0)
        sig_num = number_mob(lambda: MID_SIGMA[0] * SIFT_K ** ell.get_value(), WHITE, 30, 2)
        lay_num.add_updater(lambda m: m.move_to([5.05, 3.45, 0], aligned_edge=LEFT))
        sig_num.add_updater(lambda m: m.move_to([5.05, 2.95, 0], aligned_edge=LEFT))
        lay_num.update()
        sig_num.update()
        readout = VGroup(lay_lab, sig_lab, lay_num, sig_num)
        at(CUE2["readout"])
        self.play(FadeIn(readout), run_time=RT["readout"])

        # a linear sweep over the layers, split so each disk reaches its peak layer as it is named
        end = N_LAYERS - 1
        at(CUE2["sweep"])
        self.play(ell.animate.set_value(PEAK_LAYER[0]), run_time=CUE2["peaks"][0] - CUE2["sweep"], rate_func=linear)
        stops = list(CUE2["peaks"][1:]) + [CUE2["end"]]
        for k_, (lay, (cx, cy, r, _), col) in enumerate(zip(PEAK_LAYER, DISKS, DISK_COLORS)):
            nxt = PEAK_LAYER[k_ + 1] if k_ + 1 < len(PEAK_LAYER) else end
            ring = Circle(radius=r, color=col, stroke_width=4).move_to(on_screen(cx, cy))
            at(CUE2["peaks"][k_])
            # no play-level run_time: the flashes keep their own short durations inside the sweep segment,
            # capped so they finish before the next disk is named
            seg = stops[k_] - CUE2["peaks"][k_]
            self.play(ell.animate(rate_func=linear, run_time=seg).set_value(nxt),
                      Flash(on_screen(cx, cy), color=col, flash_radius=r + 0.15, line_length=0.25,
                            run_time=min(RT["flash"], seg)),
                      ShowPassingFlash(ring, time_width=0.9, run_time=min(RT["flash"] * 1.5, seg)))
        pad(t0 + BEAT_LEN["b2"])

        # ================= BEAT 3 (26-38 s) =================
        # "Track each disk's center across the layers. Each peaks in its own layer, at sigma equals
        # r over root two, and all three peaks reach the same height."
        t0 = self.time
        self.play(FadeOut(live), FadeOut(readout), FadeOut(z_tags), FadeOut(eq), run_time=RT["out"])

        ax = Axes(
            x_range=[0, LOGS_RANGE[1] - LOGS_RANGE[0], 0.25], y_range=[0, NORM_PEAK * 1.3, 0.25],
            x_length=PLOT_X_LEN, y_length=PLOT_Y_LEN, tips=False,
            axis_config={"include_ticks": False, "include_numbers": False, "stroke_width": 3, "color": AXIS_COLOR},
        ).move_to([PLOT_CENTER[0], PLOT_CENTER[1], 0])
        x_lab = tex(r"\sigma", fb=["σ"]).scale(0.7).next_to(ax.x_axis.get_end(), RIGHT, buff=0.15)
        y_lab = tex(r"\sigma^2 \cdot \text{cooling rate}", fb=["σ² · cooling rate"]).scale(0.7)
        y_lab.next_to(ax.y_axis.get_end(), UP, buff=0.15).align_to(ax.y_axis, LEFT)
        ticks = VGroup()
        for s in SIGMA_TICKS:
            p = ax.c2p(lx(s), 0)
            t_lab = Text(f"{s:g}", font_size=20, color=GRAY_B).next_to(p, DOWN, buff=0.15)
            ticks.add(VGroup(Line(p + UP * 0.06, p + DOWN * 0.06, color=AXIS_COLOR, stroke_width=2), t_lab))
        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), FadeIn(ticks), run_time=RT["axes"])

        ls = np.linspace(LOGS_RANGE[0], LOGS_RANGE[1], 300)
        curves = VGroup()
        for r, col in zip(radii, DISK_COLORS):
            pts = [ax.c2p(l - LOGS_RANGE[0], disk_norm_log(r, 10 ** l)) for l in ls]
            curves.add(VMobject(color=col, stroke_width=5).set_points_as_corners(pts))
        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.25), run_time=RT["curves"])

        dots = VGroup()
        for m, col in zip(MEASURED, DISK_COLORS):
            dots.add(VGroup(*[Dot(ax.c2p(lx(s), v), radius=0.05, color=col)
                              for s, v in zip(MID_SIGMA, m) if LOGS_RANGE[0] <= math.log10(s) <= LOGS_RANGE[1]]))
        dots_tag = Text("DoG layers", font_size=22, color=GRAY_B).move_to(ax.c2p(lx(1.0), NORM_PEAK * 1.22))
        at(CUE3["dots"])
        self.play(LaggedStart(*[FadeIn(d, lag_ratio=0.1) for d in dots], lag_ratio=0.2), FadeIn(dots_tag),
                  run_time=RT["dots"])

        peak_marks, peak_dots, peak_vals = VGroup(), VGroup(), VGroup()
        for r, col in zip(radii, DISK_COLORS):
            s_star = r / math.sqrt(2)
            top = ax.c2p(lx(s_star), disk_norm_log(r, s_star))
            peak_marks.add(DashedLine(ax.c2p(lx(s_star), 0), top, dash_length=0.08, stroke_width=2,
                                      color=col, stroke_opacity=0.8))
            peak_dots.add(Dot(top, radius=0.07, color=col))
            peak_vals.add(Text(f"{s_star:.2f}", font_size=22, color=col).next_to(top, UP, buff=0.12))
        rule = tex(r"\sigma^* = \frac{r}{\sqrt{2}}", fb=["σ* = r / √2"]).move_to([-1.3, 3.25, 0])
        at(CUE3["peaks"])
        self.play(Create(peak_marks), FadeIn(peak_dots), FadeIn(peak_vals), Write(rule), run_time=RT["peaks"])
        level = DashedLine(ax.c2p(0, NORM_PEAK), ax.c2p(LOGS_RANGE[1] - LOGS_RANGE[0], NORM_PEAK), dash_length=0.1,
                           stroke_width=2, color=WHITE, stroke_opacity=0.55)
        at(CUE3["level"])
        self.play(Create(level), run_time=RT["level"])
        pad(t0 + BEAT_LEN["b3"])

        # ================= BEAT 4 (38-47 s) =================
        # "Photograph the same object from three distances, and its blob peaks three layers apart.
        # Different sigma, same feature."
        t0 = self.time
        hops = VGroup()
        for a, b in ((0, 1), (1, 2)):
            pa, pb = peak_dots[a].get_center(), peak_dots[b].get_center()
            arc = CurvedArrow(pa + UP * 0.45, pb + UP * 0.45, angle=-PI / 3, color=WHITE, stroke_width=3,
                              tip_length=0.15)
            n = PEAK_LAYER[b] - PEAK_LAYER[a]
            lab = Text(f"+{n} layers", font_size=22).next_to(arc, UP, buff=0.08)
            hops.add(VGroup(arc, lab))
        at(CUE4["arrows"])
        self.play(FadeOut(peak_vals), FadeOut(dots_tag),
                  LaggedStart(*[Create(h[0]) for h in hops], lag_ratio=0.4),
                  LaggedStart(*[FadeIn(h[1]) for h in hops], lag_ratio=0.4), run_time=RT["arrows"])

        plot = VGroup(ax, x_lab, y_lab, ticks, curves, dots, peak_marks, peak_dots, level, hops, rule)
        photo2 = place_image(gray_rgb(IMAGE))
        at(CUE4["back"])
        self.play(FadeOut(plot), FadeIn(photo2), FadeIn(z_tags), run_time=RT["back"])

        found = VGroup()
        for lay, (r_, c_), col in zip(PEAK_LAYER, CENTER_PIX, DISK_COLORS):
            x, y = to_scene(r_, c_, SHAPE, PX)
            rad = math.sqrt(2) * MID_SIGMA[lay]          # detected scale -> radius: r = sqrt(2) sigma*
            found.add(Circle(radius=rad, color=col, stroke_width=4).move_to(on_screen(x, y)))
        at(CUE4["circles"])
        self.play(photo2.animate.set_opacity(GHOST_OPACITY),
                  LaggedStart(*[Create(c) for c in found], lag_ratio=0.3), run_time=RT["circles"])
        key = tex(r"(x,\, y,\, \sigma)", fb=["(x, y, σ)"]).move_to([-1.3, 3.25, 0])
        at(CUE4["key"])
        self.play(Write(key), run_time=RT["key"])
        pad(t0 + BEAT_LEN["b4"])


# ---------------------------------------------------------------------------
# Docs consulted (manim==0.21.0; https://docs.manim.community/en/stable/)
#  - ImageMobject (numpy array input, stretch_to_fit_*):
#      https://docs.manim.community/en/stable/reference/manim.mobject.types.image_mobject.ImageMobject.html
#  - FadeTransform, ShowPassingFlash, Flash, CurvedArrow, always_redraw, ValueTracker
