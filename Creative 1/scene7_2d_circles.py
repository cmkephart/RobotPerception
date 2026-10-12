"""
Scene 7 — 2D circles (SIFT scale-space video).

Render:
    manim -pql scene7_2d_circles.py TwoDCircles     # preview (480p15)
    manim -qh  scene7_2d_circles.py TwoDCircles     # final   (1080p60)

This scene shows the 2D analogue of the 1D scale-space detector: each circular blob
lights up most strongly when the Gaussian blur width matches its size. For a disk of radius r,
the peak happens near sigma ≈ r / sqrt(2). Set NO_LATEX=1 only to preview without a LaTeX install.
"""
import math
import os

import numpy as np
from manim import *

# ----------------------------------------------------------------- constants
CIRCLE_COLORS = [YELLOW, TEAL, PINK]
RADIUS_SET = [0.45, 0.9, 1.5]
CENTER_X = [-2.4, 0.0, 2.4]
SIGMA_SET = np.array([0.30, 0.45, 0.70, 1.00, 1.40, 1.90])
AXIS_COLOR = GRAY
PEAK_COLOR = ORANGE

BEAT_LEN = dict(b1=15, b2=15, b3=15)    # seconds, total 45
RT = dict(
    intro=1.2,
    stack=2.0,
    curve=2.0,
    peak=0.8,
    hold=1.5,
)

USE_LATEX = os.environ.get("NO_LATEX") != "1"


# ------------------------------------------------------------------ helpers
def tex(*parts, fb=None):
    """MathTex, or (NO_LATEX=1) a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)


def circle_response(sigma, radius):
    """Rough Gaussian response used to highlight the fact that sigma peaks near r / sqrt(2)."""
    sigma_peak = radius / math.sqrt(2.0)
    width = 0.18 + 0.22 * radius
    return math.exp(-0.5 * ((sigma - sigma_peak) / width) ** 2)


def gaussian_blob(x, y, cx, cy, sigma):
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def self_check():
    for radius in RADIUS_SET:
        peak_sigma = radius / math.sqrt(2.0)
        assert peak_sigma > 0.0, radius
        assert abs(peak_sigma - radius / math.sqrt(2.0)) < 1e-12
    assert np.all(np.diff(SIGMA_SET) > 0)
    assert len(RADIUS_SET) == 3
    assert len(CENTER_X) == 3


# -------------------------------------------------------------------- scene
class TwoDCircles(Scene):
    def construct(self):
        self_check()

        # BEAT 1 (0:00–0:12): "Back to two dimensions. Here is the real image patch we are looking at:
        # a circular blob with a radius of its own. It is not a point and it is not blurred by one fixed amount."
        xs = np.linspace(-4.0, 4.0, 300)
        ys = np.linspace(-2.4, 2.4, 200)
        X, Y = np.meshgrid(xs, ys)
        base = gaussian_blob(X, Y, 0.0, 0.0, RADIUS_SET[1] / 2.0)
        base = np.clip(base, 0.0, 1.0)
        img = ImageMobject(np.uint8(255 * base))
        img.scale(1.25)
        img.move_to(ORIGIN)
        self.play(FadeIn(img), run_time=RT["intro"])

        focus_blob = Circle(
            radius=RADIUS_SET[1] * 1.55,
            color=PEAK_COLOR,
            stroke_width=5,
            fill_color=PEAK_COLOR,
            fill_opacity=0.12,
        )
        focus_blob.move_to(img.get_center())
        focus_blob.set_stroke(PEAK_COLOR, width=5)
        self.play(Create(focus_blob), run_time=1.0)
        self.play(Flash(focus_blob, line_length=0.6, num_lines=12, color=PEAK_COLOR), run_time=0.8)

        radius_marker = DashedLine(
            img.get_left() + LEFT * 0.5,
            img.get_right() + RIGHT * 0.5,
            stroke_color=PEAK_COLOR,
            stroke_width=2,
            dash_length=0.09,
        )
        radius_label = tex("r", fb=["r"]).scale(0.8)
        radius_label.next_to(focus_blob, UP, buff=0.25)
        self.play(Create(radius_marker), FadeIn(radius_label), run_time=1.1)
        self.play(Indicate(focus_blob, scale_factor=1.08), run_time=0.75)

        img_label = tex("2D image", fb=["2D image"]).scale(1.0)
        img_label.next_to(img, UP, buff=0.75)
        img_label.shift(UP * 0.25)
        self.play(FadeIn(img_label), run_time=0.6)
        self.wait(3.0)

        # BEAT 2 (0:12–0:24): "Now we vary the Gaussian width. Too little blur, and the patch is still
        # too sharp. Too much blur, and the feature is washed out. There is a sweet spot, and it is the
        # size of the feature itself."
        sigma_values = [0.35, 0.9, 1.6]
        sigma_labels = VGroup(
            tex(r"\sigma_{small}", fb=["small σ"]).scale(0.7),
            tex(r"\sigma_{match}", fb=["match σ"]).scale(0.7),
            tex(r"\sigma_{large}", fb=["large σ"]).scale(0.7),
        )
        blur_family = Group()
        for sigma, label in zip(sigma_values, sigma_labels):
            patch = ImageMobject(np.uint8(255 * np.clip(gaussian_blob(X, Y, 0.0, 0.0, sigma), 0.0, 1.0)))
            patch.scale(0.45)
            blur_family.add(patch)
            label.next_to(patch, DOWN, buff=0.35)

        blur_family.arrange(RIGHT, buff=0.7)
        blur_family.move_to(ORIGIN)
        sigma_labels.arrange(RIGHT, buff=1.6)
        sigma_labels.move_to(blur_family.get_bottom() + DOWN * 0.55)

        self.play(
            FadeOut(img, run_time=1.0),
            FadeOut(img_label, run_time=1.0),
            FadeOut(focus_blob, run_time=1.0),
            FadeOut(radius_marker, run_time=1.0),
            FadeOut(radius_label, run_time=1.0),
        )
        self.play(FadeIn(blur_family), run_time=1.5)
        self.play(FadeIn(sigma_labels), run_time=1.0)
        self.play(Indicate(blur_family[1], scale_factor=1.12), run_time=1.2)
        self.wait(3.5)

        # BEAT 3 (0:24–0:36): "This gives us the response curve. At each blur width, the feature responds
        # differently. The signal peaks when the Gaussian width matches the blob radius, and the peak is
        # near sigma = r / sqrt(2)."
        self.play(FadeOut(blur_family, run_time=1.2), FadeOut(sigma_labels, run_time=1.2))

        ax = Axes(
            x_range=[0.0, 2.1, 0.5],
            y_range=[0.0, 1.2, 0.5],
            x_length=6.0,
            y_length=3.0,
            tips=False,
            axis_config={"include_ticks": False, "include_numbers": False, "stroke_width": 3},
        ).move_to([0.0, -0.2, 0.0])
        self.play(Create(ax), run_time=1.0)

        response_graphs = VGroup()
        for radius, color in zip(RADIUS_SET, CIRCLE_COLORS):
            xs_curve = np.linspace(0.05, 2.0, 220)
            ys_curve = np.array([circle_response(s, radius) for s in xs_curve])
            curve = VMobject(color=color, stroke_width=3)
            curve.set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs_curve, ys_curve)])
            response_graphs.add(curve)

        selected_idx = 1
        selected_radius = RADIUS_SET[selected_idx]
        selected_sigma = selected_radius / math.sqrt(2.0)
        selected_curve = response_graphs[selected_idx]
        selected_curve.set_stroke(color=PEAK_COLOR, width=5)
        muted_graphs = VGroup(*[curve for i, curve in enumerate(response_graphs) if i != selected_idx])
        for curve in muted_graphs:
            curve.set_stroke(opacity=0.35)

        self.play(LaggedStart(*[FadeIn(curve, run_time=0.8) for curve in response_graphs], lag_ratio=0.2), run_time=2.6)
        self.play(Indicate(selected_curve, scale_factor=1.04), run_time=0.9)

        peak_height = circle_response(selected_sigma, selected_radius)
        sigma_line = Line(
            ax.c2p(selected_sigma, 0),
            ax.c2p(selected_sigma, peak_height),
            color=PEAK_COLOR,
            stroke_width=2,
        )
        sigma_line.set_stroke(PEAK_COLOR, width=2, opacity=0.8)
        peak_dot = Dot(ax.c2p(selected_sigma, peak_height), color=PEAK_COLOR)
        self.play(Create(sigma_line), FadeIn(peak_dot), run_time=1.2)

        sigma_label = tex(r"\sigma", fb=["σ"]).scale(0.7)
        sigma_label.next_to(ax.c2p(selected_sigma, 0.0), DOWN, buff=0.26)
        self.play(FadeIn(sigma_label), run_time=0.8)

        label_peak = tex(r"\sigma \approx r/\sqrt{2}", fb=["σ ≈ r / √2"]).scale(0.78)
        label_peak.move_to([0.5, 2.1, 0.0])
        self.play(FadeIn(label_peak), run_time=0.8)
        self.wait(3.0)

        # BEAT 4 (0:36–0:45): "This is the key point: the best scale is not fixed globally. Small blobs
        # prefer small σ, large blobs prefer larger σ. In SIFT we search over position and scale, and keep
        # extrema in (x, y, σ)."
        peak_labels = VGroup(
            tex(r"\sigma_1", fb=["σ₁"]).scale(0.8),
            tex(r"\sigma_2", fb=["σ₂"]).scale(0.8),
            tex(r"\sigma_3", fb=["σ₃"]).scale(0.8),
        )
        for label, radius, color in zip(peak_labels, RADIUS_SET, CIRCLE_COLORS):
            x = radius / math.sqrt(2.0)
            label.set_color(color)
            label.next_to(ax.c2p(x, 0.0), DOWN, buff=0.28)
            label.shift(UP * 0.15)

        size_demo = VGroup()
        for radius, color in zip(RADIUS_SET, CIRCLE_COLORS):
            ring = Circle(radius=radius, color=color, stroke_width=3, fill_opacity=0.0)
            ring.move_to(ax.c2p(radius / math.sqrt(2.0), 0.2))
            size_demo.add(ring)

        keypoint_label = tex(r"(x, y, \sigma)", fb=["(x, y, σ)"]).scale(0.8)
        keypoint_label.next_to(ax.get_corner(UL), UP, buff=0.4)
        self.play(FadeIn(peak_labels), run_time=1.0)
        self.play(LaggedStart(*[Create(ring) for ring in size_demo], lag_ratio=0.25), run_time=2.2)
        self.play(FadeIn(keypoint_label), run_time=0.8)
        self.wait(2.3)

        self.play(
            FadeOut(sigma_label),
            FadeOut(label_peak),
            FadeOut(peak_dot),
            FadeOut(sigma_line),
            FadeOut(peak_labels),
            FadeOut(size_demo),
            FadeOut(keypoint_label),
            FadeOut(ax),
            FadeOut(selected_curve),
            *[FadeOut(curve) for curve in muted_graphs],
            run_time=1.2,
        )
        self.wait(0.5)
