# Agent handoff: RobotPerception / Creative 1

## Project context

This repo is a Manim Community animation project about why SIFT needs scale space. The main narrative is in `Creative 1/`, and the project rules are in `CLAUDE.md` plus the long-form plan in `Creative 1/README.md`.

Important constraints:
- Use Manim Community v0.21.0 only.
- Follow `Creative 1/formatting.md` for visual language. Do not invent new colors, fonts, or layout patterns.
- Keep the script tightly synchronized to the visuals. Avoid dead space, filler, or decorative elements that do not explain the concept.
- Use a script-first workflow: lock the narration, then animate to it.
- For scene files, maintain constants at the top, `# BEAT n` comments, and a `self_check()` that is called in `construct()`.
- Shared numeric and Gaussian math belongs in `Creative 1/common.py` when appropriate.

## Current status

All nine scenes now exist. Scenes 1, 5 and 6 are rendered videos in `Creative 1/Videos_others/`; scenes 2-4 and 7-9 are code. Scenes 7-9 are timed for voiceover: the locked narration and timecodes are in the Script section of `Creative 1/README.md`, each `# BEAT` comment quotes it, and each beat is padded to `BEAT_LEN` (47 s, 31 s, 32 s). `pad()` raises if a beat overruns, so retiming a beat means changing `BEAT_LEN` and the README together.

- `scene7_2d_circles.py` (`TwoDCircles`): one disk at 4Z, 2Z, Z (radii r, 2r, 4r) -> DoG sweep, each disk lights up in its own layer -> exact sigma^2 curves with measured DoG dots, peaks at r / sqrt(2), same height -> peaks three layers apart, fitted circles, `(x, y, sigma)`. It keeps the four-beat shape of the earlier draft, but every image and curve is now computed (the earlier `circle_response` was a hand-tuned bump).
- `scene8_keypoints.py` (`Keypoints`, the one `ThreeDScene`): the scene-7 DoG layers stacked into a volume -> the real 3x3x3 block around the 2Z keypoint, 8 + 9 + 9 = 26 -> keypoint circles; a faint disk and a thin ellipse are added and fail Lowe's contrast and edge tests.
- `scene9_coda.py` (`Coda`): balloon at Z and 2Z -> 3 sigma patch -> gradients -> 4x4x8 descriptor, similarity 1.000 -> textured wall matched across a 2x distance change (8 matches, all correct) -> blank wall, 0 keypoints.

The 2D pipeline (disk images, FFT blur, DoG stack, 26-neighbor extrema, Lowe's tests, descriptors, ratio-test matching) and the shared scene 7-8 image live in `Creative 1/common.py`.

LaTeX (MiKTeX) is set up on the Windows machine, and scenes 7-9 render at `-qh` with real MathTex. Still open: recording the voiceover and editing it onto the renders.

## Important design decisions to preserve

- The scene should still feel like a clean, minimal explanation; do not reintroduce decorative visual complexity.
- The *blob* is the real object, and the *graph* is the response curve. The graph is not a random extra diagram; it is the mathematical explanation of the visual patch.
- The message should be one idea at a time: feature size -> correct blur -> strongest response -> scale search.
- Keep the image and graph transitions clean and purposeful. Fade the old visual away before introducing the next one.
- Do not lock in extra circles or labels unless they directly support the narration.

## Rendering / validation

Use this command from inside `Creative 1/`:

```powershell
.\.venv\Scripts\python.exe -m manim -pql scene7_2d_circles.py TwoDCircles
.\.venv\Scripts\python.exe -m manim -pql scene8_keypoints.py Keypoints
.\.venv\Scripts\python.exe -m manim -pql scene9_coda.py Coda
```

Do not rely on the shell-activated `manim` alias on Windows because PowerShell execution policies can block the venv activation script.

## Next recommended task

- Record the voiceover against the README timecodes and cut it onto the renders.

## References to consult

- `Creative 1/README.md` for the overall narrative plan
- `Creative 1/formatting.md` for style and layout constraints
- `Creative 1/scene2_pinhole_1d.py` and `Creative 1/scene4_blur_is_heat.py` as visual benchmarks
- the reference videos in `Creative 1/Videos_others/` to match pacing and cleanliness

## Short version for another agent

Scenes 7-9 are written, computed end to end, and timed to the locked narration in `Creative 1/README.md`. What remains is the final LaTeX render and the voiceover. Preserve the clean visual language and do not add clutter.
