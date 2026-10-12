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

## Current active work

The current active scene is:
- `Creative 1/scene7_2d_circles.py`

The scene is intended to explain: a 2D circular blob has a radius, the Gaussian blur width should match the feature scale, and the strongest response occurs near `sigma ≈ r / sqrt(2)`.

## Status of Scene 7

This version is intentionally structured to avoid the earlier issues:
- no three-light opening with no conceptual purpose
- no distracting extra circles or clutter on the right edge
- no awkward transition where the graph appears before the original image fades
- no 5–10 second dead windows caused by long idle waits

The current scene structure is a four-beat narrative:
1. Show the real 2D image patch and its radius.
2. Show the blur family: too small / matched / too large.
3. Plot the response curve and emphasize the peak at `sigma ≈ r / sqrt(2)`.
4. Generalize to multiple blob sizes and bridge into the idea of searching over `(x, y, sigma)` for keypoints.

This is the right pacing direction and should fill roughly the intended 45 seconds without feeling padded.

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
```

Do not rely on the shell-activated `manim` alias on Windows because PowerShell execution policies can block the venv activation script.

## Next recommended task

Continue with the same principles for Scene 8 and Scene 9:
- keep the script explicit and short
- let each beat do one conceptual job
- avoid empty gaps and decorative moves
- keep the pacing aligned with the earlier completed scenes and reference videos

## References to consult

- `Creative 1/README.md` for the overall narrative plan
- `Creative 1/formatting.md` for style and layout constraints
- `Creative 1/scene2_pinhole_1d.py` and `Creative 1/scene4_blur_is_heat.py` as visual benchmarks
- the reference videos in `Creative 1/Videos_others/` to match pacing and cleanliness

## Short version for another agent

Scene 7 is now script-first, conceptually correct, and paced to avoid dead space. Keep the story as: image patch -> blur family -> response curve -> optimal sigma -> scale search for keypoints. Preserve the clean visual language from the earlier projects and do not add clutter. The active file is `Creative 1/scene7_2d_circles.py`.
