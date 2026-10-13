# Voiceover recording sheet: scenes 7–9

Record one clip per beat (8 clips) and save each one in this folder under the file name given in its heading. Spaces or "part" also work (`scene 7 beat 2.wav`, `Scene7-Part2.m4a`); if two files match the same beat, the newest one is used. WAV, M4A (phone voice memos) and MP3 all work. Then, from `Creative 1/`:

```powershell
.\.venv\Scripts\python.exe voiceover.py
```

The script trims the silence at both ends, matches the loudness of the scenes 2–4 narration, and starts each clip 0.25 s into its beat. It writes `media/videos/<scene>/1080p60/<Scene>_vo.mp4`. It also prints how long each clip is against the time available. If a clip says **TOO LONG**, re-record it a little faster, or ask for that beat to be lengthened.

**Tips**
- Pause a second before and after each line. Silence is trimmed automatically, so extra takes inside one file are *not*: one clean take per file.
- Read at a relaxed, explaining pace. Each beat has 1.5–3.5 s of slack at about 140 words per minute.
- Use the same room, mic and distance for all 8 clips so they sound alike.
- Say "σ" as "sigma", "r/√2" as "r over root two", "2r" as "two r", and "128" as "one hundred twenty-eight".

---

## Scene 7: 2D circles (0:47)

### `scene7_beat1` (0:00–0:10, 9.6 s available)
Back to two dimensions. Here's one object photographed from three distances: a bright disk of radius r, then 2r, then 4r.

### `scene7_beat2` (0:10–0:26, 15.6 s)
Blur, subtract the next blur level, and we get one Difference-of-Gaussians layer: how much each point cooled in between. Step sigma up, and watch: the small disk lights up first, then the middle one, then the big one.

### `scene7_beat3` (0:26–0:38, 11.6 s)
Track each disk's center across the layers. Each peaks in its own layer, at sigma equals r over root two, and all three peaks reach the same height.

### `scene7_beat4` (0:38–0:47, 8.6 s)
Photograph the same object from three distances, and its blob peaks three layers apart. Different sigma, same feature.

## Scene 8: Keypoints (0:31)

### `scene8_beat1` (0:00–0:18, 17.6 s)
Stack the layers into one volume: two axes for position, a third for scale. A keypoint is a sample brighter, or darker, than all 26 of its neighbors: 8 around it in its own layer, 9 in the layer above, 9 below.

### `scene8_beat2` (0:18–0:31, 12.6 s)
Position says where the feature is; scale says how big. Then SIFT cleans up: it drops weak responses, which are mostly noise, and points on edges, which can slide along the edge.

## Scene 9: Coda (0:32)

### `scene9_beat1` (0:00–0:18, 17.6 s)
Knowing each keypoint's size, SIFT cuts out a patch proportional to it, splits it into a four-by-four grid, and records which way the brightness changes in each cell. Seen from near or far, the same patch gives the same 128 numbers.

### `scene9_beat2` (0:18–0:32, 13.6 s)
That fingerprint is what lets a robot recognize a place it has seen before, even from twice as far away. But point it at a blank wall, with no blobs to find, and it's blind.
