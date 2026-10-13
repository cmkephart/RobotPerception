"""
Put recorded narration onto the rendered scenes 7-9.

Record one clip per beat and save it as  voiceover/scene{N}_beat{B}.{wav|m4a|mp3}  (e.g. scene7_beat2.m4a);
the line for each beat is in README.md (Script) and in the scene file's `# BEAT` comment. Then run:

    .\\.venv\\Scripts\\python.exe voiceover.py            # all three scenes (needs the -qh renders)
    .\\.venv\\Scripts\\python.exe voiceover.py 8          # one scene
    .\\.venv\\Scripts\\python.exe voiceover.py 7 --ql     # use the 480p15 preview render instead
    .\\.venv\\Scripts\\python.exe voiceover.py --movie    # also join scenes 7-9 into one movie

Each clip is trimmed of leading/trailing silence, leveled to the loudness of the scenes 2-4 narration,
and placed LEAD seconds after its beat starts (beat lengths come from the scene file's BEAT_LEN).
A clip that does not fit its beat is reported, never cut. Output: media/videos/<scene>/<quality>/<Class>_vo.mp4
and a matching .wav. The video stream is copied, not re-encoded.
"""
import argparse
import glob
import importlib
import os
import re
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCENES = {7: ("scene7_2d_circles", "TwoDCircles"), 8: ("scene8_keypoints", "Keypoints"), 9: ("scene9_coda", "Coda")}
CLIP_DIR = os.path.join(HERE, "voiceover")
REFERENCE = os.path.join(HERE, "Videos_others", "02-03-04.Creative1.HDRender.mp4")   # loudness reference
RATE = 48000
LEAD = 0.25                 # seconds of quiet at the start of each beat before the line starts
TAIL_GAP = 0.15             # a clip must end at least this long before the next beat starts
TRIM_DB = -40.0             # silence threshold, relative to the clip's peak
PAD_TRIM = 0.08             # keep this much audio around the detected speech
FALLBACK_RMS = 0.022        # speech RMS used if the reference video is missing
PEAK_LIMIT = 0.89           # never let a leveled clip peak above this (about -1 dBFS)


def read_audio(path):
    """Mono float32 samples at RATE."""
    c = av.open(path)
    rs = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    out = []
    for frame in c.decode(c.streams.audio[0]):
        for f in rs.resample(frame):
            out.append(f.to_ndarray().reshape(-1))
    for f in rs.resample(None):
        out.append(f.to_ndarray().reshape(-1))
    c.close()
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def speech_rms(x, win=0.05):
    """RMS over the windows that contain speech (louder than TRIM_DB below the peak)."""
    n = int(win * RATE)
    if len(x) < n:
        return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    w = x[: len(x) // n * n].reshape(-1, n)
    r = np.sqrt(np.mean(w**2, axis=1))
    loud = r > r.max() * 10 ** (TRIM_DB / 20)
    return float(np.sqrt(np.mean(r[loud] ** 2)))


def reference_rms():
    if not os.path.exists(REFERENCE):
        return FALLBACK_RMS
    return speech_rms(read_audio(REFERENCE))


def trim(x):
    env = np.abs(x)
    on = np.nonzero(env > env.max() * 10 ** (TRIM_DB / 20))[0]
    if len(on) == 0:
        return x[:0]
    a = max(on[0] - int(PAD_TRIM * RATE), 0)
    b = min(on[-1] + int(PAD_TRIM * RATE), len(x))
    return x[a:b]


AUDIO_EXT = (".wav", ".m4a", ".mp3", ".flac", ".ogg")


def find_clip(scene, beat, clip_dir=CLIP_DIR):
    """The clip for scene/beat: 'scene7_beat2.wav', 'scene 7 beat 2.wav', 'Scene7-Part2.m4a' all match.
    If several match, the most recently modified wins."""
    pat = re.compile(rf"^scene[\s_-]*{scene}[\s_-]*(beat|part)[\s_-]*{beat}$", re.IGNORECASE)
    hits = [p for p in glob.glob(os.path.join(clip_dir, "*"))
            if os.path.splitext(p)[1].lower() in AUDIO_EXT and pat.match(os.path.splitext(os.path.basename(p))[0])]
    return max(hits, key=os.path.getmtime) if hits else None


def build_track(scene, target_rms, clip_dir=CLIP_DIR):
    mod = importlib.import_module(SCENES[scene][0])
    lens = list(mod.BEAT_LEN.values())
    total = sum(lens)
    track = np.zeros(int(round(total * RATE)), np.float32)
    start = 0.0
    report, ok = [], True
    for b, ln in enumerate(lens, 1):
        path = find_clip(scene, b, clip_dir)
        if path is None:
            report.append(f"  beat {b} ({start:5.1f}-{start + ln:5.1f} s): no clip (scene{scene}_beat{b}.*) - left silent")
            ok = False
        else:
            x = trim(read_audio(path))
            gain = target_rms / max(speech_rms(x), 1e-9)
            gain = min(gain, PEAK_LIMIT / max(float(np.abs(x).max()), 1e-9))
            x = x * gain
            dur = len(x) / RATE
            room = ln - LEAD - TAIL_GAP
            status = "ok" if dur <= room else f"TOO LONG by {dur - room:.2f} s - read faster or ask to lengthen this beat"
            ok &= dur <= room
            report.append(f"  beat {b} ({start:5.1f}-{start + ln:5.1f} s): {os.path.basename(path)}  {dur:5.2f} s "
                          f"of {room:5.2f} s available  -> {status}")
            i0 = int(round((start + LEAD) * RATE))
            seg = x[: max(len(track) - i0, 0)]
            track[i0:i0 + len(seg)] += seg
        start += ln
    return track, total, report, ok


def write_wav(path, x):
    c = av.open(path, "w")
    s = c.add_stream("pcm_s16le", rate=RATE, layout="mono")
    f = av.AudioFrame.from_ndarray(np.clip(x, -1, 1).reshape(1, -1).astype(np.float32), format="flt", layout="mono")
    f.sample_rate = RATE
    for p in s.encode(f):
        c.mux(p)
    for p in s.encode(None):
        c.mux(p)
    c.close()


def mux(video_path, track, out_path):
    """Copy the video stream and add `track` as AAC audio."""
    src = av.open(video_path)
    vin = src.streams.video[0]
    dst = av.open(out_path, "w")
    vout = dst.add_stream_from_template(vin)
    aout = dst.add_stream("aac", rate=RATE, layout="mono")
    aout.bit_rate = 192000
    for pkt in src.demux(vin):
        if pkt.dts is None:
            continue
        pkt.stream = vout
        dst.mux(pkt)
    n = 1024
    for i in range(0, len(track), n):
        chunk = track[i:i + n]
        f = av.AudioFrame.from_ndarray(chunk.reshape(1, -1).astype(np.float32), format="fltp", layout="mono")
        f.sample_rate = RATE
        f.pts = i
        for p in aout.encode(f):
            dst.mux(p)
    for p in aout.encode(None):
        dst.mux(p)
    dst.close()
    src.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", type=int, default=sorted(SCENES))
    ap.add_argument("--ql", action="store_true", help="use the 480p15 preview render")
    ap.add_argument("--clips", default=CLIP_DIR, help="folder holding the scene{N}_beat{B} clips")
    ap.add_argument("--movie", action="store_true", help="also join scenes 7-9 into media/videos/Scenes7-9_<quality>.mp4")
    args = ap.parse_args()
    quality = "480p15" if args.ql else "1080p60"
    target = reference_rms()
    print(f"target speech loudness (RMS) {target:.4f}, from {'scenes 2-4' if os.path.exists(REFERENCE) else 'fallback'}")
    all_ok = True
    for sc in args.scenes:
        file_, cls = SCENES[sc]
        video = os.path.join(HERE, "media", "videos", file_, quality, f"{cls}.mp4")
        if not os.path.exists(video):
            print(f"scene {sc}: render not found ({video}); render it first")
            all_ok = False
            continue
        track, total, report, ok = build_track(sc, target, args.clips)
        vdur = float(av.open(video).duration / 1e6)
        print(f"scene {sc} ({cls}, {total:.0f} s; video {vdur:.2f} s)")
        print("\n".join(report))
        out = os.path.join(os.path.dirname(video), f"{cls}_vo.mp4")
        write_wav(out[:-4] + ".wav", track)
        mux(video, track, out)
        print(f"  -> {out}")
        all_ok &= ok
    print("all clips fit" if all_ok else "some beats need attention (see above)")
    if args.movie:
        parts = [os.path.join(HERE, "media", "videos", SCENES[sc][0], quality, f"{SCENES[sc][1]}_vo.mp4")
                 for sc in sorted(SCENES)]
        out = os.path.join(HERE, "media", "videos", f"Scenes7-9_{quality}.mp4")
        join_movie(parts, out)
        print(f"movie -> {out}")


def join_movie(parts, out_path):
    """Join scene videos end to end without re-encoding (they share codec and settings).

    Each scene starts exactly where the previous scene's video ends; its audio is shifted by the same amount."""
    srcs = [av.open(p) for p in parts]
    v0, a0 = srcs[0].streams.video[0], srcs[0].streams.audio[0]
    for s in srcs[1:]:
        v, a = s.streams.video[0], s.streams.audio[0]
        same = (v.codec_context.name, v.width, v.height, v.average_rate, bytes(v.codec_context.extradata or b""),
                a.codec_context.name, a.rate) == \
               (v0.codec_context.name, v0.width, v0.height, v0.average_rate, bytes(v0.codec_context.extradata or b""),
                a0.codec_context.name, a0.rate)
        assert same, f"{s.name}: encoding differs from {srcs[0].name}; re-render with the same quality"
    dst = av.open(out_path, "w")
    vout = dst.add_stream_from_template(v0)
    aout = dst.add_stream_from_template(a0)
    offset = 0.0                                        # seconds
    last_dts = {}
    for s in srcs:
        v, a = s.streams.video[0], s.streams.audio[0]
        for pkt in s.demux(v, a):
            if pkt.dts is None:
                continue
            kind = pkt.stream.type
            shift = int(round(offset / pkt.time_base))
            pkt.pts = None if pkt.pts is None else pkt.pts + shift
            pkt.dts += shift
            # AAC encoder priming/padding makes neighbouring scenes overlap by ~20 ms; those packets are
            # silent (every beat opens with LEAD seconds of silence), so the overlap is dropped
            if kind in last_dts and pkt.dts <= last_dts[kind]:
                continue
            last_dts[kind] = pkt.dts
            pkt.stream = vout if kind == "video" else aout
            dst.mux(pkt)
        offset += v.frames / float(v.average_rate)      # whole frames: the next scene starts on a frame boundary
    dst.close()
    for s in srcs:
        s.close()


if __name__ == "__main__":
    main()
