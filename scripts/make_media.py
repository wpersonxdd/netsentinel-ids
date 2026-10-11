"""Presentation media: console screenshots, slide renders, PDF and the narrated demo video.

Optional tools (not needed to run NetSentinel itself):
    pip install playwright imageio-ffmpeg
Playwright drives the Microsoft Edge already installed on Windows (no browser download). The narration uses the
Windows text-to-speech voice "Microsoft Zira Desktop" (en-US); any installed English SAPI voice can be named.

    # 1. console screenshots (start the console first: streamlit run app/streamlit_app.py --server.port 8502)
    python scripts/make_media.py screens --url http://localhost:8502 --out docs/media/screens
    # 2. slides -> PNG (one HTML file per slide, as in presentation/slides/) and a PDF
    python scripts/make_media.py slides --deck presentation/project --out docs/media/slides --pdf docs/NetSentinel_slides.pdf
    # 3. narrated slide video from slide PNGs + the narration script
    python scripts/make_media.py video --script presentation/narration.json --out docs/media/NetSentinel_slides.mp4
    # 4. animated neon demo video: scenes in presentation/video/, narration in presentation/video/script.json
    python scripts/make_media.py motion --out docs/media/NetSentinel_demo.mp4
    python scripts/make_media.py motion --only 04_pipeline --stills          # quick check: one still per line
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
# spoken form of terms the TTS voice mispronounces (captions keep the written form)
PRONOUNCE = {
    "CIC-IDS2017": "C I C, I D S, 2017",
    "DN-CY-01": "D N, C Y, zero one",
    "DDoS": "distributed denial of service",
    "DoS": "denial of service",
    "Macro F1": "Macro F one",
}


# --------------------------------------------------------------------------------------------------
# console screenshots
# --------------------------------------------------------------------------------------------------


def screens(url: str, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.get_by_text("Incidents", exact=True).first.wait_for(timeout=120_000)
        time.sleep(4)

        def shot(name: str) -> None:
            time.sleep(2.5)
            page.screenshot(path=str(out / f"{name}.png"))
            print("saved", out / f"{name}.png")

        shot("console_queue")
        page.get_by_text("Why: evidence from the strongest flow").first.scroll_into_view_if_needed()
        page.mouse.wheel(0, -260)
        shot("console_incident")
        page.get_by_role("tab", name=re.compile("Model performance")).click()
        page.get_by_text("Deployed:").first.wait_for(timeout=60_000)
        shot("console_performance")
        page.get_by_role("tab", name=re.compile("Novel attacks")).click()
        page.get_by_text("Attack families never seen in training").first.wait_for(timeout=60_000)
        shot("console_drift")
        page.get_by_role("tab", name=re.compile("Alert queue")).click()
        page.get_by_text("Score a traffic file").first.click()
        page.get_by_role("button", name="Use bundled demo traffic").click()
        page.get_by_text(re.compile(r"flows scored")).first.wait_for(timeout=180_000)
        page.mouse.wheel(0, -2000)
        shot("console_scored")
        browser.close()


# --------------------------------------------------------------------------------------------------
# slides -> PNG / PDF
# --------------------------------------------------------------------------------------------------


def slides(deck: Path, out: Path, pdf: Path | None) -> list[Path]:
    """Render each slide file (one <section> on a 1920x1080 canvas) exactly as written."""
    from playwright.sync_api import sync_playwright

    index = json.loads((deck / "deck.json").read_text(encoding="utf-8"))
    assets = json.loads((deck / "assets.json").read_text(encoding="utf-8")) if (deck / "assets.json").exists() else {}
    fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in index.get("faces", {}).values() if "href" in f)
    out.mkdir(parents=True, exist_ok=True)
    pngs = []
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        for i, sid in enumerate(index["order"], start=1):
            html = (deck / "slides" / f"{sid}.html").read_text(encoding="utf-8")
            html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)
            for blob, local in assets.items():  # deck assets -> local files for rendering
                html = html.replace(blob, (ROOT / local).resolve().as_uri())
            doc = (f"<!doctype html><html><head><meta charset='utf-8'>{fonts}<style>html,body{{margin:0;padding:0}}"
                   f"section{{width:{W}px;height:{H}px;box-sizing:border-box;position:relative;overflow:hidden}}"
                   f"h1,h2,h3,p,ul,ol{{margin:0}}</style></head><body>{html}</body></html>")
            src = Path(tmp) / f"{sid}.html"
            src.write_text(doc, encoding="utf-8")
            page.goto(src.as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            time.sleep(0.4)
            png = out / f"{i:02d}_{sid}.png"
            page.screenshot(path=str(png))
            pngs.append(png)
            print("rendered", png)
        browser.close()
    if pdf:
        from PIL import Image

        pdf.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.open(p).convert("RGB") for p in pngs]
        images[0].save(pdf, save_all=True, append_images=images[1:], resolution=144)
        print("wrote", pdf)
    return pngs


# --------------------------------------------------------------------------------------------------
# narrated video
# --------------------------------------------------------------------------------------------------


def tts(text: str, wav: Path, voice: str, rate: int) -> float:
    """Windows SAPI text-to-speech -> WAV; returns the duration in seconds."""
    for written, spoken in PRONOUNCE.items():
        text = re.sub(rf"\b{re.escape(written)}\b", spoken, text)
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}'); $s.Rate = {rate};"
        f"$s.SetOutputToWaveFile('{wav}');"
        "$t = [Console]::In.ReadToEnd(); $s.Speak($t); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], input=text, text=True, check=True,
                   encoding="utf-8")
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _chunks(text: str, start: float, duration: float) -> list[tuple[float, float, str]]:
    """Split narration into caption lines timed proportionally to their length."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    total = sum(len(s) for s in sentences) or 1
    out, t = [], start
    for s in sentences:
        d = duration * len(s) / total
        out.append((t, t + d, s))
        t += d
    return out


def video(script: Path, out: Path, voice: str, rate: int, pad: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    items = json.loads(script.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    srt, clock = [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for n, item in enumerate(items):
            image = (ROOT / item["image"]).resolve()
            wav = Path(tmp) / f"{n:02d}.wav"
            speech = tts(item["text"], wav, voice, rate)
            length = speech + 2 * pad
            clip = Path(tmp) / f"{n:02d}.mp4"
            fade_out = max(length - 0.35, 0)
            subprocess.run([
                ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(image),
                "-i", str(wav), "-filter_complex",
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0F1B2D,"
                f"format=yuv420p,fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out:.2f}:d=0.35[v];"
                f"[1:a]adelay={int(pad * 1000)}|{int(pad * 1000)},apad[a]",
                "-map", "[v]", "-map", "[a]", "-t", f"{length:.2f}", "-r", "30",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1", str(clip)], check=True)
            parts.append(clip)
            srt += _chunks(item["text"], clock + pad, speech)
            clock += length
            print(f"segment {n + 1}/{len(items)}: {length:.1f}s")
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        joined = Path(tmp) / "joined.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(joined)], check=True)
        srt_path = out.with_suffix(".srt")
        srt_path.write_text("".join(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n\n"
                                    for i, (a, b, t) in enumerate(srt, start=1)), encoding="utf-8")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(joined), "-i", str(srt_path),
                        "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
                        "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out} ({clock / 60:.1f} min) and {srt_path}")


# --------------------------------------------------------------------------------------------------
# motion video: animated neon scenes (presentation/video/) + narration, rendered frame by frame
# --------------------------------------------------------------------------------------------------

TTS_CACHE = Path(tempfile.gettempdir()) / "netsentinel_tts"


def cached_tts(text: str, voice: str, rate: int) -> tuple[Path, float]:
    """TTS with an on-disk cache, so re-rendering a scene does not re-synthesise unchanged lines."""
    import hashlib

    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    wav = TTS_CACHE / (hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16] + ".wav")
    if not wav.exists():
        tts(text, wav, voice, rate)
    with wave.open(str(wav)) as w:
        return wav, w.getnframes() / w.getframerate()


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, "16-bit PCM expected"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        return data, w.getframerate()


def _write_wav(path: Path, data, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def scene_timing(lines: list[str], voice: str, rate: int, lead: float, gap: float, tail: float):
    """Narration lines -> (wav files, line durations, cue times, scene duration)."""
    wavs, durs, cues, t = [], [], [], lead
    for line in lines:
        wav, d = cached_tts(line, voice, rate)
        wavs.append(wav)
        durs.append(round(d, 3))
        cues.append(round(t, 3))
        t += d + gap
    return wavs, durs, cues, round(t - gap + tail, 3)


def scene_audio(wavs: list[Path], cues: list[float], duration: float, out: Path, sr: int = 44100) -> None:
    import numpy as np

    track = np.zeros(int(round(duration * sr)) + 1, dtype=np.float32)
    for wav, start in zip(wavs, cues):
        data, rate = _read_wav(wav)
        if rate != sr:  # linear resampling is enough for speech
            x = np.arange(0, len(data) * sr / rate) * rate / sr
            data = np.interp(x, np.arange(len(data)), data).astype(np.float32)
        i = int(round(start * sr))
        n = min(len(data), len(track) - i)
        track[i:i + n] += data[:n]
    _write_wav(out, track, sr)


def ambient_music(duration: float, out: Path, sr: int = 44100) -> None:
    """Quiet synth pad with soft arpeggio plucks (A minor: Am - F - C - G), generated so no licensed audio is needed."""
    import numpy as np

    t = np.arange(int(duration * sr)) / sr
    note = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    chords = [[45, 52, 57, 60, 64], [41, 48, 53, 57, 60], [48, 55, 60, 64, 67], [43, 50, 55, 59, 62]]
    bar = 8.0
    mix = np.zeros_like(t)
    for c, chord in enumerate(chords * int(duration // (bar * 4) + 2)):
        start = c * bar
        if start > duration:
            break
        seg = (t >= start - 2) & (t < start + bar + 2)
        ts = t[seg] - start
        env = np.clip(np.minimum((ts + 2) / 3, (bar + 2 - ts) / 3), 0, 1) ** 2
        for m in chord:
            f = note(m)
            voice = sum(np.sin(2 * np.pi * (f + d) * ts + p) for d, p in ((-0.35, 0), (0.35, 1.3), (0, 2.1)))
            mix[seg] += env * voice * (0.05 if m < 50 else 0.03)
        for k in range(16):  # eighth-note plucks over the chord's upper notes
            on = start + k * bar / 16
            sel = (t >= on) & (t < on + 1.6)
            tt = t[sel] - on
            f = note(chord[(k * 3) % len(chord)] + 24)
            mix[sel] += 0.018 * np.exp(-tt * 4.5) * np.sin(2 * np.pi * f * tt) * (0.6 + 0.4 * np.sin(np.pi * k / 8))
    mix *= 0.5 * (1 + 0.15 * np.sin(2 * np.pi * t / 11))
    fade = np.clip(np.minimum(t / 3, (duration - t) / 4), 0, 1)
    mix = mix * fade
    mix = 0.5 * mix / (np.abs(mix).max() + 1e-9)
    _write_wav(out, mix, sr)


def render_scene(browser, html: Path, timing: dict, fps: int, out_mp4: Path | None, audio: Path | None,
                 captions: bool, stills: list[float] | None = None, stills_dir: Path | None = None) -> None:
    import imageio_ffmpeg

    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    page.add_init_script(
        "window.CAPTURE = true;"
        f"window.CUES = {json.dumps(timing['cues'])}; window.LDUR = {json.dumps(timing['durs'])};"
        f"window.LINES = {json.dumps(timing['lines'])}; window.DURATION = {timing['duration']};"
        f"window.SHOW_CAPTIONS = {'true' if captions else 'false'};")
    page.goto(html.resolve().as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready.then(() => new Promise(r => requestAnimationFrame(() => r())))")
    if stills:
        for s in stills:
            page.evaluate("t => seek(t)", s)
            page.screenshot(path=str(stills_dir / f"{html.stem}_{s:05.1f}.png"))
        page.close()
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames = int(round(timing["duration"] * fps))
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
         "-i", str(audio), "-map", "0:v", "-map", "1:a", "-t", f"{timing['duration']:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
         "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "1", str(out_mp4)], stdin=subprocess.PIPE)
    started = time.time()
    for f in range(frames):
        page.evaluate("t => seek(t)", f / fps)
        proc.stdin.write(page.screenshot(type="jpeg", quality=92))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {html.name}")
    page.close()
    print(f"  {html.stem}: {frames} frames in {time.time() - started:.0f}s")


def motion(script: Path, out: Path, fps: int, captions: bool, music: bool, only: list[str] | None,
           stills: list[float] | None, stills_dir: Path) -> None:
    import imageio_ffmpeg
    from playwright.sync_api import sync_playwright

    spec = json.loads(script.read_text(encoding="utf-8"))
    voice, rate = spec.get("voice", "Microsoft Zira Desktop"), spec.get("rate", 1)
    lead, gap, tail = spec.get("lead", 0.9), spec.get("gap", 0.35), spec.get("tail", 1.3)
    scenes = [s for s in spec["scenes"] if not only or any(o in s["file"] for o in only)]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        browser = p.chromium.launch(channel="msedge", headless=True)
        clips, srt, clock = [], [], 0.0
        for n, scene in enumerate(scenes):
            wavs, durs, cues, duration = scene_timing(scene["lines"], voice, rate, lead, gap, tail)
            timing = {"cues": cues, "durs": durs, "lines": scene["lines"], "duration": duration}
            html = script.parent / scene["file"]
            if stills is not None:
                stills_dir.mkdir(parents=True, exist_ok=True)
                points = stills or [c + d * 0.8 for c, d in zip(cues, durs)]
                render_scene(browser, html, timing, fps, None, None, captions, points, stills_dir)
                print(f"stills {html.stem}: cues {cues}, duration {duration}s")
                continue
            audio = tmp / f"{n:02d}.wav"
            scene_audio(wavs, cues, duration, audio)
            clip = tmp / f"{n:02d}.mp4"
            print(f"scene {n + 1}/{len(scenes)} {html.stem}: {duration:.1f}s")
            render_scene(browser, html, timing, fps, clip, audio, captions)
            clips.append(clip)
            srt += [(clock + c, clock + c + d, line) for c, d, line in zip(cues, durs, scene["lines"])]
            clock += duration
        browser.close()
        if stills is not None:
            return
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        joined = tmp / "joined.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(joined)], check=True)
        mixed = joined
        if music:
            bed = tmp / "music.wav"
            ambient_music(clock, bed)
            mixed = tmp / "mixed.mp4"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(joined), "-i", str(bed), "-filter_complex",
                            "[0:a]asplit=2[v][key];[1:a]volume=0.32[m];"
                            "[m][key]sidechaincompress=threshold=0.02:ratio=6:attack=30:release=600[duck];"
                            "[v][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(mixed)],
                           check=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        srt_path = out.with_suffix(".srt")
        srt_path.write_text("".join(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n\n"
                                    for i, (a, b, t) in enumerate(srt, start=1)), encoding="utf-8")
        # final encode: limited-range video at a size GitHub accepts (~30 MB for 6 minutes)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mixed), "-i", str(srt_path),
                        "-map", "0:v", "-map", "0:a", "-map", "1", "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-maxrate", "2500k", "-bufsize", "5000k",
                        "-tune", "animation", "-c:a", "copy", "-c:s", "mov_text",
                        "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out} ({clock / 60:.1f} min) and {srt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("screens")
    p.add_argument("--url", default="http://localhost:8502")
    p.add_argument("--out", default="docs/media/screens")
    p = sub.add_parser("slides")
    p.add_argument("--deck", default="presentation")
    p.add_argument("--out", default="docs/media/slides")
    p.add_argument("--pdf")
    p = sub.add_parser("video")
    p.add_argument("--script", default="presentation/narration.json")
    p.add_argument("--out", default="docs/media/NetSentinel_demo.mp4")
    p.add_argument("--voice", default="Microsoft Zira Desktop")
    p.add_argument("--rate", type=int, default=0)
    p.add_argument("--pad", type=float, default=0.6)
    p = sub.add_parser("motion")
    p.add_argument("--script", default="presentation/video/script.json")
    p.add_argument("--out", default="docs/media/NetSentinel_demo.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--only", nargs="*", help="render only scenes whose file name contains one of these")
    p.add_argument("--stills", nargs="*", type=float, help="save PNG stills (at these seconds, or one per line) instead")
    p.add_argument("--stills-dir", default="build/stills")
    args = parser.parse_args()
    if args.cmd == "motion":
        motion(ROOT / args.script, ROOT / args.out, args.fps, not args.no_captions, not args.no_music, args.only,
               args.stills, ROOT / args.stills_dir)
    elif args.cmd == "screens":
        screens(args.url, ROOT / args.out)
    elif args.cmd == "slides":
        slides(ROOT / args.deck, ROOT / args.out, ROOT / args.pdf if args.pdf else None)
    else:
        video(ROOT / args.script, ROOT / args.out, args.voice, args.rate, args.pad)


if __name__ == "__main__":
    main()
