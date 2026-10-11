"""Presentation media: console screenshots, slide renders, PDF and the narrated demo video.

Optional tools (not needed to run NetSentinel itself):
    pip install playwright imageio-ffmpeg
Playwright drives the Microsoft Edge already installed on Windows (no browser download). The narration uses the
Windows text-to-speech voice "Microsoft Zira Desktop" (en-US); any installed English SAPI voice can be named.

    # 1. console screenshots (start the console first: streamlit run app/streamlit_app.py --server.port 8502)
    python scripts/make_media.py screens --url http://localhost:8502 --out docs/media/screens
    # 2. slides -> PNG (one HTML file per slide, as in presentation/slides/) and a PDF
    python scripts/make_media.py slides --deck presentation/project --out docs/media/slides --pdf docs/NetSentinel_slides.pdf
    # 3. narrated video from slide PNGs + the narration script
    python scripts/make_media.py video --script presentation/narration.json --out docs/media/NetSentinel_demo.mp4
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
    args = parser.parse_args()
    if args.cmd == "screens":
        screens(args.url, ROOT / args.out)
    elif args.cmd == "slides":
        slides(ROOT / args.deck, ROOT / args.out, ROOT / args.pdf if args.pdf else None)
    else:
        video(ROOT / args.script, ROOT / args.out, args.voice, args.rate, args.pad)


if __name__ == "__main__":
    main()
