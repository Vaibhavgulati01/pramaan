"""Render a recorded terminal session (see `record_session.py`) to a GIF.

Written because `vhs` and `asciinema` both need a Unix pty and neither
runs on this machine. Frames are drawn with Pillow -- already a
dependency -- and encoded by ffmpeg using a generated palette, which is
what keeps a 256-colour GIF of antialiased text from banding.

Timing is compressed rather than faked: real inter-line gaps are kept in
proportion but clamped, so a pipeline that spends 20 s inside one step
does not spend 20 s of GIF on a static frame. `--speed` scales what
remains. The transcript keeps true timings either way, so nothing here
can misrepresent how long the run took, and the caption states the real
duration alongside the replay.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# A dark palette close to a default terminal, chosen to stay legible
# against both GitHub themes.
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
CHROME = (33, 38, 45)

ANSI_16 = {
    30: (72, 76, 82),
    31: (255, 123, 114),
    32: (86, 211, 100),
    33: (232, 187, 65),
    34: (121, 192, 255),
    35: (210, 168, 255),
    36: (86, 211, 194),
    37: FG,
    90: DIM,
    91: (255, 166, 158),
    92: (126, 231, 135),
    93: (242, 204, 96),
    94: (150, 209, 255),
    95: (223, 190, 255),
    96: (120, 226, 213),
    97: (255, 255, 255),
}

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
NON_SGR_ESC_RE = re.compile(r"\x1b\[[0-9;?]*[A-LN-Za-ln-z]")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\CascadiaMono.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


@dataclass
class Span:
    """A run of characters sharing one colour."""

    text: str
    colour: tuple[int, int, int]


def xterm256(n: int) -> tuple[int, int, int]:
    """Map an xterm-256 index to RGB."""
    if n < 8:
        return ANSI_16[30 + n]
    if n < 16:
        return ANSI_16[90 + (n - 8)]
    if n < 232:
        n -= 16
        levels = [0, 95, 135, 175, 215, 255]
        return (levels[n // 36], levels[(n // 6) % 6], levels[n % 6])
    grey = 8 + (n - 232) * 10
    return (grey, grey, grey)


def parse_ansi(line: str) -> list[Span]:
    """Split a line into coloured spans, honouring the SGR codes we emit.

    Cursor movement and other non-colour escapes are dropped rather than
    interpreted: this renders an append-only transcript, so there is no
    cursor to move.
    """
    line = NON_SGR_ESC_RE.sub("", line)
    spans: list[Span] = []
    colour = FG
    pos = 0
    for match in SGR_RE.finditer(line):
        if match.start() > pos:
            spans.append(Span(line[pos : match.start()], colour))
        codes = [int(c) for c in match.group(1).split(";") if c.isdigit()] or [0]
        i = 0
        while i < len(codes):
            code = codes[i]
            if code == 0:
                colour = FG
            elif code == 2:
                colour = DIM
            elif code in ANSI_16:
                colour = ANSI_16[code]
            elif code == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
                colour = xterm256(codes[i + 2])
                i += 2
            elif code == 38 and i + 4 < len(codes) and codes[i + 1] == 2:
                colour = (codes[i + 2], codes[i + 3], codes[i + 4])
                i += 4
            i += 1
        pos = match.end()
    if pos < len(line):
        spans.append(Span(line[pos:], colour))
    return spans or [Span("", FG)]


def concat_quote(path: Path) -> str:
    """Quote a path for ffmpeg's concat demuxer.

    It uses shell-ish single-quote rules and takes double quotes
    *literally*, so `file "C:/x.png"` makes it look for a file whose name
    begins with a quote character. A literal single quote is written as
    `'\\''`, closing and reopening the quoted run.
    """
    return "'" + path.as_posix().replace("'", "'\\''") + "'"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("no monospace font found; add one to FONT_CANDIDATES")


def build_timeline(records: list[dict], max_gap: float, speed: float) -> list[float]:
    """Real inter-line gaps, clamped then scaled, as per-frame durations."""
    times = [float(r["t"]) for r in records]
    gaps = [max(0.0, b - a) for a, b in zip(times, times[1:], strict=False)]
    gaps.append(1.0)  # hold on the final frame before the loop restarts
    return [max(0.05, min(gap, max_gap)) / speed for gap in gaps]


def render(args: argparse.Namespace) -> int:
    raw = [
        json.loads(line)
        for line in args.transcript.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = raw[0]["meta"] if raw and "meta" in raw[0] else {}
    records = [r for r in raw if "line" in r]
    if not records:
        raise SystemExit("transcript has no lines")

    font = load_font(args.font_size)
    char_w = font.getlength("M")
    bbox = font.getbbox("Mgy")
    line_h = int((bbox[3] - bbox[1]) * 1.7)

    pad = 18
    title_h = 34
    width = int(char_w * args.cols) + pad * 2
    height = line_h * args.rows + pad * 2 + title_h

    durations = build_timeline(records, args.max_gap, args.speed)

    caption = args.caption or f"$ {meta.get('command', '')}"
    duration_s = meta.get("duration_s")
    subtitle = f"real: {duration_s:.0f}s" if isinstance(duration_s, int | float) else ""

    tmp = Path(tempfile.mkdtemp(prefix="pramaan-gif-"))
    try:
        frames: list[tuple[Path, float]] = []
        visible: list[str] = []

        for idx, rec in enumerate(records):
            visible.append(str(rec["line"]))
            window = visible[-args.rows :]

            img = Image.new("RGB", (width, height), BG)
            draw = ImageDraw.Draw(img)

            draw.rectangle((0, 0, width, title_h), fill=CHROME)
            for i, dot in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
                x0 = pad + i * 18
                draw.ellipse((x0, 12, x0 + 10, 22), fill=dot)
            draw.text((pad + 66, 8), caption, font=font, fill=DIM)
            if subtitle:
                draw.text(
                    (width - pad - font.getlength(subtitle), 8),
                    subtitle,
                    font=font,
                    fill=DIM,
                )

            for row, line in enumerate(window):
                y = title_h + pad + row * line_h
                x = float(pad)
                budget = args.cols
                for span in parse_ansi(line):
                    if budget <= 0:
                        break
                    chunk = span.text[:budget]
                    budget -= len(chunk)
                    if chunk.strip():
                        draw.text((x, y), chunk, font=font, fill=span.colour)
                    x += font.getlength(chunk)

            path = tmp / f"f{idx:05d}.png"
            img.save(path)
            frames.append((path, durations[idx]))

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg not found on PATH")

        concat = tmp / "frames.txt"
        lines = []
        for path, dur in frames:
            lines.append(f"file {concat_quote(path)}")
            lines.append(f"duration {dur:.3f}")
        # The concat demuxer drops the final entry's duration, so the last
        # frame is repeated to give it one.
        lines.append(f"file {concat_quote(frames[-1][0])}")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

        palette = tmp / "palette.png"
        base = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat),
        ]
        vf = f"fps={args.fps},scale={width}:-1:flags=lanczos"
        subprocess.run(
            [*base, "-vf", f"{vf},palettegen=stats_mode=diff", str(palette)], check=True
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *base, "-i", str(palette),
                "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
                "-loop", "0", str(args.out),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({len(records)} frames, {width}x{height}, {size_kb:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("assets/demo.gif"))
    ap.add_argument("--caption", default=None, help="overrides the recorded command line")
    ap.add_argument("--cols", type=int, default=110)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument(
        "--max-gap",
        type=float,
        default=1.2,
        help="clamp real inter-line gaps to this many seconds before scaling",
    )
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"no transcript at {args.transcript}", file=sys.stderr)
        return 1
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
