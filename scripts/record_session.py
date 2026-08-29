"""Capture a real terminal session to a timed transcript.

The demo GIF in the README has to be a recording of an actual run, not a
mock-up -- a hand-written "demo" of a pipeline is precisely the kind of
thing this repository argues against everywhere else.

The usual tools (`vhs`, `asciinema`) both need a Unix pty and do not run
on this machine, so capture and rendering are done here instead:
this script records, `render_session_gif.py` draws. Splitting them means
a slow run is captured once and can be re-rendered any number of times
without re-running the pipeline.

Output is JSONL, one object per line: ``{"t": <seconds since start>,
"line": "<raw line, ANSI intact>"}``.

Usage::

    python scripts/record_session.py -o session.jsonl -- pramaan all --scale smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def redact(line: str, root: Path, label: str) -> str:
    r"""Replace absolute repo paths with a short placeholder.

    The transcript and the GIF rendered from it are committed to a public
    repository, and raw output is full of
    `C:\Users\<name>\...\Razorpay_Hackathon\data\...`. That leaks a home
    directory for no benefit, and pushes the informative end of every
    line off the right edge of the frame.
    """
    # Case-insensitively, and across both separators. Windows hands back
    # `c:\Users\...` from `Path.cwd()` while child processes print
    # `C:\Users\...`, so an exact `str.replace` matches nothing and silently
    # leaves every path unredacted -- which is what it did on the first run.
    for variant in {str(root), str(root).replace("\\", "/"), root.as_posix()}:
        line = re.sub(re.escape(variant), label, line, flags=re.IGNORECASE)
    return line


def record(command: list[str], out_path: Path, root: Path, label: str) -> int:
    env = dict(os.environ)
    # Ask the child for colour even though it is writing to a pipe, and pin
    # the width so the rendered frame matches what the run actually emitted.
    env["FORCE_COLOR"] = "1"
    env["COLUMNS"] = env.get("COLUMNS", "100")
    env["PYTHONUNBUFFERED"] = "1"

    # The child emits box-drawing and other non-cp1252 characters. Mirroring
    # them to a Windows console under the default codepage raises
    # UnicodeEncodeError *inside the recorder*, which kills the capture and
    # discards the transcript -- losing a run that took minutes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    started = time.monotonic()
    records: list[dict[str, object]] = []

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = redact(raw.rstrip("\n").rstrip("\r"), root, label)
        records.append({"t": round(time.monotonic() - started, 3), "line": line})
        # Mirror to our own stdout so a long capture is not a silent wait.
        print(line, flush=True)
    code = proc.wait()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        meta = {
            "command": redact(" ".join(command), root, label),
            "exit_code": code,
            "duration_s": round(time.monotonic() - started, 3),
            "lines": len(records),
        }
        fh.write(json.dumps({"meta": meta}) + "\n")
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(
        f"\nrecorded {len(records)} lines over {meta['duration_s']}s -> {out_path}",
        file=sys.stderr,
    )
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--redact-root", type=Path, default=Path.cwd(),
                    help="absolute path to replace in captured output")
    ap.add_argument("--redact-as", default="~/pramaan",
                    help="what to replace it with")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("no command given (put it after `--`)")
    return record(command, args.out, args.redact_root.resolve(), args.redact_as)


if __name__ == "__main__":
    raise SystemExit(main())
