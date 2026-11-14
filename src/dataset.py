"""Dataset indexing, provenance deduplication audit, and duration-based frame decoding."""
import os
import subprocess
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def build_drive_index(drive_datasets: Path) -> dict:
    """Index video files under Google Drive Datasets folder by filename."""
    index = {}
    if not drive_datasets.exists():
        return index
    for p in drive_datasets.rglob("*"):
        if p.suffix.lower() in VIDEO_EXT:
            index.setdefault(p.name, []).append(p)
    return index

def resolve_drive_video(row: dict, drive_index: dict):
    """Resolve a master corpus row against indexed Google Drive files."""
    name = row.get("filename") if isinstance(row.get("filename"), str) else (
        Path(row["hf_rel_path"]).name if isinstance(row.get("hf_rel_path"), str) else None
    )
    cands = drive_index.get(name, []) if name else []
    if len(cands) > 1 and isinstance(row.get("hf_rel_path"), str):
        parts = row["hf_rel_path"].replace("\\", "/").split("/")
        for n in range(2, len(parts) + 1):
            m = [c for c in cands if c.as_posix().endswith("/" + "/".join(parts[-n:]))]
            if not m:
                break
            cands = m
            if len(cands) == 1:
                break
    return cands[0] if cands else None

def decode_cv2(path: Path, t_frames: int = 32, img_size: int = 224) -> np.ndarray:
    """Uniformly sample `t_frames` across a video clip using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(n - 1, 0), t_frames).round().astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            f = out[-1] if out else np.zeros((img_size, img_size, 3), np.uint8)
        else:
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (img_size, img_size), interpolation=cv2.INTER_AREA)
        out.append(f)
    cap.release()
    return np.stack(out)

def probe_duration(path: Path) -> float:
    """Query container duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60
    ).stdout.strip()
    return float(out)

def decode_seek(path: Path, t_frames: int = 32, img_size: int = 224) -> np.ndarray:
    """Fast input-seek decoding via ffmpeg for recordings longer than 20 seconds."""
    dur = probe_duration(path)
    out = []
    for t in np.linspace(0, max(dur - 0.5, 0), t_frames):
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={img_size}:{img_size}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=120
        ).stdout
        if len(r) == img_size * img_size * 3:
            out.append(np.frombuffer(r, np.uint8).reshape(img_size, img_size, 3))
        else:
            out.append(out[-1] if out else np.zeros((img_size, img_size, 3), np.uint8))
    return np.stack(out)
