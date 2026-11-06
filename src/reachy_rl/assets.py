"""Extract model data from a hash-verified official PyPI wheel, without SDK imports."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def asset_dir() -> Path:
    return Path(os.environ.get("REACHY_ASSET_DIR", ROOT / "vendor" / "reachy_mini")).expanduser().resolve()


def fetch(version: str | None = None) -> None:
    lock_path = ROOT / "model.lock.json"
    if version is not None or not lock_path.exists():
        suffix = f"/{version}" if version else ""
        with urllib.request.urlopen(f"https://pypi.org/pypi/reachy-mini{suffix}/json", timeout=60) as response:
            meta = json.load(response)
        wheel = next(item for item in meta["urls"] if item["filename"].endswith("py3-none-any.whl"))
        lock = {"package": "reachy-mini", "version": meta["info"]["version"],
                "url": wheel["url"], "sha256": wheel["digests"]["sha256"]}
    else:
        lock = json.loads(lock_path.read_text())
    target = asset_dir()
    manifest = target / "source.json"
    if manifest.exists() and json.loads(manifest.read_text()) == lock:
        print(f"Model assets already available: {target}")
        return
    with urllib.request.urlopen(lock["url"], timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != lock["sha256"]:
        raise RuntimeError("Reachy wheel SHA256 does not match model.lock.json")
    with zipfile.ZipFile(io.BytesIO(content)) as wheel:
        for member in wheel.infolist():
            name = PurePosixPath(member.filename)
            if ".." in name.parts or name.is_absolute() or member.is_dir():
                continue
            if member.filename.startswith(("reachy_mini/descriptions/", "reachy_mini/assets/")):
                relative = Path(*name.parts[1:])
            elif ".dist-info/" in member.filename and "license" in member.filename.lower():
                relative = Path("licenses", *name.parts[1:])
            else:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(wheel.read(member))
    scene = target / "descriptions/reachy_mini/mjcf/scenes/empty.xml"
    if not scene.is_file():
        raise RuntimeError("Official wheel did not contain the expected Reachy MJCF scene")
    manifest.write_text(json.dumps(lock, indent=2) + "\n")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Extracted Reachy Mini {lock['version']} model to {target}")
