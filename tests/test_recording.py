import json
import shutil
import subprocess

import numpy as np
import pytest

from reachy_rl.recording import MP4Writer


def test_existing_output_is_not_modified(tmp_path):
    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original file")
    with pytest.raises(ValueError, match="already exists"):
        MP4Writer(path, 64, 48, 25)
    assert path.read_bytes() == b"original file"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires system FFmpeg and ffprobe")
def test_encoded_clip_has_expected_dimensions_duration_and_frame_count(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = MP4Writer(path, 64, 48, 25)
    for index in range(25):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, index:index + 10, 0] = 255
        writer.write(frame)
    writer.close()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    metadata = json.loads(probe.stdout)
    assert metadata["streams"][0]["width"] == 64
    assert metadata["streams"][0]["height"] == 48
    assert metadata["streams"][0]["nb_frames"] == "25"
    assert float(metadata["format"]["duration"]) == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires system FFmpeg")
def test_failed_recording_does_not_publish_output(tmp_path):
    path = tmp_path / "failed.mp4"
    writer = MP4Writer(path, 64, 48, 25)
    writer.close(publish=False)
    assert not path.exists()
    assert not list(tmp_path.glob(".reachy-record-*"))
