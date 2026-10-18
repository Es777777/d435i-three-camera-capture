"""Paced three-stream recording test in the user's actual output directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

from recording_io import RECORDING_MODES, SessionWriter


def benchmark_recording(folder, seconds=3., width=1280, height=720, fps=30, mode='balanced'):
    target = Path(folder).resolve()
    target.mkdir(parents=True, exist_ok=True)
    required = int(width*height*6*fps*seconds*1.25)+64*1024*1024
    if shutil.disk_usage(target).free < required:
        raise ValueError(f'保存目录剩余空间不足以进行测速，至少需要约 {required/1024**3:.1f} GB。')
    rng = np.random.default_rng(37)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
    rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    expected = {'ir1': gray, 'ir2': gray, 'rgb': rgb}
    count = max(1, int(round(seconds*fps)))
    peaks = []
    with tempfile.TemporaryDirectory(prefix='.recording_speed_', dir=target) as temporary:
        assert Path(temporary).resolve().parent == target
        writer = SessionWriter(Path(temporary)/'test', dict(serial='speedtest', recording_mode=mode))
        started = time.perf_counter()
        imu_number = 0
        try:
            for number in range(count):
                time.sleep(max(0., started+number/fps-time.perf_counter()))
                for _ in range(10):
                    for sensor in ('gyro', 'accel'):
                        rec = dict(serial='speedtest', stream=sensor, frame_number=imu_number,
                            device_ts_ms=number*1000/fps, host_monotonic_ns=time.monotonic_ns(),
                            host_wall_ns=time.time_ns(), timestamp_domain='hardware_clock', x=0., y=0.,
                            z=9.80665 if sensor == 'accel' else 0., units='rad/s' if sensor == 'gyro' else 'm/s^2')
                        if not writer.submit(rec):
                            break
                    imu_number += 1
                for stream, image in expected.items():
                    rec = dict(serial='speedtest', stream=stream, frame_number=number, nominal_fps=fps,
                        device_ts_ms=number*1000/fps, host_monotonic_ns=time.monotonic_ns(),
                        host_wall_ns=time.time_ns(), timestamp_domain='hardware_clock')
                    if not writer.submit(rec, image):
                        break
                peaks.append(writer.snapshot_stats()['video_buffer_mb'])
                if writer.error:
                    break
            producer_seconds = time.perf_counter()-started
            # Measure drain delay after the final sample's nominal frame interval.
            time.sleep(max(0., started+count/fps-time.perf_counter()))
            stopped = time.perf_counter()
        finally:
            writer.close()
        # Include OS write-cache flushing in the directory speed test. A cached
        # half-gigabyte burst alone is not evidence of disk write throughput.
        for file in writer.root.iterdir():
            if file.is_file():
                with file.open('r+b') as handle:
                    os.fsync(handle.fileno())
        flush_seconds = time.perf_counter()-stopped
        drain_seconds = max(0., flush_seconds-writer.verification_seconds)
        summary = json.loads((writer.root/'summary.json').read_text(encoding='utf8'))
        exact, decodable = True, True
        lossless_checks = []
        for stream, image in expected.items():
            state = writer.videos['speedtest_'+stream]
            lossless = not state['codec'].startswith('H264')
            reader = cv2.VideoCapture(str(writer.root/state['video_file']))
            try:
                indices = [0, max(0, int(reader.get(cv2.CAP_PROP_FRAME_COUNT))-1)]
                for index in indices:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, decoded = reader.read()
                    if ok and image.ndim == 2:
                        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                    same = ok and np.array_equal(decoded, image)
                    exact = exact and same
                    decodable = decodable and ok and decoded.shape == image.shape
                    if lossless:
                        lossless_checks.append(same)
            finally:
                reader.release()
        correct_counts = all(writer.counts.get('speedtest_'+stream) == count for stream in expected)
        # A large buffer must not disguise sustained inability to keep up.
        tail = peaks[len(peaks)//2:]
        growing = len(tail)>2 and tail[-1]-min(tail) > width*height*6/1024/1024*fps*.3
        sustained = (summary['complete'] and correct_counts and decodable and all(lossless_checks) and not growing
                     and producer_seconds <= seconds+1. and drain_seconds <= max(1., seconds*.3))
        size = sum((writer.root/s['video_file']).stat().st_size for s in writer.videos.values())
        return dict(ok=bool(sustained), folder=str(target), mode=mode, width=width, height=height, fps=fps,
            test_seconds=seconds, producer_seconds=producer_seconds, flush_seconds=flush_seconds,
            drain_and_fsync_seconds=drain_seconds, verification_seconds=writer.verification_seconds,
            exact_pixels=bool(exact), expected_frames_per_stream=count,
            lossless_pixels_ok=all(lossless_checks) if lossless_checks else None,
            all_streams_decodable=bool(decodable),
            flushed_to_disk=True,
            sustained_backlog_growth=bool(growing), video_bytes=size,
            average_file_mb_s=size/max(seconds, .001)/1e6, summary=summary)


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--mode', choices=RECORDING_MODES, default='balanced')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    try:
        report = benchmark_recording(args.folder, width=args.width, height=args.height, fps=args.fps, mode=args.mode)
    except Exception as exc:
        report = dict(ok=False, folder=args.folder, error=str(exc))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
