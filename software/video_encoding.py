"""Bundled FFmpeg encoders. No dependency on the system PATH or shell."""
import os
from pathlib import Path
import subprocess
import threading
import time

import imageio_ffmpeg


def executable():
    # Use the pinned wheel's binary, not an unrelated PATH/environment override.
    binaries = Path(imageio_ffmpeg.__file__).parent / 'binaries'
    matches = sorted(binaries.glob('ffmpeg-win*.exe')) if os.name == 'nt' else []
    return str(matches[0]) if matches else imageio_ffmpeg.get_ffmpeg_exe()


def hidden_process_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}


def encoding_args(codec, fps):
    if codec == 'FFV1':
        return ['-c:v', 'ffv1', '-level', '3', '-coder', '0', '-context', '0',
                '-g', '1', '-slices', '16', '-slicecrc', '1', '-threads', '2', '-pix_fmt', 'gray']
    return ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-crf', '18' if codec == 'H264_RGB' else '12',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-threads', '2', '-g', str(max(1, round(fps))),
            '-bf', '0', '-movflags', '+faststart']


class FFmpegWriter:
    def __init__(self, path, codec, fps, shape, write_timeout=30.):
        self.path = Path(path)
        self.closed = False
        self.shape = shape
        h, w = shape[:2]
        if codec != 'FFV1' and (w % 2 or h % 2):
            raise ValueError('MP4 requires even image width and height')
        self.log_path = self.path.with_suffix(self.path.suffix + '.encoder.log')
        self.log = self.log_path.open('wb')
        command = [executable(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24' if len(shape) == 3 else 'gray',
                   '-video_size', f'{w}x{h}', '-framerate', str(fps), '-i', 'pipe:0',
                   '-an', *encoding_args(codec, fps), str(self.path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log, **hidden_process_options())
        except Exception:
            self.log.close()
            raise
        self.write_timeout = write_timeout
        self.operation_started = None
        self.timed_out = False
        self.watchdog_stop = threading.Event()
        self.watchdog = threading.Thread(target=self._watch_encoder, name='EncoderWatchdog', daemon=True)
        self.watchdog.start()

    def _watch_encoder(self):
        while not self.watchdog_stop.wait(.25):
            started = self.operation_started
            if started is not None and time.monotonic()-started > self.write_timeout:
                self.timed_out = True
                # Killing the reader also releases a writer blocked on stdin.
                try:
                    self.process.kill()
                except OSError:
                    pass
                return

    def write(self, image):
        if image.shape != self.shape or str(image.dtype) != 'uint8':
            raise ValueError('Video image format changed during recording')
        if self.process.poll() is not None:
            raise RuntimeError('Video encoder exited before recording finished')
        self.operation_started = time.monotonic()
        try:
            self.process.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError) as exc:
            if self.timed_out:
                raise RuntimeError('Video encoder stalled; stopped waiting for input pipe') from exc
            raise
        finally:
            self.operation_started = None

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.operation_started = time.monotonic()
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            finally:
                self.operation_started = None
            try:
                code = self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError('Video encoder finalization timed out')
        finally:
            self.watchdog_stop.set()
            self.watchdog.join()
            self.log.close()
        if self.timed_out:
            raise RuntimeError('Video encoder stalled; partial video may be incomplete')
        if code:
            detail = self.log_path.read_text(encoding='utf8', errors='replace')[-2000:]
            raise RuntimeError(f'Video encoder failed ({code}): {detail}')


def verify_video(path, expected=None, full=False):
    """Check actual decoding, not only the AVI/MP4 header's reported count."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open saved video: ' + str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if expected is not None and count != expected:
            raise ValueError(f'Video frame count mismatch: {Path(path).name}: saved {count}, expected {expected}')
        if count <= 0:
            raise ValueError('Saved video contains no frames')
        decoded = 0
        if full:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError('Decoded image size changed')
                decoded += 1
            if decoded != count:
                raise ValueError(f'Video decode count mismatch: {decoded} / {count}')
        else:
            for index in sorted({0, count//2, count-1}):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (height, width):
                    raise ValueError(f'Cannot decode video frame {index}')
                decoded += 1
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        return dict(file=Path(path).name, bytes=Path(path).stat().st_size, frames=count, fps=fps,
                    duration_seconds=count/fps if fps else None, width=width, height=height,
                    fourcc=''.join(chr((fourcc >> 8*i) & 255) for i in range(4)),
                    decoded_frames_checked=decoded, verification='full_decode' if full else 'first_middle_last_decode')
    finally:
        cap.release()
