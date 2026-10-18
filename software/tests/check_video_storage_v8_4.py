"""Compressed recording preserves IR, ordinary MP4 playback, and legacy conversion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
from PyQt5 import QtWidgets

from recording_io import SessionWriter
from recording_benchmark import benchmark_recording
from video_encoding import FFmpegWriter, verify_video
from video_tools import inspect_session, export_mp4
from video_tools_gui import VideoToolsWindow


def rec(stream, number):
    return dict(serial='test', stream=stream, frame_number=number, nominal_fps=30,
                host_monotonic_ns=number, host_wall_ns=number, device_ts_ms=number/30*1000)


def run():
    reports = {}
    with tempfile.TemporaryDirectory(prefix='video_v84_') as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(23)
        # Default mode must preserve all 256 gray levels in both measured IR streams.
        writer = SessionWriter(root/'balanced', {'serial':'test'})
        originals = []
        for n in range(20):
            gray = rng.integers(0, 256, (96, 128), dtype=np.uint8)
            originals.append(gray)
            for stream in ('rgb', 'ir1', 'ir2'):
                im = np.full((96,128,3),(10,40,220),np.uint8) if stream=='rgb' else gray
                assert writer.submit(rec(stream, n), im)
        writer.close()
        assert not writer.error, writer.error
        metadata = json.loads((writer.root/'session.json').read_text(encoding='utf8'))
        assert metadata['lossy_streams']==['rgb']
        for stream in ('ir1', 'ir2'):
            cap = cv2.VideoCapture(str(writer.root/f'test_{stream}.avi'))
            for original in originals:
                ok, decoded = cap.read()
                assert ok and np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), original)
            assert not cap.read()[0]
            cap.release()
        reports['balanced'] = inspect_session(writer.root, full=True)
        assert reports['balanced']['ok']
        # Use Windows' own Media Foundation decoder, independent of FFmpeg/OpenCV's FFmpeg backend.
        if os.name=='nt':
            # Keep Media Foundation's COM initialization out of the Qt GUI test process.
            code = ('import cv2,numpy as np,sys; '
                    'cap=cv2.VideoCapture(sys.argv[1],cv2.CAP_MSMF); ok,im=cap.read(); '
                    'assert ok and im.shape==(96,128,3); '
                    'assert np.max(np.abs(im.astype(float).mean(axis=(0,1))-[10,40,220]))<5; cap.release()')
            subprocess.run([sys.executable, '-c', code, str(writer.root/'test_rgb.mp4')], check=True, timeout=30)
            reports['windows_media_foundation'] = 'decoded_h264_mp4'

        for mode in ('mp4','fast_lossless'):
            w = SessionWriter(root/mode, {'serial':'test','recording_mode':mode})
            for n in range(12):
                for stream in ('rgb','ir1','ir2'):
                    gray = np.full((96,128),n*20,np.uint8)
                    im = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) if stream=='rgb' else gray
                    assert w.submit(rec(stream,n),im)
            w.close()
            assert not w.error,w.error
            assert inspect_session(w.root,full=True)['ok']
            if mode=='mp4':
                assert len(list(w.root.glob('*.mp4')))==3
        old = root/'fast_lossless'
        original_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()}
        converted = export_mp4(old)
        assert converted['ok'], converted
        assert all(f['frames']==12 and f['verification']=='full_decode' for f in converted['files'])
        assert original_hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()}
        # A forged/truncated frame index must not pass the storage checker.
        indices = old/'frames.jsonl'
        saved = indices.read_bytes()
        with indices.open('ab') as handle: handle.write(saved.splitlines(keepends=True)[0])
        assert not inspect_session(old)['ok']
        indices.write_bytes(saved)

        class FailingFinalizer(FFmpegWriter):
            def release(self):
                super().release()
                raise RuntimeError('Injected MP4 finalization error')
        with patch('recording_io.FFmpegWriter',FailingFinalizer):
            w = SessionWriter(root/'failed', {'serial':'test','recording_mode':'mp4'})
            assert w.submit(rec('rgb',0),np.zeros((96,128,3),np.uint8))
            w.close()
        assert not json.loads((w.root/'summary.json').read_text(encoding='utf8'))['complete']
        assert w.error_category=='video_finalize_failed'

        reports['throughput'] = benchmark_recording(root/'benchmark',seconds=6.,mode='balanced')
        assert reports['throughput']['ok'],reports['throughput']
        assert reports['throughput']['lossless_pixels_ok']
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = VideoToolsWindow()
        win.folder.setText(str(old))
        win.full.setChecked(True)
        win.start('inspect')
        deadline = time.monotonic()+20
        while win.process is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert win.process is None and win.last_report['ok'],win.log.toPlainText()
        win.start('export')
        deadline = time.monotonic()+20
        while win.process is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert win.process is None and win.last_report['ok'],win.log.toPlainText()
        win.close()
    logs = Path(__file__).resolve().parents[1]/'logs'
    (logs/'video_storage_v8_4.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
    print('PASS v8.4: IR exact pixels, H264/MP4 Windows decoder, independent full decode, legacy conversion '
          'preserves originals, failed finalization rejected, paced 720p/30fps plus IMU, background GUI inspect/export')


if __name__=='__main__': run()
