"""Encoder stalls must not overflow IMU; test real 720p/30fps lossless writes."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

from capture_app import Main
from recording_benchmark import benchmark_recording
from recording_io import SessionWriter


def run():
    real_writer = cv2.VideoWriter
    entered, gate = threading.Event(), threading.Event()

    class BlockedWriter:
        def __init__(self, *args): self.actual = real_writer(*args)
        def isOpened(self): return self.actual.isOpened()
        def write(self, image):
            entered.set()
            if not gate.wait(8): raise TimeoutError('Test did not release encoder')
            self.actual.write(image)
        def release(self): self.actual.release()

    def record(stream, number):
        return dict(serial='test',stream=stream,frame_number=number,nominal_fps=30,
                    host_monotonic_ns=number,host_wall_ns=number,device_ts_ms=number/30*1000,
                    timestamp_domain='hardware_clock',x=0.,y=0.,z=0.)

    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        with patch('recording_io.cv2.VideoWriter',BlockedWriter):
            writer=SessionWriter(root/'independent',{'serial':'test','recording_mode':'fast_lossless'})
            try:
                assert writer.submit(record('ir1',0),np.zeros((48,64),np.uint8))
                assert entered.wait(3)
                for n in range(2000): assert writer.submit(record('gyro',n))
                deadline=time.monotonic()+4
                while writer.snapshot_stats()['written_samples'].get('test_gyro',0)<2000 and time.monotonic()<deadline:
                    time.sleep(.01)
                assert writer.counts.get('test_gyro')==2000, 'IMU must be written while the encoder is blocked'
                assert not writer.error
            finally:
                gate.set();writer.close()
            assert not writer.error
        entered.clear();gate.clear()
        with patch('recording_io.cv2.VideoWriter',BlockedWriter):
            writer=SessionWriter(root/'bounded',{'serial':'test','video_buffer_mb':16,'recording_mode':'fast_lossless'})
            try:
                image=np.zeros((720,1280,3),np.uint8)
                assert writer.submit(record('rgb',0),image)
                assert entered.wait(3)
                rejected=None
                for n in range(1,20):
                    if not writer.submit(record('rgb',n),image): rejected=n;break
                assert rejected is not None and writer.error_category=='video_buffer_full'
                assert writer.snapshot_stats()['peak_video_buffer_mb']<=16.
            finally:
                gate.set();writer.close()
            summary=json.loads((writer.root/'summary.json').read_text(encoding='utf8'))
            assert not summary['complete'] and summary['rejected_sample']['frame_number']==rejected
            assert summary['accepted_sample_counts']==summary['received_sample_counts']
        report=benchmark_recording(root/'speed',seconds=3.,mode='fast_lossless')
        assert report['ok'],report
        assert report['exact_pixels'] and report['expected_frames_per_stream']==90
        assert report['summary']['received_sample_counts']['speedtest_gyro']==900
        logs=Path(__file__).resolve().parents[1]/'logs'
        logs.mkdir(exist_ok=True)
        (logs/'recording_throughput_v8_3.json').write_text(json.dumps(report,indent=2),encoding='utf8')

        # Exercise the actual background process and return UI controls to normal.
        app=QtWidgets.QApplication([])
        main=Main({'resolution':[64,48],'fps':30})
        main.path.setText(str(root/'ui_speed'))
        main.start_recording_benchmark()
        assert not main.benchmark_button.isEnabled() and not main.recording_mode.isEnabled()
        deadline=time.monotonic()+20
        while main.benchmark_process is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert main.benchmark_process is None,'Benchmark process did not finish'
        assert main.benchmark_button.isEnabled() and main.recording_mode.isEnabled()
        assert any('录制测速通过' in line for line in main.log.toPlainText().splitlines()),main.log.toPlainText()
        assert not list((root/'ui_speed').glob('.recording_speed_*'))
        main.close()
    print('PASS v8.3: blocked encoder leaves IMU independent; bounded memory and accepted-frame flush; '
          'three 1280x720 streams at 30fps plus 600Hz IMU; exact raw pixels, asynchronous directory speed test')


if __name__=='__main__':run()
