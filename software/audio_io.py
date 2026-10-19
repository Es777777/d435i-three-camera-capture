"""PCM audio with PortAudio ADC timestamps mapped to the local monotonic clock."""
import json
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np


def input_devices():
    import sounddevice as sd
    return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]


class AudioRecorder:
    def __init__(self, root, device=None, rate=48000):
        self.root, self.rate = Path(root), int(rate)
        self.device = device
        self.queue = queue.Queue(maxsize=256)
        self.error = None
        self.status_count = 0
        self.last_status = None
        self.samples = 0
        self.level = 0.
        self.stream = None
        self.thread = None

    def start(self):
        import sounddevice as sd
        self.stream = sd.RawInputStream(samplerate=self.rate, channels=1, dtype='int16',
                                        device=self.device, blocksize=1024, callback=self._callback)
        self.device_info = dict(sd.query_devices(self.device, 'input'))
        self.thread = threading.Thread(target=self._write, name='AudioWriter')
        self.thread.start()
        try:
            self.stream.start()
        except Exception:
            self.close()
            raise

    def _callback(self, data, frames, clock, status):
        if self.error:
            return
        now = time.monotonic_ns()
        adc, current = float(clock.inputBufferAdcTime), float(clock.currentTime)
        # ADC time estimates exposure-independent audio time; callback arrival is not sample time.
        valid_adc = adc > 0 and current > 0
        rec = dict(sample_start=self.samples, sample_count=frames,
                   host_monotonic_ns=now, portaudio_adc_s=adc, portaudio_current_s=current,
                   first_sample_host_ns=int(now+(adc-current)*1e9) if valid_adc else None,
                   status=str(status))
        if status:
            # PortAudio status describes a discontinuity, not a dead stream.
            # Keep recording subsequent blocks and preserve the warning for QA.
            self.status_count += 1
            self.last_status = str(status)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.level = float(np.sqrt(np.mean(samples*samples))/32768.) if len(samples) else 0.
        try:
            self.queue.put_nowait((bytes(data), rec))
            self.samples += frames
        except queue.Full:
            self.error = 'Audio writer queue full'

    def _write(self):
        try:
            with wave.open(str(self.root/'audio.wav'), 'wb') as wav, \
                    (self.root/'audio_blocks.jsonl').open('w', encoding='utf8') as log:
                wav.setparams((1, 2, self.rate, 0, 'NONE', 'not compressed'))
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    data, rec = item
                    wav.writeframesraw(data)
                    log.write(json.dumps(rec)+'\n')
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        if self.stream:
            try:
                self.stream.stop()
            except Exception as exc:
                self.error = self.error or 'Audio stop: '+str(exc)
            finally:
                try:
                    self.stream.close()
                except Exception as exc:
                    self.error = self.error or 'Audio close: '+str(exc)
                self.stream = None
        if self.thread:
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=.2)
                    break
                except queue.Full:
                    pass
            self.thread.join()
        (self.root/'audio.json').write_text(json.dumps(dict(
            rate=self.rate, channels=1, sample_width=2, samples=self.samples,
            complete=not bool(self.error or self.status_count), error=self.error,
            status_count=self.status_count, last_status=self.last_status,
            device=getattr(self, 'device_info', {}),
            clock='PortAudio ADC mapped to host monotonic; not hardware-synced to cameras'), indent=2), encoding='utf8')
