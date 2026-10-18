"""A native encoder failure must never silently produce a complete session."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from recording_io import SessionWriter


def run():
    real_writer = cv2.VideoWriter

    class DroppingWriter:
        def __init__(self, *args):
            self.writer = real_writer(*args)
            self.index = 0

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, image):
            self.index += 1
            if self.index != 2:
                self.writer.write(image)

        def release(self):
            self.writer.release()

    with tempfile.TemporaryDirectory() as folder:
        for dropped in (False, True):
            target = Path(folder)/('dropped' if dropped else 'normal')
            with patch('recording_io.cv2.VideoWriter', DroppingWriter if dropped else real_writer):
                writer = SessionWriter(target, {'serial': 'test', 'recording_mode': 'fast_lossless'})
                for n in range(4):
                    assert writer.submit(dict(serial='test', stream='ir1', frame_number=n, nominal_fps=30),
                                         np.full((48, 64), n*20, np.uint8))
                writer.close()
            summary = json.loads((target/'summary.json').read_text(encoding='utf8'))
            assert summary['complete'] == (not dropped), summary
            if dropped:
                assert 'saved 3, expected 4' in writer.error, writer.error
            else:
                assert writer.error is None
        rng = np.random.default_rng(7)
        # Repeated native encoder opens exercise the intermittent failure seen in 4.10.
        for attempt in range(12):
            writer = SessionWriter(Path(folder)/f'round_{attempt}', {'serial': 'test', 'recording_mode': 'fast_lossless'})
            for n in range(12):
                gray = rng.integers(0, 255, (480, 640), dtype=np.uint8)
                for stream in ('ir1', 'ir2', 'rgb'):
                    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if stream == 'rgb' else gray
                    assert writer.submit(dict(serial='test', stream=stream, frame_number=n, nominal_fps=30), image)
            writer.close()
            assert writer.error is None, writer.error
            assert all(count == 12 for count in writer.counts.values())
    print('PASS v7: normal recording complete; silent encoder loss rejected; '
          '12 rounds of three-stream lossless encoding')


if __name__ == '__main__':
    run()
