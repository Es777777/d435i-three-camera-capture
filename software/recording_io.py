"""Independent video writers with explicit pixel fidelity and separate sample metadata."""
import csv
from copy import deepcopy
import json
import queue
import threading
import time
from pathlib import Path

import cv2

from pose_stamps import SCHEMA as POSE_SCHEMA, make_pose_stamp, motion_alarm
from video_encoding import FFmpegWriter, verify_video

RECORDING_MODES = {
    'balanced': {'rgb': 'H264_RGB', 'ir1': 'FFV1', 'ir2': 'FFV1'},
    'mp4': {'rgb': 'H264_RGB', 'ir1': 'H264_IR', 'ir2': 'H264_IR'},
    'fast_lossless': {'rgb': 'RGBA', 'ir1': 'Y800', 'ir2': 'Y800'},
    'ffv1': {'rgb': 'FFV1', 'ir1': 'FFV1', 'ir2': 'FFV1'},
}
IMU_FIELDS = ['serial', 'stream', 'frame_number', 'device_ts_ms', 'timestamp_domain',
              'host_monotonic_ns', 'host_wall_ns', 'x', 'y', 'z', 'units']


class SessionWriter:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.metadata = deepcopy(metadata)
        self.mode = self.metadata.get('recording_mode', 'balanced')
        if self.mode not in RECORDING_MODES:
            raise ValueError('Unknown recording mode: '+str(self.mode))
        self.codecs = dict(RECORDING_MODES[self.mode])
        self.buffer_limit = max(16, min(1024, int(self.metadata.get('video_buffer_mb', 256))))*1024*1024
        self.pose_stamp = make_pose_stamp(self.metadata['calibration']) if self.metadata.get('calibration') else None
        self.metadata.update(recording_mode=self.mode, recording_codecs=self.codecs,
                             lossy_streams=[s for s, c in self.codecs.items() if c.startswith('H264')],
                             video_buffer_mb=self.buffer_limit//1024//1024,
                             writer_architecture='independent_video_and_metadata_queues_v1')
        if self.pose_stamp:
            self.metadata.update(pose_stamp_id=self.pose_stamp['pose_stamp_id'],
                                 pose_stamps_file='pose_stamps.json',
                                 settlement_status='deferred_until_fusion', fixed_target_assumed=True)
        self.root.mkdir(parents=True, exist_ok=False)
        # IMU samples and already encoded frame metadata contain no image buffers.
        self.items = queue.Queue(maxsize=16384)
        self.videos = {}
        self.lock = threading.RLock()
        self.close_lock = threading.Lock()
        self.error = None
        self.error_category = None
        self.stop_reason = None
        self.audio_status = None
        self.rejected_sample = None
        self.counts, self.accepted, self.gaps, self.last_numbers = {}, {}, {}, {}
        self.accepting, self.closed = True, False
        self.buffer_bytes = self.peak_buffer_bytes = self.peak_metadata_items = 0
        self.first_motion_ns = None
        self.started = time.monotonic()
        self.verification_seconds = 0.
        (self.root/'session.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf8')
        if self.pose_stamp:
            (self.root/'calibration.json').write_text(json.dumps(self.metadata['calibration'], indent=2), encoding='utf8')
            (self.root/'pose_stamps.json').write_text(json.dumps(dict(schema=POSE_SCHEMA,
                stamps={self.pose_stamp['pose_stamp_id']: self.pose_stamp}), indent=2), encoding='utf8')
        self.thread = threading.Thread(target=self._run_metadata, name='SampleMetadataWriter')
        self.thread.start()

    def _fail(self, message, category):
        with self.lock:
            if not self.error:
                self.error, self.error_category = str(message), category
            self.accepting = False

    def _reject(self, rec, message, category):
        self.rejected_sample = {k: rec.get(k) for k in ('serial', 'stream', 'frame_number', 'host_monotonic_ns')}
        self._fail(message, category)
        return False

    def submit(self, record, image=None):
        rec = dict(record)
        key = rec['serial']+'_'+rec['stream']
        with self.lock:
            if not self.accepting:
                return False
            if image is None:
                try:
                    self.items.put_nowait((rec, False))
                    self.peak_metadata_items = max(self.peak_metadata_items, self.items.qsize())
                except queue.Full:
                    return self._reject(rec, 'IMU/帧索引写入队列已满，已停止录像并保留已接收数据；详见 summary.json。', 'metadata_queue_full')
            else:
                size = int(image.nbytes)
                if self.buffer_bytes+size > self.buffer_limit:
                    return self._reject(rec,
                        f'视频缓存已达 {self.buffer_limit//1024//1024} MB：编码或保存速度持续不足，已停止并保留已接收数据；请查看逐路写入统计或测试保存目录速度。',
                        'video_buffer_full')
                if key not in self.videos:
                    state = dict(queue=queue.Queue(maxsize=180), encoded=0, peak_queue=0,
                                 encode_seconds=0., max_encode_seconds=0., bytes=int(image.nbytes),
                                 codec=self.codecs.get(rec['stream'], 'FFV1'))
                    state['video_file'] = key + ('.mp4' if state['codec'].startswith('H264') else '.avi')
                    state['thread'] = threading.Thread(target=self._run_video, args=(key, state), name='VideoWriter_'+key)
                    self.videos[key] = state
                    state['thread'].start()
                state = self.videos[key]
                try:
                    self.buffer_bytes += size
                    state['queue'].put_nowait((rec, image))
                except queue.Full:
                    self.buffer_bytes -= size
                    return self._reject(rec, f'{rec["stream"].upper()} 视频队列持续积压，已停止录像；详见逐路写入统计。', 'video_queue_full')
                self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)
                state['peak_queue'] = max(state['peak_queue'], state['queue'].qsize())
            self.accepted[key] = self.accepted.get(key, 0)+1
            return True

    @staticmethod
    def _send_end(target, thread):
        while thread.is_alive():
            try:
                target.put(None, timeout=.2)
                return
            except queue.Full:
                pass

    def close(self):
        with self.close_lock:
            with self.lock:
                if self.closed:
                    return
                self.accepting = False
                states = list(self.videos.values())
            # Finish video workers before stopping metadata, so all accepted frame
            # indices reach disk. A dead worker must never hang a full queue.
            for state in states:
                self._send_end(state['queue'], state['thread'])
            for state in states:
                state['thread'].join()
            self._send_end(self.items, self.thread)
            self.thread.join()
            # An IMU event can reach the metadata worker after a video record was
            # encoded. Reconcile final flags against the earliest event timestamp.
            if self.pose_stamp and self.first_motion_ns is not None:
                try:
                    path = self.root/'frames.jsonl'
                    temporary = self.root/'frames_finalizing.jsonl'
                    with path.open(encoding='utf8') as source, temporary.open('w', encoding='utf8') as target:
                        for line in source:
                            row = json.loads(line)
                            if row['host_monotonic_ns'] >= self.first_motion_ns:
                                row.update(pose_valid=False, pose_invalid_reason='camera_motion_alarm')
                            target.write(json.dumps(row)+'\n')
                    temporary.replace(path)
                except Exception as exc:
                    self._fail('Cannot finalize frame pose validity: '+str(exc), 'pose_metadata_finalize_failed')
            verification_started = time.monotonic()
            video_files = []
            for key, state in self.videos.items():
                try:
                    result = verify_video(self.root/state['video_file'], state['encoded'])
                    result['pixel_lossless'] = not state['codec'].startswith('H264')
                    video_files.append(result)
                except Exception as exc:
                    self._fail('Cannot verify saved video: '+str(exc), 'video_verify_failed')
            self.verification_seconds = time.monotonic()-verification_started
            if self.counts != self.accepted:
                self._fail('已接收样本与写入索引数量不一致；录像不完整，详见样本计数。', 'sample_count_mismatch')
            summary = dict(complete=not bool(self.error), error=self.error, error_category=self.error_category,
                           stop_reason=self.stop_reason, audio_status=self.audio_status,
                           received_sample_counts=dict(self.counts), accepted_sample_counts=dict(self.accepted),
                           rejected_sample=self.rejected_sample, frame_number_gaps=dict(self.gaps),
                           video_files=video_files, total_video_bytes=sum(v['bytes'] for v in video_files),
                           video_verification_seconds=self.verification_seconds,
                           finished_wall_ns=time.time_ns(), writer_performance=self.snapshot_stats())
            if self.pose_stamp:
                summary.update(pose_stamp_id=self.pose_stamp['pose_stamp_id'],
                               first_pose_motion_alarm_host_ns=self.first_motion_ns)
            (self.root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            self.closed = True

    def snapshot_stats(self):
        with self.lock:
            elapsed = max(.001, time.monotonic()-self.started)
            streams = {key: dict(queued=state['queue'].qsize(), encoded=state['encoded'],
                video_file=state['video_file'],
                peak_queued=state['peak_queue'], codec=state['codec'], saved_fps=state['encoded']/elapsed,
                mean_encode_ms=state['encode_seconds']*1000/max(1, state['encoded']),
                max_encode_ms=state['max_encode_seconds']*1000) for key, state in self.videos.items()}
            return dict(elapsed_seconds=elapsed, video_buffer_mb=self.buffer_bytes/1024/1024,
                        peak_video_buffer_mb=self.peak_buffer_bytes/1024/1024,
                        video_buffer_limit_mb=self.buffer_limit/1024/1024,
                        metadata_queued=self.items.qsize(), peak_metadata_queued=self.peak_metadata_items,
                        streams=streams, written_samples=dict(self.counts))

    def _run_video(self, key, state):
        writer = None
        try:
            while True:
                item = state['queue'].get()
                if item is None:
                    break
                rec, image = item
                try:
                    started = time.monotonic()
                    if writer is None:
                        h, w = image.shape[:2]
                        path = self.root/state['video_file']
                        if self.mode in ('balanced', 'mp4'):
                            writer = FFmpegWriter(path, state['codec'], rec['nominal_fps'], image.shape)
                        else:
                            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*state['codec']),
                                                     rec['nominal_fps'], (w, h), image.ndim == 3)
                            if not writer.isOpened():
                                raise RuntimeError('Cannot open video: '+key+' / '+state['codec'])
                    writer.write(image)
                    duration = time.monotonic()-started
                    rec.update(video_file=state['video_file'], video_index=state['encoded'],
                               pixel_lossless=not state['codec'].startswith('H264'))
                    with self.lock:
                        state['encoded'] += 1
                        state['encode_seconds'] += duration
                        state['max_encode_seconds'] = max(state['max_encode_seconds'], duration)
                    while self.thread.is_alive():
                        try:
                            self.items.put((rec, True), timeout=.2)
                            break
                        except queue.Full:
                            pass
                    else:
                        raise RuntimeError('Frame metadata writer has stopped')
                finally:
                    with self.lock:
                        self.buffer_bytes -= int(image.nbytes)
        except Exception as exc:
            self._fail(str(exc), 'video_write_failed')
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    self._fail(str(exc), 'video_finalize_failed')

    def _run_metadata(self):
        try:
            with (self.root/'frames.jsonl').open('w', encoding='utf8') as frames, \
                    (self.root/'imu.csv').open('w', newline='', encoding='utf8') as imu:
                csvout = csv.DictWriter(imu, fieldnames=IMU_FIELDS)
                csvout.writeheader()
                while True:
                    item = self.items.get()
                    if item is None:
                        break
                    rec, video = item
                    key = rec['serial']+'_'+rec['stream']
                    if not video:
                        if self.pose_stamp and rec['stream'] in ('accel', 'gyro') and motion_alarm(rec):
                            timestamp = int(rec['host_monotonic_ns'])
                            self.first_motion_ns = timestamp if self.first_motion_ns is None else min(self.first_motion_ns, timestamp)
                        csvout.writerow({field: rec.get(field, '') for field in IMU_FIELDS})
                    else:
                        if self.pose_stamp:
                            valid = self.first_motion_ns is None or rec['host_monotonic_ns'] < self.first_motion_ns
                            rec.update(pose_stamp_id=self.pose_stamp['pose_stamp_id'],
                                       camera_center_target_mm=self.pose_stamp['camera_center_target_mm'],
                                       pose_valid=valid, pose_source='fixed_target_calibration', pose_reference_stream='ir1')
                            if not valid:
                                rec['pose_invalid_reason'] = 'camera_motion_alarm'
                        frames.write(json.dumps(rec)+'\n')
                    with self.lock:
                        number = rec['frame_number']
                        if key in self.last_numbers:
                            self.gaps[key] = self.gaps.get(key, 0)+max(0, number-self.last_numbers[key]-1)
                        self.last_numbers[key] = number
                        self.counts[key] = self.counts.get(key, 0)+1
        except Exception as exc:
            self._fail(str(exc), 'metadata_write_failed')
