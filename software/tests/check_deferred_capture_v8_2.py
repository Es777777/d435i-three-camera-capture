"""Local calibration enables recording; poses survive until deferred settlement."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from calibration_settlement import settle_recorded_calibrations, verify_settlement_record
from capture_app import Main
from capture_runtime import CaptureWorker
from pose_stamps import make_pose_stamp, read_pose_stamps
from recording_io import SessionWriter


def run():
    app = QtWidgets.QApplication([])
    k = dict(width=64, height=48, fx=60., fy=60., cx=32., cy=24.,
             coeffs=[0.]*5, distortion='distortion.none')
    T = np.eye(4)
    T[2, 3] = 1000.
    pose = dict(serial='demo', stream='ir1', target='fixed_target', target_scale=.7,
                marker_inset_mm=.55, units='mm', intrinsics=k,
                T_camera_from_target_mm=T.tolist(), T_target_from_camera_mm=np.linalg.inv(T).tolist(),
                camera_center_target_mm=[0., 0., -1000.], point_ids=list('ABCDEF'),
                target_xyz_mm=[[-10,0,0],[10,0,0],[0,-10,0],[0,10,0],[0,0,-10],[0,0,10]],
                per_point_error_px=[.2]*6, undistorted_reprojection_rms_px=.2)
    metadata = dict(serial='demo', ir1_intrinsics=k, sync_role='master', sync_mode=1,
                    emitter_enabled=0)

    class Preview:
        def __init__(self): self.commands=[]
        def command(self,*args): self.commands.append(args)
        def snapshot(self): return {}

    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        main=Main({'sync_role':'master'})
        preview=Preview()
        main.worker=preview
        main.camera_ready(metadata)
        main.path.setText(str(root/'calibrations'))
        main.show()
        app.processEvents()
        assert not main.record_button.isEnabled()
        main.save_calibration_pose(deepcopy(pose))
        assert main.calibration is not None and main.rig_calibration is None
        assert main.record_button.isEnabled(), main.recording_hint.text()
        QtTest.QTest.mouseClick(main.record_button,QtCore.Qt.LeftButton)
        assert len(preview.commands)==1 and preview.commands[0][0]=='record'
        assert preview.commands[0][3] is None, 'No three-camera settlement is supplied'
        main.recording_changed(True,'test')
        assert not main.record_button.isEnabled()
        main.recording_changed(False,'test')
        assert main.record_button.isEnabled()
        # Changed local calibration remains recordable without an old rig.
        main.save_calibration_pose(deepcopy(pose))
        assert main.record_button.isEnabled() and main.rig_calibration is None
        artifact=Path(__file__).resolve().parent/'artifacts'
        artifact.mkdir(exist_ok=True)
        main.grab().save(str(artifact/'deferred_recording_v8_2.png'))
        main.camera_finished()
        main.recording_changed(False,'late stop signal')
        assert not main.record_button.isEnabled(), 'A late signal must not enable recording after disconnection'
        main.close()

        worker=CaptureWorker({})
        worker.metadata=deepcopy(metadata)
        worker.serial='demo'
        bound=worker.recording_metadata(pose)
        assert 'rig_calibration' not in bound and 'rig_id' not in bound
        assert bound['calibration']==pose
        changed=deepcopy(pose);changed['serial']='other'
        try:
            worker.recording_metadata(changed)
            raise AssertionError('Wrong camera calibration accepted')
        except ValueError: pass
        changed=deepcopy(pose);changed['diagnostic_only']=True
        try:
            worker.recording_metadata(changed)
            raise AssertionError('Rejected diagnostic pose accepted')
        except ValueError: pass

        writer=SessionWriter(root/'session',bound)
        bound['calibration']['point_ids'][0]='changed-after-start'
        for n in range(4):
            imu=dict(serial='demo',stream='gyro',frame_number=n,device_ts_ms=n*100.,
                     timestamp_domain='hardware_clock',host_monotonic_ns=n*100,
                     host_wall_ns=n,x=.1 if n==2 else 0.,y=0.,z=0.,units='rad/s')
            assert writer.submit(imu)
            for stream in ('rgb','ir1','ir2'):
                rec=dict(serial='demo',stream=stream,frame_number=n,device_ts_ms=n*100.,
                         timestamp_domain='hardware_clock',host_monotonic_ns=n*100+1,
                         host_wall_ns=n,nominal_fps=10)
                image=np.zeros((48,64,3),np.uint8) if stream=='rgb' else np.zeros((48,64),np.uint8)
                assert writer.submit(rec,image)
        writer.close()
        assert writer.error is None,writer.error
        meta=json.loads((writer.root/'session.json').read_text(encoding='utf8'))
        stamps=read_pose_stamps(writer.root,meta)
        assert list(stamps)==[make_pose_stamp(pose)['pose_stamp_id']]
        assert meta['calibration']==pose and meta['settlement_status']=='deferred_until_fusion'
        rows=[json.loads(line) for line in (writer.root/'frames.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(rows)==12 and all(row['pose_stamp_id'] in stamps for row in rows)
        assert all(row['pose_valid']==(row['frame_number']<2) for row in rows)
        assert all(row['camera_center_target_mm']==[0.,0.,-1000.] for row in rows)
        report=json.loads((writer.root/'summary.json').read_text(encoding='utf8'))
        assert report['first_pose_motion_alarm_host_ns']==200
        # Independent encoders may finish a video before an older IMU event is
        # delivered. Final frame flags must still honor that event's timestamp.
        late=SessionWriter(root/'late_imu',worker.recording_metadata(pose))
        assert late.submit(dict(serial='demo',stream='ir1',frame_number=0,nominal_fps=30,
                                host_monotonic_ns=100),np.zeros((48,64),np.uint8))
        deadline=time.monotonic()+3
        while late.snapshot_stats()['written_samples'].get('demo_ir1',0)<1 and time.monotonic()<deadline:
            time.sleep(.01)
        assert late.counts.get('demo_ir1')==1
        assert late.submit(dict(serial='demo',stream='gyro',frame_number=0,host_monotonic_ns=50,x=.1,y=0.,z=0.))
        late.close()
        late_row=json.loads((late.root/'frames.jsonl').read_text(encoding='utf8'))
        assert not late.error and not late_row['pose_valid']
        doc=json.loads((writer.root/'pose_stamps.json').read_text(encoding='utf8'))
        doc['stamps'][meta['pose_stamp_id']]['T_target_from_camera_mm'][0][3]+=1e-10
        (writer.root/'pose_stamps.json').write_text(json.dumps(doc),encoding='utf8')
        assert read_pose_stamps(writer.root,meta), 'Harmless cross-CPU roundoff must be accepted'
        doc['stamps'][meta['pose_stamp_id']]['camera_center_target_mm'][0]=99.
        (writer.root/'pose_stamps.json').write_text(json.dumps(doc),encoding='utf8')
        try:
            read_pose_stamps(writer.root,meta)
            raise AssertionError('Inconsistent pose stamp accepted')
        except ValueError: pass

        poses=[]
        for n in range(3):
            p=deepcopy(pose);p['serial']=f'camera-{n}'
            poses.append(p)
        settled=settle_recorded_calibrations(poses)
        assert settled['validation_passed'] and settled['settlement_source']=='recorded_frame_poses'
        assert not settled['operator_confirmed_geometry'] and not settled['operator_confirmed_same_target_pose']
        verify_settlement_record(settled)
    print('PASS v8.2: local-only Qt recording, backend accepts no rig, invalid local pose rejected; '
          'immutable pose snapshots and frame links, motion flags, tamper rejection and deferred settlement')


if __name__=='__main__': run()
