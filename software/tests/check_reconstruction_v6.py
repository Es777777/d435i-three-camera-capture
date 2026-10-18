"""Synthetic data tests; no hardware or microphone access."""
import copy
import json
import os
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5 import QtWidgets

from audio_io import AudioRecorder
from audio_alignment import envelope,estimate_audio_map
from stereo_geometry import PassiveStereo,project,extrinsics
from reconstruction import reconstruct_session,merge_reconstructions,read_rows,display_tracks,imu_motion_report
from calibration_settlement import verify_settlement_record
from pose_stamps import make_pose_stamp
from recording_io import SessionWriter
from sync_health import frame_health
from reconstruction_gui import ReconstructionWindow


def run():
    app=QtWidgets.QApplication([])
    rng=np.random.default_rng(42)
    k=dict(width=240,height=160,fx=200.,fy=200.,cx=120.,cy=80.,coeffs=[0.]*5,
           distortion='distortion.none',ir1_to_stream_rotation=np.eye(3).flatten(order='F').tolist(),
           ir1_to_stream_translation_m=[0.,0.,0.])
    kr=copy.deepcopy(k);kr['ir1_to_stream_translation_m']=[-.05,0.,0.]
    kc=copy.deepcopy(k)
    streams={'stream.infrared:1':k,'stream.infrared:2':kr,'stream.color:0':kc}
    stereo=PassiveStereo(dict(streams=streams))
    left=rng.integers(30,170,(160,240),dtype=np.uint8)
    for point in ((120,40),(140,80),(160,120)):cv2.circle(left,point,2,250,-1)
    right=np.zeros_like(left);right[:,:-10]=left[:,10:]
    rgb=np.zeros((160,240,3),np.uint8);rgb[:,:,2]=240
    result=stereo.reconstruct(left,right,rgb)
    assert len(result['xyz_mm'])>100
    assert abs(np.median(result['xyz_mm'][:,2])-1000)<3
    assert np.all(result['rgb'][result['color_valid']]==[240,0,0])
    assert len(result['markers_mm'])==3,result['markers_mm']
    assert np.allclose(result['markers_mm'][:,2],1000,atol=1)
    flat=stereo.reconstruct(np.full_like(left,100),np.full_like(left,100),rgb)
    assert len(flat['xyz_mm'])==0
    rot=cv2.Rodrigues(np.array([.03,-.08,.02]))[0]
    ke=copy.deepcopy(kr);ke['ir1_to_stream_rotation']=rot.flatten(order='F').tolist()
    assert np.allclose(extrinsics(ke)[0],rot)
    for model in ('none','brown_conrady','inverse_brown_conrady','modified_brown_conrady'):
        test_k=copy.deepcopy(k);test_k['distortion']='distortion.'+model
        test_k['coeffs']=[.01,-.002,.0001,-.0002,.0001] if model!='none' else [0.]*5
        intr=rs.intrinsics();intr.width=240;intr.height=160;intr.fx=200;intr.fy=200;intr.ppx=120;intr.ppy=80
        intr.coeffs=test_k['coeffs'];intr.model=getattr(rs.distortion,model)
        point=np.array([[.12,.15,1.]])
        if model=='inverse_brown_conrady':
            pixel=project(point,test_k)[0]
            back=rs.rs2_deproject_pixel_to_point(intr,pixel.tolist(),1.)
            assert np.allclose(back,point[0],atol=1e-5),(model,back,point)
        else:
            expected=rs.rs2_project_point_to_pixel(intr,point[0].tolist())
            assert np.allclose(project(point,test_k)[0],expected,atol=1e-4)
    health=frame_health({},1,30)
    assert health['cable_connected'] is None and health['streams']['ir1']['state']=='no_frames'
    gap=[dict(time_ms=0,markers=[dict(id='A',xyz_mm=[0,0,1000])]),
         dict(time_ms=100,markers=[]),dict(time_ms=200,markers=[dict(id='A',xyz_mm=[2,0,1000])])]
    display_tracks(gap)
    assert gap[1]['markers']==[] and gap[1]['display_markers'][0]['source']=='interpolated'

    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        imu_root=root/'imu_test';imu_root.mkdir()
        assert not imu_motion_report(imu_root)['checked']
        (imu_root/'imu.csv').write_text('stream,x,y,z,host_monotonic_ns\ngyro,0,0,0,100\ngyro,0.1,0,0,200\n',encoding='utf8')
        assert imu_motion_report(imu_root)['first_motion_host_ns']==200
        # Exercise callback timestamping and WAV writing without touching an input device.
        audio_root=root/'audio_test';audio_root.mkdir()
        recorder=AudioRecorder(audio_root)
        recorder._callback(np.zeros(1024,dtype='<i2').tobytes(),1024,
                           SimpleNamespace(inputBufferAdcTime=1.,currentTime=1.02),'')
        recorder.queue.put(None);recorder._write()
        with wave.open(str(audio_root/'audio.wav')) as wav:assert wav.getnframes()==1024
        audio_rec=read_rows(audio_root/'audio_blocks.jsonl')[0]
        assert abs(audio_rec['host_monotonic_ns']-audio_rec['first_sample_host_ns']-20000000)<2
        rate=48000;duration=12
        # Non-repeating syllable-like modulation shared by all three microphones.
        control=rng.uniform(.02,.8,150)
        t=np.arange(rate*15)/rate
        amp=np.interp(t,np.arange(len(control))*.1,control)
        signal=(np.sin(t*2*np.pi*350)+.3*np.sin(t*2*np.pi*731))*amp*.6
        clips=[signal[int(off*rate):int(off*rate)+rate*duration] for off in (0.,.2,.4)]
        mapping=estimate_audio_map(envelope(clips[0],rate),envelope(clips[1],rate))
        assert abs(mapping['offset_s']-.2)<.005,mapping
        reverse=estimate_audio_map(envelope(clips[1],rate),envelope(clips[0],rate))
        assert abs(reverse['offset_s']+.2)<.005,reverse
        base=envelope(signal,rate)
        drifted=np.interp(np.arange(12000)*1.001+200,np.arange(len(base)),base)
        drift=estimate_audio_map(base,drifted)
        assert abs(drift['scale']-1.001)<.0005 and abs(drift['offset_s']-.2)<.005,drift
        try:
            estimate_audio_map(np.zeros(12000),np.zeros(12000))
            raise AssertionError('Silence must fail alignment')
        except ValueError:pass
        target=[[-30,-30,1000],[30,-30,1000],[-30,30,1000],[30,30,1000],[-30,0,980],[30,0,1020]]
        files=[]
        for i in range(3):
            T=np.eye(4);T[0,3]=i*20
            pose=dict(serial=str(i),stream='ir1',target='test',target_scale=1.,marker_inset_mm=0.,units='mm',
                      intrinsics=k,T_camera_from_target_mm=T.tolist(),
                      point_ids=list('ABCDEF'),target_xyz_mm=target,per_point_error_px=[.1]*6,
                      undistorted_reprojection_rms_px=.1)
            path=root/f'cal_{i}.json';path.write_text(json.dumps(pose),encoding='utf8');files.append(path)
        folders=[];local=[]
        alternate_id=None
        for i in range(3):
            folder=root/f'session_{i}';folders.append(folder)
            role=('master','slave1','slave2')[i]
            pose=json.loads(files[i].read_text(encoding='utf8'))
            meta=dict(serial=str(i),streams=streams,calibration=pose,
                      sync_role=role,sync_mode=1 if i==0 else 2,started_wall_ns=i+1)
            writer=SessionWriter(folder,meta)
            for n in range(60):
                for name,im in [('ir1',left),('ir2',right),('rgb',rgb)]:
                    rec=dict(serial=str(i),stream=name,frame_number=n,device_ts_ms=1000*(i+1)+n*200,
                             host_monotonic_ns=int((100*(i+1)+n*.2)*1e9),host_wall_ns=n,
                             timestamp_domain='hardware_clock',nominal_fps=5)
                    # Test writer producer throttling is separate from production overflow handling.
                    while writer.items.qsize()>60:
                        import time;time.sleep(.001)
                    assert writer.submit(rec,im)
            writer.close();assert not writer.error
            recorded=json.loads((folder/'session.json').read_text(encoding='utf8'))
            assert 'rig_id' not in recorded and 'rig_calibration' not in recorded
            if i==2:
                # Exercise a later pose snapshot: fusion must use each source
                # frame's stamp, rather than the session's initial transform.
                updated=copy.deepcopy(pose)
                updated['T_camera_from_target_mm'][0][3]=240.
                stamp=make_pose_stamp(updated);alternate_id=stamp['pose_stamp_id']
                registry=json.loads((folder/'pose_stamps.json').read_text(encoding='utf8'))
                registry['stamps'][alternate_id]=stamp
                (folder/'pose_stamps.json').write_text(json.dumps(registry),encoding='utf8')
                source_rows=read_rows(folder/'frames.jsonl')
                for row in source_rows:
                    if row['frame_number']>=30:
                        row.update(pose_stamp_id=alternate_id,camera_center_target_mm=stamp['camera_center_target_mm'])
                (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in source_rows),encoding='utf8')
            with wave.open(str(folder/'audio.wav'),'wb') as wav:
                wav.setparams((1,2,rate,0,'NONE','not compressed'))
                wav.writeframes((clips[i]*30000).astype('<i2').tobytes())
            (folder/'audio.json').write_text(json.dumps(dict(rate=rate,complete=True)),encoding='utf8')
            with (folder/'audio_blocks.jsonl').open('w',encoding='utf8') as f:
                for n in range(0,rate*duration,1024):
                    f.write(json.dumps(dict(sample_start=n,first_sample_host_ns=int((100*(i+1)+n/rate)*1e9),status=''))+'\n')
            out=root/f'local_{i}'
            reconstruct_session(folder,out,dict(cloud_stride=8),progress=lambda x:None)
            local.append(out)
            records=read_rows(out/'frames_3d.jsonl')
            assert all(len(r['markers'])==3 for r in records)
        out=root/'fused'
        merge_reconstructions(local,out,dict(max_time_error_ms=10))
        fused=json.loads((out/'result.json').read_text(encoding='utf8'))
        assert fused['complete'] and not fused['exposure_sync_verified']
        assert fused['settlement_source']=='recorded_frame_poses' and len(fused['rig_ids'])==2
        rigs=json.loads((out/'rig_calibrations.json').read_text(encoding='utf8'))
        for rig in rigs.values(): verify_settlement_record(rig)
        frames=read_rows(out/'frames_3d.jsonl')
        assert len(frames)>40 and all(len(r['used_views'])==3 for r in frames)
        later=[r for r in frames if r['source_indices'][2]>=30]
        assert later and all(r['pose_stamp_ids'][2]==alternate_id for r in later)
        assert all(any(np.allclose(m['xyz_mm'],[-240.,-200.,1000.],atol=1.) for m in r['markers']) for r in later)
        assert len(list((out/'sources').glob('*/*.avi')))==6
        assert len(list((out/'sources').glob('*/*.mp4')))==3
        assert len(list(out.glob('aligned_audio_*.wav')))==3
        win=ReconstructionWindow();win.show();app.processEvents()
        win.load_result(str(out));app.processEvents()
        assert win.frames and len(win.cloud.xyz)>0
        win.grab().save(str(Path(__file__).parent/'reconstruction_v6_check.png'))
        win.close()
    print('PASS v6: passive stereo scale, RGB colors, marker triangulation, flat rejection, SDK distortion, '
          'audio ADC/WAV, offset estimation, local reconstruction, three-camera fusion, nine videos, three audio tracks, replay UI')


if __name__=='__main__':run()
