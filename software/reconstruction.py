"""Offline local passive stereo and three-session fusion with replayable output."""
import json
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from stereo_geometry import PassiveStereo
from calibration_settlement import calibration_digest, settle_recorded_calibrations
from pose_stamps import read_pose_stamps
from audio_alignment import (read_audio, envelope, estimate_audio_map, audio_host_map,
                             camera_host_map, aligned_audio)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def read_rows(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def imu_motion_report(folder):
    path = Path(folder)/'imu.csv'
    first = None
    count = 0
    if path.exists():
        with path.open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                norm = float(np.linalg.norm([float(row[k]) for k in ('x','y','z')]))
                count += 1
                alarm = (row['stream']=='gyro' and norm>.05) or (row['stream']=='accel' and abs(norm-9.80665)>1.)
                if alarm:
                    t = int(row['host_monotonic_ns'])
                    first = t if first is None else min(t,first)
    return dict(samples=count, first_motion_host_ns=first, checked=count>0,
                note='Motion alarm only; no translation correction or proof of stationary camera.')


class Session:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.meta = json.loads((self.folder/'session.json').read_text(encoding='utf8'))
        self.pose_stamps = read_pose_stamps(self.folder, self.meta) if self.meta.get('calibration') else {}
        summary = json.loads((self.folder/'summary.json').read_text(encoding='utf8'))
        if not summary.get('complete'):
            raise ValueError('采集未完整结束：'+str(summary.get('error')))
        rows = read_rows(self.folder/'frames.jsonl')
        self.rows, self.times, self.caps = {}, {}, {}
        for stream in ('ir1','ir2','rgb'):
            r = sorted([v for v in rows if v['stream']==stream],key=lambda v:v['video_index'])
            if not r:
                raise ValueError('缺少流 '+stream)
            ts = np.array([v['device_ts_ms'] for v in r])
            if len({v['timestamp_domain'] for v in r})!=1 or np.any(np.diff(ts)<=0):
                raise ValueError('时间域变化或设备时钟重置：'+stream)
            if [v['video_index'] for v in r] != list(range(len(r))):
                raise ValueError('视频索引不连续：'+stream)
            for row in r:
                key = row.get('pose_stamp_id', self.meta.get('pose_stamp_id'))
                if key is not None and key not in self.pose_stamps:
                    raise ValueError('视频帧引用了不存在的本机位姿戳')
            self.rows[stream], self.times[stream] = r, ts

    def image(self, stream, index):
        row = self.rows[stream][index]
        path = (self.folder/row['video_file']).resolve()
        if path.parent != self.folder:
            raise ValueError('视频路径不在 session 目录内')
        if stream not in self.caps:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError('无法打开 '+str(path))
            self.caps[stream] = cap
        cap = self.caps[stream]
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok, im = cap.read()
        if not ok:
            raise ValueError(f'{stream} 无法解码帧 {index}')
        return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) if stream!='rgb' else im

    def nearest(self, stream, reference, tolerance_ms):
        if self.rows[stream][0]['timestamp_domain']!=reference['timestamp_domain']:
            return None
        times = self.times[stream]
        j = int(np.searchsorted(times,reference['device_ts_ms']))
        j = min((max(0,j-1),min(len(times)-1,j)),key=lambda k:abs(times[k]-reference['device_ts_ms']))
        return j if abs(times[j]-reference['device_ts_ms'])<=tolerance_ms else None

    def close(self):
        for cap in self.caps.values():
            cap.release()


class MarkerTracker:
    def __init__(self, max_distance_mm=25., max_gap_ms=200.):
        self.tracks = {}
        self.counter = 0
        self.distance, self.gap = max_distance_mm, max_gap_ms

    def update(self, points, time_ms):
        points = np.asarray(points,float).reshape(-1,3)
        self.tracks = {k:v for k,v in self.tracks.items() if 0<=time_ms-v[1]<=self.gap}
        ids, assigned = list(self.tracks), {}
        if ids and len(points):
            old = np.array([self.tracks[k][0] for k in ids])
            cost = np.linalg.norm(old[:,None,:]-points[None,:,:],axis=2)
            a,b = linear_sum_assignment(cost)
            for i,j in zip(a,b):
                competitors = np.delete(cost[:,j],i)
                alternatives = np.delete(cost[i],j)
                ambiguous = (len(competitors) and competitors.min()<cost[i,j]+2) or \
                            (len(alternatives) and alternatives.min()<cost[i,j]+2)
                if cost[i,j]<=self.distance and not ambiguous:
                    assigned[j] = ids[i]
        result = []
        for j, point in enumerate(points):
            pid = assigned.get(j)
            if pid is None:
                self.counter += 1
                pid = f'M{self.counter:04d}'
            self.tracks[pid] = (point,time_ms)
            result.append(dict(id=pid,xyz_mm=point.tolist()))
        return result


def display_tracks(frames, gap_ms=200., smoothing_ms=60.):
    """Bounded interpolation and time-aware EMA; measured coordinates remain unchanged."""
    by_id = {}
    for i,frame in enumerate(frames):
        frame['display_markers'] = []
        for m in frame['markers']:
            by_id.setdefault(m['id'],[]).append((i,np.array(m['xyz_mm'],float)))
    for pid, obs in by_id.items():
        previous = None
        for n,(i,xyz) in enumerate(obs):
            t = frames[i]['time_ms']
            if previous is not None and t-previous[0]<=gap_ms:
                alpha = 1-np.exp(-(t-previous[0])/max(1.,smoothing_ms))
                shown = previous[1]+alpha*(xyz-previous[1])
            else:
                shown = xyz
            frames[i]['display_markers'].append(dict(id=pid,xyz_mm=shown.tolist(),source='smoothed'))
            if n:
                old_i, old_xyz = obs[n-1]
                old_t = frames[old_i]['time_ms']
                if 0<t-old_t<=gap_ms:
                    for j in range(old_i+1,i):
                        f = (frames[j]['time_ms']-old_t)/(t-old_t)
                        p = previous[1]*(1-f)+shown*f
                        frames[j]['display_markers'].append(dict(id=pid,xyz_mm=p.tolist(),source='interpolated'))
            previous = (t,shown)


def prepare_output(output, sources):
    output = Path(output).resolve()
    for s in sources:
        if output==Path(s).resolve() or Path(s).resolve() in output.parents:
            raise ValueError('输出目录必须在输入 session/重建目录之外，避免混入原始数据')
    output.mkdir(parents=True,exist_ok=False)
    (output/'clouds').mkdir()
    write_json(output/'result.json',dict(complete=False,status='processing'))
    return output


def reconstruct_session(folder, output, options=None, progress=lambda x:None, cancel=lambda:False):
    options = options or {}
    session = Session(folder)
    output = prepare_output(output,[folder])
    frames = []
    try:
        stereo = PassiveStereo(session.meta, options.get('near_mm',300.),options.get('far_mm',1800.),
                               options.get('threshold',210))
        tracker = MarkerTracker()
        motion = imu_motion_report(folder)
        indices = range(0,len(session.rows['ir1']),max(1,int(options.get('frame_step',1))))
        rgb_missing, stereo_missing = 0,0
        for n,i in enumerate(indices):
            if cancel():
                raise InterruptedError('处理已取消')
            row = session.rows['ir1'][i]
            j = session.nearest('ir2',row,options.get('stereo_tolerance_ms',2.))
            c = session.nearest('rgb',row,options.get('rgb_tolerance_ms',20.))
            frame = dict(time_ms=row['device_ts_ms'],host_monotonic_ns=row['host_monotonic_ns'],
                         source_indices=dict(ir1=i,ir2=j,rgb=c),markers=[],cloud=None)
            frame['rig_pose_valid'] = motion['first_motion_host_ns'] is None or row['host_monotonic_ns'] < motion['first_motion_host_ns']
            frame['rig_pose_valid'] = frame['rig_pose_valid'] and row.get('pose_valid', True)
            frame['pose_stamp_id'] = row.get('pose_stamp_id', session.meta.get('pose_stamp_id'))
            if j is None:
                stereo_missing += 1
                frame['status']='missing_stereo_pair'
            else:
                color = session.image('rgb',c) if c is not None else None
                rgb_missing += int(color is None)
                result = stereo.reconstruct(session.image('ir1',i),session.image('ir2',j),color,
                                            options.get('roi'),options.get('cloud_stride',4))
                filename = f'clouds/{n:08d}.npz'
                np.savez_compressed(output/filename,xyz_mm=result['xyz_mm'],rgb=result['rgb'],
                                    color_valid=result['color_valid'],depth_mm=result['depth_mm'])
                frame.update(cloud=filename,markers=tracker.update(result['markers_mm'],row['device_ts_ms']),
                             valid_fraction=result['valid_fraction'],status='processed')
                for m,e in zip(frame['markers'],result['marker_errors_px']):
                    m['reprojection_error_px']=float(e)
                if n==0:
                    cv2.imwrite(str(output/'rectified_ir1.png'),result['rectified_ir1'])
                    cv2.imwrite(str(output/'rectified_ir2.png'),result['rectified_ir2'])
            frames.append(frame)
            if n%10==0:
                progress(f'单机 {n+1}/{len(indices)}；有效 Mark {len(frame["markers"])}')
        display_tracks(frames,options.get('gap_ms',200.),options.get('smoothing_ms',60.))
        with (output/'frames_3d.jsonl').open('w',encoding='utf8') as f:
            for frame in frames:
                f.write(json.dumps(frame,allow_nan=False)+'\n')
        meta = dict(schema='d435i_local_reconstruction_v1',complete=True,coordinate_frame='ir1',units='mm',
                    source_session=str(session.folder),session_metadata=session.meta,options=options,
                    frames=len(frames),missing_stereo_pairs=stereo_missing,missing_rgb_pairs=rgb_missing,
                    observed_marker_samples=sum(len(f['markers']) for f in frames),
                    imu_motion=motion,
                    pose_stamps=session.pose_stamps,
                    note='Local IDs are geometric tracks, not anatomical identities. No IMU translation correction.')
        write_json(output/'result.json',meta)
        return output
    except Exception as exc:
        write_json(output/'result.json',dict(complete=False,error=str(exc)))
        raise
    finally:
        session.close()


def transform(xyz,T):
    xyz = np.asarray(xyz).reshape(-1,3)
    return xyz@T[:3,:3].T+T[:3,3]


def fuse_points(markers, gate_mm=12.):
    """Only cluster across views; never merge two detections from the same camera."""
    clusters=[]
    for view,points in enumerate(markers):
        for point in points:
            distances=[np.linalg.norm(np.mean(c['points'],axis=0)-point) if view not in c['views'] else np.inf
                       for c in clusters]
            eligible=[i for i,d in enumerate(distances) if d<gate_mm]
            if len(eligible)==1:
                c=clusters[eligible[0]]
                c['points'].append(point)
                c['views'].append(view)
            else:
                clusters.append(dict(points=[point],views=[view]))
    return [dict(xyz=np.median(c['points'],axis=0),views=c['views'],
                 spread_mm=float(np.max(np.linalg.norm(np.array(c['points'])-np.median(c['points'],axis=0),axis=1))))
            for c in clusters]


def merge_reconstructions(folders, output, options=None, progress=lambda x:None, cancel=lambda:False):
    options=options or {}
    if len(folders)!=3:
        raise ValueError('需要三份本机重建目录')
    folders=[Path(p).resolve() for p in folders]
    metas=[json.loads((p/'result.json').read_text(encoding='utf8')) for p in folders]
    if any(not m.get('complete') or m.get('schema')!='d435i_local_reconstruction_v1' for m in metas):
        raise ValueError('需要已处理完成的三份单机结果')
    sessions=[Path(m['source_session']) for m in metas]
    # If moved together, the user may override the three source session paths.
    if options.get('sessions'):
        sessions=[Path(p).resolve() for p in options['sessions']]
    sm=[m['session_metadata'] for m in metas]
    if len({m['serial'] for m in sm})!=3:
        raise ValueError('相机序列号重复')
    if {m.get('sync_role') for m in sm}!={'master','slave1','slave2'}:
        raise ValueError('必须导入 Master、Slave1、Slave2 各一份')
    registries=[]
    for folder,m in zip(sessions,sm):
        actual=json.loads((folder/'session.json').read_text(encoding='utf8'))
        if actual['serial']!=m['serial'] or actual['started_wall_ns']!=m['started_wall_ns']:
            raise ValueError('原始 session 与本机重建不属于同一次采集')
        if calibration_digest(actual['calibration'])!=calibration_digest(m['calibration']):
            raise ValueError('原始 session 与本机重建的标定快照不一致')
        registries.append(read_pose_stamps(folder,actual))
    defaults=[m.get('pose_stamp_id') or calibration_digest(m['calibration']) for m in sm]
    settle_recorded_calibrations([registries[i][key]['calibration'] for i,key in enumerate(defaults)])
    output=prepare_output(output,folders+sessions)
    try:
        rows=[read_rows(p/'frames_3d.jsonl') for p in folders]
        if any(not records for records in rows):
            raise ValueError('本机重建结果没有可合成的帧')
        ref=next(i for i,m in enumerate(sm) if m['sync_role']=='master')
        envelopes=[]
        for session in sessions:
            pcm,rate=read_audio(session/'audio.wav')
            envelopes.append(envelope(pcm,rate))
        audio_maps=[dict(scale=1.,offset_s=0.) if i==ref else estimate_audio_map(envelopes[ref],e)
                    for i,e in enumerate(envelopes)]
        timeline=[]
        clock_reports=[]
        settlement_cache={}
        for i,(folder,records,mapping) in enumerate(zip(sessions,rows,audio_maps)):
            audio_clock=audio_host_map(folder)
            video_records=[r for r in read_rows(folder/'frames.jsonl') if r['stream']=='ir1']
            originals={r['video_index']:r for r in video_records}
            motion=imu_motion_report(folder)
            for record in records:
                original=originals[record['source_indices']['ir1']]
                key=original.get('pose_stamp_id') or defaults[i]
                if key not in registries[i] or record.get('pose_stamp_id',key) not in (None,key):
                    raise ValueError('重建帧与原始视频的位姿戳不一致')
                record['_pose_key']=key
                record['_pose_valid']=original.get('pose_valid',True) and (
                    motion['first_motion_host_ns'] is None or original['host_monotonic_ns']<motion['first_motion_host_ns'])
            video_clock=camera_host_map(video_records)
            device_s=np.array([r['time_ms']/1000 for r in records])
            local_host=device_s*video_clock['slope']+video_clock['intercept_s']
            audio_s=(local_host-audio_clock['intercept_s'])/audio_clock['slope']
            timeline.append((mapping['scale']*audio_s+mapping['offset_s'])*1000)
            clock_reports.append(dict(audio=audio_clock,video=video_clock))
        # Optional two visible-event anchors supersede audio-assisted video time mapping.
        anchors=options.get('video_anchors')
        if anchors:
            if len(anchors)!=3 or any(len(a)!=2 or not 0<=a[0]<a[1]<len(rows[i]) for i,a in enumerate(anchors)):
                raise ValueError('每个机位需要两个有效且递增的视频事件索引')
            reference_times=np.array([timeline[ref][k] for k in anchors[ref]])
            for i in range(3):
                a,b=anchors[i]
                if not 0<=a<b<len(rows[i]):
                    raise ValueError('两次视频事件索引无效')
                t=np.array([r['time_ms'] for r in rows[i]])
                scale=(reference_times[1]-reference_times[0])/(t[b]-t[a])
                timeline[i]=(t-t[a])*scale+reference_times[0]
        progress('音频与时间轴已匹配，正在融合点云和轨迹')
        start=max(t[0] for t in timeline)
        end=min(t[-1] for t in timeline)
        if end<=start:
            raise ValueError('没有共同采集时间区间')
        tracker=MarkerTracker()
        frames=[]
        tolerance=float(options.get('max_time_error_ms',10.))
        for idx,time_ms in enumerate(timeline[ref]):
            if time_ms<start or time_ms>end:
                continue
            if cancel():
                raise InterruptedError('合成已取消')
            clouds,colors,marks,used,source_indices=[],[],[],[],[]
            residuals=[]
            pose_ids=[]
            for i,t in enumerate(timeline):
                k=int(np.searchsorted(t,time_ms))
                k=min((max(0,k-1),min(len(t)-1,k)),key=lambda j:abs(t[j]-time_ms))
                residual=float(abs(t[k]-time_ms))
                source_indices.append(k)
                residuals.append(residual)
                record=rows[i][k]
                key=record['_pose_key']
                pose_ids.append(key)
                T=np.asarray(registries[i][key]['T_target_from_camera_mm'],float)
                if residual>tolerance or not record.get('rig_pose_valid',True) or not record['_pose_valid']:
                    marks.append([])
                    continue
                used.append(i)
                if record['cloud']:
                    with np.load(folders[i]/record['cloud']) as data:
                        clouds.append(transform(data['xyz_mm'],T))
                        colors.append(data['rgb'])
                points=[m['xyz_mm'] for m in record['markers']]
                marks.append(transform(points,T) if points else [])
            combination=tuple(pose_ids)
            if combination not in settlement_cache:
                settlement_cache[combination]=settle_recorded_calibrations(
                    [registries[i][key]['calibration'] for i,key in enumerate(pose_ids)])
            rig=settlement_cache[combination]
            merged=fuse_points(marks,options.get('spatial_gate_mm',12.))
            pts=np.array([m['xyz'] for m in merged]).reshape(-1,3)
            tracked=tracker.update(pts,float(time_ms-start))
            for m,c in zip(tracked,merged):
                m.update(views=c['views'],spread_mm=c['spread_mm'])
            cloud=np.concatenate(clouds) if clouds else np.empty((0,3))
            color=np.concatenate(colors) if colors else np.empty((0,3),np.uint8)
            if len(cloud):
                _,keep=np.unique(np.floor(cloud/2).astype(int),axis=0,return_index=True)
                cloud,color=cloud[keep],color[keep]
            name=f'clouds/{len(frames):08d}.npz'
            np.savez_compressed(output/name,xyz_mm=cloud.astype(np.float32),rgb=color)
            frames.append(dict(time_ms=float(time_ms-start),markers=tracked,cloud=name,
                               used_views=used,source_indices=source_indices,time_residual_ms=residuals,
                               pose_stamp_ids=pose_ids,rig_id=rig['rig_id']))
            if len(frames)%10==0:
                progress(f'已融合 {len(frames)} 帧；本帧使用 {len(used)} 个机位')
        display_tracks(frames,options.get('gap_ms',200.),options.get('smoothing_ms',60.))
        with (output/'frames_3d.jsonl').open('w',encoding='utf8') as f:
            for frame in frames:
                f.write(json.dumps(frame,allow_nan=False)+'\n')
        aligned_audio(sessions,audio_maps,start/1000,end/1000,output,cancel)
        sources=[]
        for i,folder in enumerate(sessions):
            dst=output/'sources'/f'camera_{i+1}'
            dst.mkdir(parents=True)
            for file in folder.iterdir():
                if cancel():
                    raise InterruptedError('合成已取消')
                if file.is_file() and file.suffix.lower() in ('.avi','.mp4','.mkv','.wav','.json','.jsonl','.csv'):
                    shutil.copy2(file,dst/file.name)
            sources.append(dict(serial=sm[i]['serial'],role=sm[i]['sync_role'],path=str(dst.relative_to(output))))
        rigs={record['rig_id']:record for record in settlement_cache.values()}
        write_json(output/'rig_calibrations.json',rigs)
        if len(rigs)==1:
            write_json(output/'rig_calibration.json',next(iter(rigs.values())))
        write_json(output/'result.json',dict(schema='d435i_fused_reconstruction_v1',complete=True,
            coordinate_frame='target',units='mm',rig_id=next(iter(rigs)) if len(rigs)==1 else None,
            rig_ids=list(rigs),settlement_source='recorded_frame_poses',
            frames=len(frames),sources=sources,
            audio_maps=audio_maps,clock_reports=clock_reports,reference_camera=ref,
            reference_audio_start_s=float(start/1000),options=options,
            exposure_sync_verified=False,
            note='Audio correlation does not prove IR/RGB exposure lock. Sources retain all nine videos and three audio tracks.'))
        return output
    except Exception as exc:
        write_json(output/'result.json',dict(complete=False,error=str(exc)))
        raise
