"""Inspect finalized sessions and export ordinary MP4 copies without modifying sources."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

from video_encoding import executable, hidden_process_options, verify_video


def inspect_session(folder, full=False):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError('录像文件夹不存在：'+str(root))
    report = dict(folder=str(root), ok=True, warnings=[], files=[])
    summary_path = root/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf8'))
        report['recording_complete'] = summary.get('complete')
        report['frame_number_gaps'] = summary.get('frame_number_gaps', {})
        if not summary.get('complete'):
            report['ok'] = False
            report['warnings'].append('采集报告标为不完整：'+str(summary.get('error')))
        if any(report['frame_number_gaps'].values()):
            report['warnings'].append('原始帧号存在缺口，视频可播放不等于采集无丢帧。')
    else:
        report['recording_complete'] = None
        report['ok'] = False
        report['warnings'].append('未找到 summary.json；可能仍在录像或未正常结束，无法确认保存完整。')
    expected = defaultdict(int)
    indices_ok = True
    index_path = root/'frames.jsonl'
    if index_path.exists():
        with index_path.open(encoding='utf8') as handle:
            for line in handle:
                row = json.loads(line)
                name = row['video_file']
                path = (root/name).resolve()
                if path.parent != root:
                    raise ValueError('录像索引包含目录外路径')
                if row.get('video_index') != expected[name]:
                    indices_ok = False
                expected[name] += 1
        if not indices_ok:
            report['ok'] = False
            report['warnings'].append('视频帧索引存在重复或不连续。')
    else:
        report['ok'] = False
        report['warnings'].append('缺少 frames.jsonl，无法核对视频与采集帧索引。')
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in ('.avi', '.mp4', '.mkv') and p.is_file())
    if not files:
        raise ValueError('此文件夹没有 AVI、MP4 或 MKV 录像；请选择具体的 session 文件夹。')
    for path in files:
        print('检查：'+path.name, file=sys.stderr, flush=True)
        try:
            detail = verify_video(path, expected.get(path.name), full=full)
            detail['ok'] = True
            if index_path.exists() and path.name not in expected:
                report['ok'] = False
                report['warnings'].append('视频缺少对应采集帧索引：'+path.name)
            if detail['fourcc'].lower() in ('rgba', 'y800', 'ffv1'):
                detail['playback_note'] = '科学数据编码；普通播放器可能不支持，可导出 MP4 播放副本。'
        except Exception as exc:
            detail = dict(file=path.name, bytes=path.stat().st_size, ok=False, error=str(exc))
            report['ok'] = False
        report['files'].append(detail)
    for name in expected:
        if not (root/name).is_file():
            report['ok'] = False
            report['warnings'].append('索引引用的视频缺失：'+name)
    report['total_video_bytes'] = sum(f['bytes'] for f in report['files'])
    report['verification'] = 'full_decode' if full else 'first_middle_last_decode'
    return report


def export_mp4(folder):
    root = Path(folder).resolve()
    # A finalized summary is the existing software's signal that all writers stopped.
    if not (root/'summary.json').is_file():
        raise ValueError('尚未找到保存报告。请先停止录像并等待保存完成，再导出播放副本。')
    report = inspect_session(root)
    if any(not f['ok'] for f in report['files']):
        raise ValueError('存在无法读取或帧数不符的视频，请先查看检查结果。')
    output = root/('mp4_playback_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    exported = []
    try:
        for file in report['files']:
            source = root/file['file']
            destination = output/(source.name+'.mp4')
            print('转出 MP4：'+source.name, file=sys.stderr, flush=True)
            # Copy existing H.264 MP4 streams; transcode legacy raw/FFV1 videos.
            codec_args = ['-c:v', 'copy'] if source.suffix.lower()=='.mp4' and file['fourcc'].lower()=='h264' else [
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18' if '_rgb' in source.stem else '12',
                '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-threads', '2', '-bf', '0']
            command = [executable(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                       '-i', str(source), '-map', '0:v:0', '-an', '-fps_mode', 'passthrough',
                       *codec_args, '-movflags', '+faststart', str(destination)]
            process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     **hidden_process_options())
            if process.returncode:
                raise RuntimeError(process.stderr.decode('utf8', errors='replace')[-2000:])
            checked = verify_video(destination, file['frames'], full=True)
            checked['source_file'] = source.name
            exported.append(checked)
        result = dict(ok=True, output=str(output), source_recording_complete=report['recording_complete'],
                      warnings=report['warnings'], files=exported,
                      note='MP4 播放副本；转码包含有损压缩。原始视频、标定、音频和帧索引未修改。副本不替代测量源文件。')
    except Exception as exc:
        result = dict(ok=False, output=str(output), error=str(exc), files=exported)
    (output/'export_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


def main():
    sys.stdout.reconfigure(encoding='utf8')
    sys.stderr.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect', 'export'))
    parser.add_argument('folder')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    try:
        result = export_mp4(args.folder) if args.action=='export' else inspect_session(args.folder, args.full)
    except Exception as exc:
        result = dict(ok=False, error=str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
