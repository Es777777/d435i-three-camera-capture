"""Machine-readable preflight; absent hardware does not block offline tools."""
import argparse
from datetime import datetime
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = {
    'numpy': 'numpy', 'opencv-python': 'cv2', 'pyrealsense2': 'pyrealsense2',
    'PyYAML': 'yaml', 'PyQt5': 'PyQt5.QtWidgets', 'PyQt5-Qt5': 'PyQt5.QtCore',
    'PyQt5_sip': 'PyQt5.sip', 'sounddevice': 'sounddevice', 'scipy': 'scipy.signal',
    'cffi': 'cffi', 'pycparser': 'pycparser', 'imageio-ffmpeg': 'imageio_ffmpeg',
}


def check_environment(quick=False, devices=False):
    report = dict(ok=True, checked_at=datetime.now().astimezone().isoformat(),
                  python=sys.version.split()[0], executable=sys.executable,
                  platform=platform.platform(), root=str(ROOT), checks=[], devices={})

    def add(name, ok, detail, required=True):
        report['checks'].append(dict(name=name, ok=bool(ok), detail=str(detail), required=required))
        if required and not ok:
            report['ok'] = False

    add('Python', sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8,
        f'{sys.version.split()[0]} / {struct.calcsize("P")*8} bit')
    for line in (ROOT/'requirements.txt').read_text(encoding='utf8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
            module = importlib.import_module(MODULES[name])
            consistent = actual == expected
            detail = f'{actual}（要求 {expected}）'
            if name == 'opencv-python':
                binary_version = '.'.join(expected.split('.')[:3])
                consistent = consistent and module.__version__ == binary_version
                detail += f' / 运行库 {module.__version__}'
            add(name, consistent, detail)
        except Exception as exc:
            add(name, False, exc)
    if not report['ok']:
        return report

    # Offscreen is local to this diagnostic process, never to the capture process.
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt5 import QtWidgets
    from qt_runtime import configure_qt
    configure_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    add('Qt', app is not None, '界面运行库已加载')
    if quick:
        return report

    try:
        for folder in ('sessions', 'telemetry', 'logs'):
            (ROOT/folder).mkdir(exist_ok=True)
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='环境检查_', dir=ROOT/'logs') as folder:
            video = str(Path(folder)/'codec.avi')
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'FFV1'), 30, (64, 48))
            try:
                if not writer.isOpened():
                    raise RuntimeError('无法创建 FFV1 无损视频')
                frame = np.full((48, 64, 3), 97, np.uint8)
                writer.write(frame)
            finally:
                writer.release()
            reader = cv2.VideoCapture(video)
            try:
                ok, decoded = reader.read()
                add('无损录像 / 目录写入', ok and np.array_equal(decoded, frame), 'FFV1 写入与回读')
            finally:
                reader.release()
            # Fast recording uses exact gray/RGBA raw AVI to avoid FFV1 CPU stalls.
            for codec, color in (('Y800', False), ('RGBA', True)):
                path = str(Path(folder)/(codec+'.avi'))
                pattern = np.random.default_rng(3).integers(0, 256, (48, 64, 3) if color else (48, 64), dtype=np.uint8)
                fast = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), 30, (64, 48), color)
                try:
                    if not fast.isOpened():
                        raise RuntimeError('无法创建快速无损视频：'+codec)
                    fast.write(pattern)
                finally:
                    fast.release()
                cap = cv2.VideoCapture(path)
                try:
                    valid, decoded = cap.read()
                    if valid and not color:
                        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                    add('快速无损 / '+codec, valid and np.array_equal(decoded, pattern), '原始像素写入与回读一致')
                finally:
                    cap.release()
            from video_encoding import FFmpegWriter, verify_video
            for codec, shape, extension in [('H264_RGB', (48, 64, 3), '.mp4'), ('FFV1', (48, 64), '.avi')]:
                path = Path(folder)/(codec+extension)
                pattern = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
                encoder = FFmpegWriter(path, codec, 30, shape)
                try:
                    encoder.write(pattern)
                finally:
                    encoder.release()
                verified = verify_video(path, 1, full=True)
                add('压缩录像 / '+codec, True, verified)
    except Exception as exc:
        add('无损录像 / 目录写入', False, exc)

    try:
        import yaml
        cfg = yaml.safe_load((ROOT/'capture_config.yaml').read_text(encoding='utf8'))
        if not isinstance(cfg, dict):
            raise ValueError('capture_config.yaml 必须为配置字典')
        target = json.loads((ROOT/'points_blender52_final.json').read_text(encoding='utf8'))
        add('采集配置 / 标定点表', bool(target['points']), '文件可读取')
    except Exception as exc:
        add('采集配置 / 标定点表', False, exc)

    if devices:
        try:
            import pyrealsense2 as rs
            context = rs.context()
            cameras = [dict(name=d.get_info(rs.camera_info.name),
                            serial=d.get_info(rs.camera_info.serial_number))
                       for d in context.query_devices()]
            report['devices']['cameras'] = cameras
            add('相机', bool(cameras), '；'.join(d['name']+' / '+d['serial'] for d in cameras)
                or '未连接；连接 D435i 后重新检查，仍可使用离线处理', required=False)
        except Exception as exc:
            add('相机', False, exc, required=False)
        try:
            import sounddevice as sd
            microphones = [dict(index=i, name=d['name']) for i, d in enumerate(sd.query_devices())
                           if d['max_input_channels'] > 0]
            report['devices']['microphones'] = microphones
            add('麦克风', bool(microphones), '；'.join(d['name'] for d in microphones)
                or '未发现输入设备；录像前连接麦克风并检查系统权限', required=False)
        except Exception as exc:
            add('麦克风', False, exc, required=False)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--devices', action='store_true')
    args = parser.parse_args()
    try:
        report = check_environment(args.quick, args.devices)
    except Exception as exc:
        report = dict(ok=False, checks=[dict(name='检查失败', ok=False, required=True, detail=str(exc))])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
