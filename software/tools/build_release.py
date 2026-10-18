"""Build clean online/offline Windows distributions from an explicit allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT.parent/'releases'


def source_files():
    for path in sorted(ROOT.glob('*.py')):
        if not path.name.startswith('check_'):
            yield path
    for pattern in ('*.bat', '*.yaml', 'points_blender52_final.json', 'requirements.txt', 'README.md'):
        yield from sorted(ROOT.glob(pattern))
    for pattern in ('*.py', '*.ps1', '*.json'):
        yield from sorted((ROOT/'tools').glob(pattern))
    yield from sorted((ROOT/'tests').glob('*.py'))
    yield from sorted((ROOT/'assets').glob('*.png'))


def prepare_offline():
    offline = ROOT/'offline'
    wheels = offline/'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'tools'/'runtime.json').read_text(encoding='utf8'))
    runtime = offline/manifest['filename']
    if not runtime.exists():
        cache = ROOT/'.runtime'/'downloads'/manifest['filename']
        if cache.exists():
            runtime.write_bytes(cache.read_bytes())
        else:
            urllib.request.urlretrieve(manifest['url'], runtime)
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Offline Python archive SHA256 mismatch')
    pins = {}
    for line in (ROOT/'requirements.txt').read_text(encoding='utf8').splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==')
            pins[name.lower().replace('-', '_').replace('.', '_')] = version
    def select_wheels():
        return [p for p in sorted(wheels.glob('*.whl'))
                if pins.get(p.name.split('-')[0].lower()) == p.name.split('-')[1]]
    selected = select_wheels()
    if len(selected) != len(pins):
        subprocess.run([sys.executable, '-I', '-X', 'utf8', '-m', 'pip', '--isolated', 'download',
                        '--disable-pip-version-check', '--only-binary=:all:',
                        '--platform', 'win_amd64', '--python-version', '3.12', '--implementation', 'cp',
                        '--abi', 'cp312', '--dest', str(wheels), '--index-url', 'https://pypi.org/simple',
                        '-r', str(ROOT/'requirements.txt')], check=True)
        selected = select_wheels()
    if len(selected) != len(pins):
        raise ValueError('The offline wheel set does not match the locked requirements')
    return [runtime, *selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    files = list(source_files())
    if args.offline:
        files.extend(prepare_offline())
    folder = 'D435I_ThreeHost_v1.0.0'
    RELEASES.mkdir(exist_ok=True)
    output = RELEASES/(folder+('_offline' if args.offline else '')+'.zip')
    hashes = {}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, f'{folder}/{relative}')
        archive.writestr(f'{folder}/SHA256SUMS.json', json.dumps(hashes, ensure_ascii=False, indent=2))
    output.with_suffix('.zip.sha256').write_text(hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n', encoding='ascii')
    print(f'{output} ({output.stat().st_size/1024/1024:.1f} MB, {len(files)} files)')


if __name__ == '__main__':
    main()
