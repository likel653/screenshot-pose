"""Prepare official source/checkpoints in an isolated environment; MANO is user supplied."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

REPOS = {'WILOR': 'https://github.com/rolpotamias/WiLoR.git',
         'HAMER': 'https://github.com/geopavlakos/hamer.git'}
SOURCES = {'WILOR': 'https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt',
           'HAMER': 'https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz'}


def resolve_mano(value, target):
    source = Path(value).resolve()
    if not source.is_file():
        raise ValueError('MANO 文件不存在')
    if source.suffix.lower() != '.zip':
        if source.name.upper() != 'MANO_RIGHT.PKL' or not 100_000 < source.stat().st_size < 50_000_000:
            raise ValueError('请选择官方 MANO_RIGHT.pkl 或 MANO 模型 ZIP')
        return str(source)
    with zipfile.ZipFile(source) as archive:
        candidates = [v for v in archive.infolist() if not v.is_dir() and Path(v.filename).name == 'MANO_RIGHT.pkl' and not v.filename.startswith('__MACOSX/')]
        if len(candidates) != 1 or not 100_000 < candidates[0].file_size < 50_000_000:
            raise ValueError('MANO 压缩包没有唯一有效的 MANO_RIGHT.pkl')
        data = archive.read(candidates[0])
    destination = target / 'MANO_RIGHT.pkl'
    temporary = target / 'MANO_RIGHT.pkl.tmp'
    temporary.write_bytes(data);temporary.replace(destination)
    return str(destination)


def download(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    size = part.stat().st_size if part.is_file() else 0
    headers = {'User-Agent': 'ScreenshotPose/0.30'}
    if size:
        headers['Range'] = 'bytes=%d-' % size
    try:
        import truststore
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60, context=context) as response:
        append = size > 0 and response.status == 206
        size = size if append else 0
        expected = int(response.headers.get('Content-Length', 0)) + size
        with part.open('ab' if append else 'wb') as stream:
            last = time.monotonic()
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 8 * 1024 ** 3:
                    raise ValueError('模型下载超过 8 GiB')
                stream.write(chunk)
                if time.monotonic() - last > 15:
                    print(path.name, round(size / 1024 ** 2), 'MiB', flush=True)
                    last = time.monotonic()
        if expected and size != expected:
            raise ValueError('模型下载不完整')
    if size < 1024:
        raise ValueError('下载内容过小')
    part.replace(path)


def weights(backend, root, repo):
    folder = root / backend.lower()
    folder.mkdir(parents=True, exist_ok=True)
    if backend == 'WILOR':
        checkpoint = folder / 'wilor_final.ckpt'
        download(SOURCES[backend], checkpoint)
        config = repo / 'pretrained_models/model_config.yaml'
        mean = repo / 'mano_data/mano_mean_params.npz'
    else:
        checkpoint = folder / 'hamer.ckpt'
        config = folder / 'model_config.yaml'
        mean = folder / 'mano_mean_params.npz'
        if not all(p.is_file() for p in (checkpoint, config, mean)):
            archive = folder / 'hamer_demo_data.tar.gz'
            download(SOURCES[backend], archive)
            wanted = {'hamer.ckpt': checkpoint, 'model_config.yaml': config, 'mano_mean_params.npz': mean}
            with tarfile.open(archive, 'r:gz') as tar:
                for item in tar:
                    name = Path(item.name).name
                    if name not in wanted or not item.isfile():
                        continue
                    if name == 'model_config.yaml' and 'hamer_ckpts' not in item.name:
                        continue
                    target = wanted[name]
                    temporary = target.with_suffix(target.suffix + '.tmp')
                    with tar.extractfile(item) as src, temporary.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    temporary.replace(target)
            if not all(p.is_file() for p in (checkpoint, config, mean)):
                raise ValueError('HaMeR 官方压缩包缺少推理文件')
    with checkpoint.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(repo=str(repo), checkpoint=str(checkpoint), config=str(config), mean_params=str(mean),
                source=SOURCES[backend], sha256=checksum)


def prepare(root, runtime, mano='', skip_env=False):
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    target = runtime / 'hand3d'
    target.mkdir(parents=True, exist_ok=True)
    env = root / '.venv_hand3d'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not skip_env:
        if not python.is_file():
            subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'wheel'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-build-isolation', '-r',
                        str(Path(__file__).with_name('requirements_hand3d.txt'))], check=True)
        if Path(sys.executable).resolve() != python.resolve():
            command = [str(python), str(Path(__file__).resolve()), '--root', str(root),
                       '--runtime', str(runtime), '--skip-env']
            if mano:
                command += ['--mano', mano]
            subprocess.run(command, check=True)
            return json.loads((target / 'setup_report.json').read_text(encoding='utf8'))
    repos = {}
    for backend, url in REPOS.items():
        repo = root / 'vendor' / ('WiLoR' if backend == 'WILOR' else 'hamer')
        if not (repo / '.git').is_dir():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--depth', '1', url, str(repo)], check=True)
        repos[backend] = repo
    old = target / 'config.json'
    previous = json.loads(old.read_text(encoding='utf8')) if old.is_file() else {}
    cfg = dict(schema='screenshot_pose/hand3d_config_v1', python=str(python),
               mano=resolve_mano(mano, target) if mano else previous.get('mano', str(target / 'MANO_RIGHT.pkl')),
               models={})
    detector=target/'detector.pt'
    download('https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt',detector)
    cfg['hand_detector']=str(detector)
    # Publish useful paths even if a network download fails; readiness remains
    # false and the UI reports the actual missing asset rather than a fallback.
    failures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {b: pool.submit(weights, b, target, r) for b, r in repos.items()}
        for backend, future in jobs.items():
            try:
                cfg['models'][backend] = future.result()
                cfg['models'][backend]['source_commit'] = subprocess.check_output(
                    ['git','-C',str(repos[backend]),'rev-parse','HEAD'],text=True).strip()
            except Exception as exc:
                failures[backend] = str(exc)
                cfg['models'][backend] = previous.get('models', {}).get(backend, {'repo': str(repos[backend])})
    for optional in ('HAMER_TEXTURE',):
        if optional in previous.get('models',{}):cfg['models'][optional]=previous['models'][optional]
    old.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf8')
    try:
        check = subprocess.run([str(python), str(Path(__file__).with_name('hand3d_runner.py')),
                    '--config', str(old), '--check-imports'], capture_output=True, text=True, timeout=120)
        if check.returncode:
            raise RuntimeError((check.stderr or check.stdout)[-1500:])
        cfg['environment'] = json.loads(check.stdout.strip().splitlines()[-1])
    except Exception as exc:
        cfg['environment'] = dict(imports_ok=False, cuda=False, error=str(exc))
    old.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf8')
    try:
        from .hand3d import readiness
    except ImportError:
        from hand3d import readiness
    report = dict(models={b: readiness(runtime, b) for b in REPOS}, downloads_failed=failures,
                  mano_website='https://mano.is.tue.mpg.de/',
                  assets_not_bundled_in_addon=True)
    (target / 'setup_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=True), flush=True)
    if failures:
        raise RuntimeError('部分权重下载失败，详情见 hand3d/setup_report.json')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--mano', default='')
    p.add_argument('--skip-env', action='store_true')
    prepare(**vars(p.parse_args()))
