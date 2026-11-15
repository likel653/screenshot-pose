"""Install a private inference environment; invoked only by the setup button/CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
             'pose_landmarker_full/float16/1/pose_landmarker_full.task')


def python_at(root):
    return Path(root) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def install(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = root / 'python_env'
    python = python_at(env)
    if not python.is_file():
        print('Creating private Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    print('Installing local pose recognition dependencies...', flush=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '-r', str(Path(__file__).with_name('requirements.txt'))], check=True)
    model = root / 'pose_landmarker_full.task'
    if not model.is_file() or model.stat().st_size < 1000000:
        print('Downloading official MediaPipe pose model...', flush=True)
        temp = model.with_suffix('.download')
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temp.open('wb') as output:
                total = 0
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 100*1024*1024:
                        raise ValueError('Unexpected model size')
                    output.write(chunk)
            if temp.stat().st_size < 1000000:
                raise ValueError('Model download incomplete')
            temp.replace(model)
        finally:
            temp.unlink(missing_ok=True)
    # Loading the model here verifies the runtime, model, and native dependencies.
    check = ('import mediapipe as mp; '
             'o=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=open(__import__("sys").argv[1],"rb").read())); '
             'p=mp.tasks.vision.PoseLandmarker.create_from_options(o); '
             'p.close(); print("POSE_RUNTIME_READY")')
    subprocess.run([str(python), '-c', check, str(model)], check=True)
    report = {'python':str(python),'model':str(model),'model_url':MODEL_URL,
              'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
    (root/'runtime.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Setup complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    try:
        install(args.runtime)
    except Exception as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        sys.exit(1)
