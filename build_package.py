"""Build and inspect a Blender-installable ZIP, without bundling user data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'screenshot_pose'
OUTPUT=ROOT/'ScreenshotPose_v0.39.0.zip'
allowed={'.py','.txt','.json'}
files=[p for p in PACKAGE.iterdir() if p.is_file() and p.suffix in allowed
       and p.name not in {'local_runtime.json', 'local_runtime.example.json'}]
with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in sorted(files):
        if path.suffix=='.py':
            compile(path.read_text(encoding='utf-8'),str(path),'exec')
        archive.write(path,'screenshot_pose/'+path.name)
    archive.write(ROOT/'README_中文.md','screenshot_pose/README_中文.md')
    archive.write(ROOT/'LICENSE','screenshot_pose/LICENSE')
with zipfile.ZipFile(OUTPUT) as archive:
    assert archive.testzip() is None
    names=archive.namelist()
    assert 'screenshot_pose/__init__.py' in names
    assert 'screenshot_pose/requirements.txt' in names
    assert len(names)==len(set(names))
    assert not any('..' in name or name.startswith(('/', '\\')) for name in names)
    destination=ROOT/'test_output'/'package_install'
    for name in names:
        path=destination/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(archive.read(name))
report={'zip':OUTPUT.name,'bytes':OUTPUT.stat().st_size,
        'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'files':names}
(ROOT/'PACKAGE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
