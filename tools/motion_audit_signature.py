"""Input/plug-in identity for safely resuming a diagnostic export."""
import hashlib,json
from pathlib import Path

def input_signature(config,root):
    digest=hashlib.sha256()
    def read(path):
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    manifest=Path(config['cache']).resolve();cache=manifest.parent
    read(config['template']);read(manifest)
    for item in json.loads(manifest.read_text('utf8'))['frames']:
        if item['status']!='ok':continue
        name=Path(item['pose']);pose=(cache/name).resolve()
        if name.is_absolute() or not pose.is_relative_to(cache) or not pose.is_file():raise ValueError('Pose cache path escapes its directory or is missing')
        digest.update(b'screenshot_pose.pose_data');read(pose)
    for path in sorted((Path(root)/'screenshot_pose').glob('*.py')):
        digest.update(path.name.encode());read(path)
    return digest.hexdigest()
