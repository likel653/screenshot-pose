"""Explicit optional RTMW setup. Only this installer accesses the network."""
import argparse,hashlib,json,subprocess,sys,urllib.request,zipfile
from pathlib import Path
MODELS={
 'rtmw_detector.onnx':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
 'rtmw_pose.onnx':'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',
}
def install(root,python,skip_deps=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if not skip_deps:
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(Path(__file__).with_name('requirements_high.txt'))],check=True)
    report={}
    for name,url in MODELS.items():
        target=root/name
        if not target.is_file():
            archive=root/(name+'.download');part=root/(name+'.partial')
            try:
                print('Downloading '+name,flush=True)
                with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as out:
                    size=0
                    while chunk:=response.read(1024*1024):
                        size+=len(chunk)
                        if size>1500*1024*1024:raise ValueError('Unexpected download size')
                        out.write(chunk)
                with zipfile.ZipFile(archive) as z:
                    entries=[i for i in z.infolist() if i.filename.endswith('.onnx')]
                    if len(entries)!=1 or entries[0].file_size>1500*1024*1024:raise ValueError('Unexpected model archive')
                    with z.open(entries[0]) as src,part.open('wb') as dst:
                        while chunk:=src.read(1024*1024):dst.write(chunk)
                part.replace(target)
            finally:
                archive.unlink(missing_ok=True);part.unlink(missing_ok=True)
        report[name]={'url':url,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    check='import onnxruntime as o,sys; [o.InferenceSession(p,providers=["CPUExecutionProvider"]) for p in sys.argv[1:]]; print("RTMW_READY")'
    subprocess.run([str(python),'-c',check,*[str(root/n) for n in MODELS]],check=True)
    (root/'rtmw_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--skip-deps',action='store_true');a=p.parse_args()
    install(a.runtime,a.python,a.skip_deps)
