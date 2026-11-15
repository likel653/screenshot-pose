"""Run optional auxiliary SAM views locally, caching per-image predictions."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
try:
 from . import worker,worker_sam,pose_data,multi_view
except ImportError:
 import worker,worker_sam,pose_data,multi_view

def pixels(path):
 im=worker.prepare_image(path)
 return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
def fingerprint(model,native):
 root=Path(model).resolve().parent;paths=[root/n for n in ('pose_landmarker_full.task','rtmw_detector.onnx','rtmw_pose.onnx','rtmw3d_pose.onnx','face_landmarker.task','hand_landmarker.task')]
 paths += list((root/'sam3d_gguf').glob('*.gguf'))+list((native/'native_build/vulkan/bin').glob('*.exe'))+list((native/'native_build/vulkan/bin').glob('*.dll'))
 identity=[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(paths) if p.is_file()]
 code=[hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('worker_sam.py','worker_gpu.py','worker_high.py','local_pose_worker.py','sam_data.py')]
 return hashlib.sha256(json.dumps([identity,code]).encode()).hexdigest()
def run(input,references,model,output,overlay,strength=.35,confidence=.5,native_root_path=None):
 base=pose_data.read_pose(input);multi_view.skeleton(base)
 request_path=Path(references).resolve();out=Path(output).resolve();preview=Path(overlay).resolve();report_path=out.with_suffix('.report.json')
 if request_path.stat().st_size>65536:raise ValueError('辅助图请求过大')
 entries=json.loads(request_path.read_text(encoding='utf-8'));start=time.perf_counter()
 if not isinstance(entries,list) or not 1<=len(entries)<=3:raise ValueError('请选择 1 至 3 张辅助图')
 images=[Path(e['path']).resolve() for e in entries]
 protected={Path(input).resolve(),request_path,Path(model).resolve(),Path(base['image']['path']).resolve(),*images}
 if len({out,preview,report_path})!=3 or {out,preview,report_path}&protected:raise ValueError('多图输出不能覆盖输入或模型')
 main=base['image']['path'];main_hash=pixels(main);seen={main_hash};accepted=[];rejected=[];cache_hits=0
 native=Path(native_root_path) if native_root_path else worker_sam.native_root();cache_id=fingerprint(model,native)
 cache=Path(model).resolve().parent/'multiview_cache';cache.mkdir(parents=True,exist_ok=True)
 for number,(entry,path) in enumerate(zip(entries,images),1):
  try:
   person=int(entry.get('person_index',1))-1
   worker.validate_options(person,4,confidence)
   digest=pixels(path)
   if digest in seen:raise ValueError('图片与主图或其他辅助图重复')
   seen.add(digest);key=hashlib.sha256(json.dumps([digest,person,confidence,cache_id]).encode()).hexdigest();folder=cache/key
   folder.mkdir(exist_ok=True);pose=folder/'pose.json'
   print(f'辅助图 {number}/{len(images)}：{path.name}',flush=True)
   reference=None
   if pose.is_file():
    try:reference=pose_data.read_pose(pose);multi_view.skeleton(reference);cache_hits+=1
    except (ValueError,KeyError,OSError):reference=None
   if reference is None:reference=worker_sam.detect(path,model,pose,folder/'overlay.png',person_index=person,confidence=confidence,native_root_path=native)
   accepted.append(dict(image=str(path),data=reference))
  except Exception as exc:rejected.append(dict(image=str(path),reason=str(exc)[:300]))
 data=multi_view.fuse(base,accepted,strength);data['multiview']['rejected'].extend(rejected)
 data['multiview'].update(cache_hits=cache_hits,total_seconds=time.perf_counter()-start,primary_json=str(Path(input).resolve()))
 worker.write_json(out,data);worker.write_json(report_path,data['multiview'])
 im=worker.prepare_image(main);worker.save_png(worker.draw_overlay(im,data['landmarks'],.35,data.get('person_index',0)),preview)
 print(json.dumps(dict(ok=True,accepted=data['multiview']['accepted'],rejected=len(data['multiview']['rejected']),report=str(report_path))),flush=True)
 return data
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('input','references','model','output','overlay'):p.add_argument('--'+n,required=True)
 p.add_argument('--strength',type=float,default=.35);p.add_argument('--confidence',type=float,default=.5);p.add_argument('--native-root-path')
 try:run(**vars(p.parse_args()))
 except Exception as exc:print('多图辅助未完成：'+str(exc),file=sys.stderr);raise SystemExit(1)
