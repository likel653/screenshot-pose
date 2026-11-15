"""SAM native GPU body + independent RTMW/local hand observations."""
import argparse,json,os,re,struct,subprocess,time
from pathlib import Path
import numpy as np
try:
 from . import worker,worker_gpu,sam_data
except ImportError:
 import worker,worker_gpu,sam_data

def native_root():
 try:return Path(json.loads(Path(__file__).with_name('local_runtime.json').read_text())['sam_root'])
 except (OSError,KeyError,ValueError):return Path(__file__).resolve().parents[1]

def read_result(path):
 with Path(path).open('rb') as f:
  def unpack(fmt):
   b=f.read(struct.calcsize(fmt))
   if len(b)!=struct.calcsize(fmt):raise ValueError('SAM 结果被截断')
   return struct.unpack(fmt,b)
  if f.read(8)!=b'S3DOUT01':raise ValueError('SAM 结果格式无效')
  count,=unpack('<I');result={}
  if count>64:raise ValueError('SAM 结果字段过多')
  for _ in range(count):
   n,=unpack('<I')
   if n>128:raise ValueError('SAM 字段名无效')
   name=f.read(n).decode();kind,rank,size=unpack('<IIQ')
   if kind not in (1,2) or rank>4 or size>2000000 or name in result:raise ValueError('SAM 张量类型或尺寸无效')
   shape=unpack('<'+'Q'*rank)
   if int(np.prod(shape))!=size:raise ValueError('SAM 张量尺寸不匹配')
   value=np.frombuffer(f.read(size*4),dtype='<f4' if kind==1 else '<i4').reshape(shape).copy()
   if not np.isfinite(value).all():raise ValueError('SAM 结果含非有限值')
   result[name]=value
  if f.read(1):raise ValueError('SAM 结果含多余内容')
  return result

class PersonNotFound(ValueError):
 pass


def detect(image,model,output,overlay,person_index=0,max_poses=4,confidence=.5,native_root_path=None,engines=None,continuity=None,camera_calibration=None,screenshot_checks=False,reference_device='cuda'):
 try:from .person_detector import make_detector
 except ImportError:from person_detector import make_detector
 root=Path(native_root_path) if native_root_path else native_root()
 binary=root/'native_build/vulkan/bin';exe=binary/'sam3d-body-infer.exe';module=binary/'ggml-vulkan.dll';models=Path(model).resolve().parent/'sam3d_gguf'
 assets=[models/n for n in ('body-dinov3-f32.gguf','body-pose-branch-f32.gguf','mhr-lod1-f32.gguf')]
 if not all(p.is_file() for p in [exe,module,*assets]):raise ValueError('SAM 原生 GPU 运行器或权重不存在，请在偏好设置检查 SAM 目录')
 worker.validate_options(person_index,max_poses,confidence)
 # Reuse the tested GPU/local workflow for independent 133-point observations.
 # SAM errors propagate; never label a previous model's output as SAM.
 output=Path(output).resolve();overlay=Path(overlay).resolve();original=Path(image).resolve()
 protected=[exe,module,*assets,original,Path(model).resolve()]
 paths=[output,overlay,output.with_suffix('.sam.input'),output.with_suffix('.sam.bin'),output.with_suffix('.sam.npz'),output.with_suffix('.sam.log'),output.with_suffix('.sam.reference.json'),output.with_suffix('.sam.reference.png')]
 if len(set(paths))!=len(paths) or any(p in protected for p in paths):raise ValueError('输出路径不能覆盖图片、模型或运行器')
 im=worker.prepare_image(original);w,h=im.size;rgb=np.asarray(im);bgr=np.ascontiguousarray(rgb[:,:,::-1])
 create=lambda:make_detector(str(Path(model).parent/'rtmw_detector.onnx'),score_thr=confidence,backend='onnxruntime',device='cpu')
 detector=engines.get(('yolox_cpu',str(Path(model).resolve().parent),confidence),create) if engines else create()
 boxes=sorted(detector(bgr),key=lambda b:float((b[2]-b[0])*(b[3]-b[1])),reverse=True)[:max_poses]
 if person_index>=len(boxes):raise PersonNotFound('SAM 未找到所选人物')
 box=continuity.select(boxes,im.size) if continuity else boxes[person_index]
 shared_box=continuity is not None or screenshot_checks
 reference=worker_gpu.detect(original,model,output.with_suffix('.sam.reference.json'),output.with_suffix('.sam.reference.png'),0 if shared_box else person_index,max_poses,confidence,device=reference_device,engines=engines,person_box=box if shared_box else None,continuity=continuity)
 if screenshot_checks:reference.update(person_index=person_index,pose_count=len(boxes))
 focal=float(np.hypot(w,h));camera=[focal,focal,w/2,h/2]
 camera_method='image_diagonal_default_not_calibration'
 if camera_calibration is not None:
  try:from . import camera_data
  except ImportError:import camera_data
  camera=camera_data.intrinsics(camera_calibration,w,h).tolist()
  if camera_data.pixel_digest(rgb)!=camera_calibration.get('image_sha256'):raise ValueError('SAM 相机标定与当前图片内容不一致')
  camera_method=camera_calibration.get('source','estimated_intrinsics')
 wire=output.with_suffix('.sam.input');result=output.with_suffix('.sam.bin');log=output.with_suffix('.sam.log')
 wire.write_bytes(b'S3DIMG01'+struct.pack('<III',w,h,w*3)+struct.pack('<4f',*box)+struct.pack('<4f',*camera)+rgb.tobytes())
 env=dict(os.environ,GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
 env['PATH']=str(binary)+os.pathsep+env.get('PATH','')
 args=[exe,module,'Vulkan','0','-',*assets,wire,result,'8'];start=time.perf_counter()
 inference_ms=None
 if engines:
  try:from .inference_session import NativeWorker
  except ImportError:from inference_session import NativeWorker
  native_log=output.parent.parent/'sam_native_session.log'
  persistent=engines.get(('sam_native',str(root.resolve()),str(models)),lambda:NativeWorker([exe,'--worker',module,'Vulkan','0','-',*assets,'8'],env,native_log))
  inference_ms=persistent.infer(wire,result);text=native_log.read_text(encoding='utf8',errors='replace')
  log.write_text('Persistent Vulkan worker\nTIMING body_infer_ms '+str(inference_ms)+'\nSession log: '+str(native_log),encoding='utf8')
 else:
  with log.open('w',encoding='utf-8') as stream:
   done=subprocess.run([str(a) for a in args],env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  if done.returncode:raise RuntimeError('SAM GPU 推理失败，请查看 '+str(log))
  text=log.read_text(encoding='utf-8',errors='replace')
 wall=time.perf_counter()-start
 if 'Vulkan' not in text:raise RuntimeError('SAM 日志未确认 GPU，已停止')
 raw=read_result(result);np.savez_compressed(output.with_suffix('.sam.npz'),**raw)
 data=sam_data.convert(raw,reference,camera=dict(intrinsics=camera,method=camera_method))
 if screenshot_checks:
  sam_data.check_screenshot_observations(data,raw)
  area=lambda b:float((b[2]-b[0])*(b[3]-b[1]))
  if len(boxes)>1:data['screenshot_person_area_ratio']=area(boxes[0])/max(area(boxes[1]),1.)
 processed=output.with_name(output.stem+'.input.png');worker.save_png(im,processed)
 data['image']=dict(path=str(processed),original_path=str(original),width=w,height=h)
 match=re.search(r'TIMING body_infer_ms ([0-9.]+)',text)
 data['runtime']=dict(device='vulkan',model='SAM 3D Body DINOv3 F32',inference_ms=inference_ms if inference_ms is not None else float(match[1]) if match else None,process_wall_seconds=wall,reference=reference['runtime'],native_result=str(result),native_log=str(log),persistent_session=bool(engines))
 if continuity:continuity.commit(data)
 worker.save_png(worker.draw_overlay(im,data['landmarks'],.35,person_index),overlay);worker.write_json(output,data)
 return data

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('image','model','output','overlay'):p.add_argument('--'+n,required=True)
 p.add_argument('--person-index',type=int,default=0);p.add_argument('--max-poses',type=int,default=4);p.add_argument('--confidence',type=float,default=.5);p.add_argument('--native-root-path')
 try:print(json.dumps(detect(**vars(p.parse_args()))['runtime']))
 except Exception as exc:print('SAM 识别失败：'+str(exc),file=__import__('sys').stderr);raise SystemExit(1)
