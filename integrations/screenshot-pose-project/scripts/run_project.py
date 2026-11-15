"""Launch inference and verified Blender delivery without touching a live scene."""
import argparse,hashlib,json,shutil,subprocess,sys,urllib.request,urllib.parse
from pathlib import Path
HERE=Path(__file__).resolve().parent

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(cmd,log):
 with log.open('ab') as f:
  p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 if p.returncode:raise RuntimeError('步骤失败，退出码 %d；日志：%s'%(p.returncode,log))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'local_config.json'));p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 p.add_argument('--kind',choices=['auto','image','video','cache'],default='auto')
 for n in ('template','rig'):p.add_argument('--'+n)
 for n,v in [('start',0.),('end',0.),('sample-fps',12.),('fps',24.)]:p.add_argument('--'+n,type=float,default=v)
 p.add_argument('--start-frame',type=int,default=1);p.add_argument('--render',action='store_true');p.add_argument('--keep-camera',action='store_true');args=p.parse_args()
 cfg=json.loads(Path(args.config).read_text(encoding='utf8'));root=Path(cfg['project_root']).resolve();template=Path(args.template or cfg['template']).resolve();blender=Path(cfg['blender']);python=Path(cfg['python'])
 for path in (template,blender,python,root/'screenshot_pose/worker_sam.py'):
  if not path.is_file():raise ValueError('缺少本地资源：'+str(path))
 if args.fps<=0 or args.sample_fps<0 or args.start<0 or args.end<0:raise ValueError('时间或帧率无效')
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
 def progress(stage,**kw):write(out/'progress.json',dict(stage=stage,**kw));print(stage,flush=True)
 try:
  before=digest(template);progress('准备媒体');source=args.input;kind=args.kind
  url=urllib.parse.urlparse(source);ext=Path(url.path).suffix.lower();images={'.png','.jpg','.jpeg','.webp','.bmp'}
  if url.scheme in ('http','https'):
   if ext in images:
    source=out/('source'+ext)
    with urllib.request.urlopen(args.input,timeout=60) as response,source.open('wb') as f:shutil.copyfileobj(response,f)
   else:
    run([python,'-m','yt_dlp','--no-playlist','--no-overwrites','-f','bestvideo+bestaudio/best','--merge-output-format','mp4','--print-to-file','after_move:filepath',out/'download_path.txt','-o',out/'source.%(ext)s','--',args.input],out/'run.log')
    paths=(out/'download_path.txt').read_text(encoding='utf8').splitlines();source=Path(paths[-1])
  else:
   source=Path(source).resolve()
   if not source.is_file():raise ValueError('输入文件不存在：'+str(source))
   if kind!='cache' and source.suffix.lower()!='.json':
    dest=out/('source'+source.suffix.lower());shutil.copy2(source,dest);source=dest
  source=Path(source).resolve()
  if kind=='auto':kind='image' if source.suffix.lower() in images else 'cache' if source.suffix.lower()=='.json' else 'video'
  runtime_cfg=json.loads((root/'screenshot_pose/local_runtime.json').read_text(encoding='utf8'));models=Path(runtime_cfg.get('runtime',root/'test_output/models'));model=models/'pose_landmarker_full.task'
  cache=out/'cache';progress('识别或载入缓存',kind=kind)
  if kind=='cache':
   shutil.copytree(source.parent,cache);sequence=cache/source.name
  elif kind=='image':
   cache.mkdir();pose=cache/'pose.json'
   run([python,root/'screenshot_pose/worker_sam.py','--image',source,'--model',model,'--output',pose,'--overlay',cache/'overlay.png','--native-root-path',runtime_cfg.get('sam_root',root)],out/'run.log')
   # Two identical samples make a reusable still-pose action through the same
   # transactional retarget path as video. There is no invented motion.
   sequence=cache/'sequence.json';write(sequence,dict(schema='screenshot_pose/video_v1',start_seconds=0,source_video=str(source),frames=[dict(time=t,status='ok',pose='pose.json') for t in (0,1/args.fps)]))
  else:
   run([python,root/'screenshot_pose/worker_video.py','--video',source,'--output',cache,'--model',model,'--backend','SAM3D','--start',args.start,'--end',args.end,'--sample-fps',args.sample_fps,'--native-root-path',runtime_cfg.get('sam_root',root)],out/'run.log');sequence=cache/'sequence.json'
  job=dict(project_root=str(root),template=str(template),source=str(source),source_request=args.input,kind=kind,cache=str(sequence),rig=args.rig or cfg.get('rig'),fps=args.fps,start_frame=args.start_frame,render=args.render,keep_camera=args.keep_camera,output=str(out))
  write(out/'job.json',job);progress('生成角色关键帧')
  run([blender,'--factory-startup','-b',template,'-t','4','--python-exit-code','1','--python',HERE/'build_project.py','--',out/'job.json'],out/'blender.log')
  progress('重新加载验证工程')
  run([blender,'--factory-startup','-b',out/'project.blend','-t','4','--python-exit-code','1','--python',HERE/'verify_project.py','--',out/'job.json'],out/'verify.log')
  assert before==digest(template),'源模板发生意外变化'
  result=json.loads((out/'result.json').read_text(encoding='utf8'));result.update(success=True,stage="complete",source_template_unchanged=True,reopen_verified=True);write(out/'result.json',result);progress('完成',project=str(out/'project.blend'))
 except Exception as exc:
  write(out/'failure.json',dict(error=str(exc)));progress('失败',error=str(exc));raise
if __name__=='__main__':main()
