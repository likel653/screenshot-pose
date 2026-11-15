"""Calibrate, optionally rerun SAM with the estimated intrinsics, preserve hands."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
try:from . import pose_data,worker,worker_sam
except ImportError:import pose_data,worker,worker_sam


def run(input,model,output,native_root_path=None,refresh_body=True):
    source=Path(input).resolve();out=Path(output).resolve();runtime=Path(model).resolve().parent
    base=pose_data.read_pose(source);cfgpath=runtime/'camera/config.json';cfg=json.loads(cfgpath.read_text(encoding='utf8'))
    protected={source,cfgpath,Path(model).resolve(),Path(base['image']['path']).resolve()}
    if out in protected or out.exists() or any(out.with_suffix(suffix).exists() for suffix in ('.camera.json','.log')):
        raise ValueError('相机任务不能覆盖输入或已有结果')
    if base.get('edited_indices'):raise ValueError('当前图片有人工关节校正，请先保存到姿势库，再从原图执行标定')
    out.parent.mkdir(parents=True,exist_ok=True);calfile=out.with_suffix('.camera.json');log=out.with_suffix('.log')
    with log.open('w',encoding='utf8') as stream:
        subprocess.run([cfg['python'],str(Path(__file__).with_name('worker_camera.py')),'--image',base['image']['path'],
          '--config',str(cfgpath),'--output',str(calfile)],stdout=stream,stderr=subprocess.STDOUT,check=True,
          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=300)
    calibration=json.loads(calfile.read_text(encoding='utf8'));updated=False
    if refresh_body and base.get('full_skeleton',{}).get('source')=='sam3d_mhr':
        result=worker_sam.detect(base['image']['path'],model,out,out.with_suffix('.png'),person_index=base.get('person_index',0),
            native_root_path=native_root_path,camera_calibration=calibration)
        result['image']['original_path']=base['image'].get('original_path',base['image']['path'])
        result.setdefault('orientation',{})['hands']=copy.deepcopy(base.get('orientation',{}).get('hands',{}))
        candidates=copy.deepcopy(base.get('full_skeleton_candidates',[]))
        candidates.append(dict(source='sam3d_before_camera_calibration',name='标定前身体（对照）',
            skeleton=copy.deepcopy(base['full_skeleton']),world_landmarks=copy.deepcopy(base['world_landmarks'])))
        result['full_skeleton_candidates']=candidates
        updated=True
    else:result=copy.deepcopy(base)
    calibration['processed_file_sha256']=hashlib.sha256(Path(result['image']['path']).read_bytes()).hexdigest()
    result['camera_calibration']=calibration
    result['camera_refinement']=dict(primary_json=str(source),primary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
      body_reinferred_with_calibration=updated,hands_preserved=True,estimated_not_measured=True,auto_applied=False)
    worker.write_json(out,result);print(json.dumps(result['camera_refinement']),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('input','model','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--native-root-path');p.add_argument('--no-refresh-body',dest='refresh_body',action='store_false')
    run(**vars(p.parse_args()))
