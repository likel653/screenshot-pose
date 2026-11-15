"""Portable local pose entries: relative assets, atomic writes, recoverable trash."""
import copy,datetime,hashlib,json,os,re,shutil,uuid
from pathlib import Path
from . import pose_data
ID=re.compile(r'^[0-9a-f]{32}$')

def entry(root,key):
 if not ID.fullmatch(key):raise ValueError('姿势条目编号无效')
 root=Path(root).resolve();raw=root/key;path=raw.resolve()
 if raw.is_symlink() or path.parent!=root or path.name!=key:raise ValueError('姿势条目路径无效')
 return path

def asset(folder,name):
 path=(Path(folder)/name).resolve()
 if path.parent!=Path(folder).resolve() or not path.is_file():raise ValueError('姿势库素材不存在或路径无效')
 return path

def write(path,data):
 path=Path(path);temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
 try:
  temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(temp,path)
 finally:
  if temp.exists():temp.unlink()

def metadata(folder):
 path=Path(folder)/'entry.json'
 if path.stat().st_size>65536:raise ValueError('姿势元数据过大')
 meta=json.loads(path.read_text(encoding='utf8'))
 if meta.get('schema')!='screenshot_pose/library_v1' or meta.get('id')!=Path(folder).name:raise ValueError('不是有效的姿势库条目')
 asset(folder,meta['image']);asset(folder,'pose.json')
 return meta

def list_entries(root,search=''):
 root=Path(root);result=[];bad=0
 if not root.is_dir():return result,bad
 for path in root.iterdir():
  if not ID.fullmatch(path.name) or not path.is_dir():continue
  try:
   folder=entry(root,path.name);meta=metadata(folder)
   if search.casefold() in (meta['name']+' '+meta.get('tags','')+' '+meta.get('notes','')).casefold():result.append(meta)
  except (OSError,ValueError,KeyError,TypeError):bad+=1
 return sorted(result,key=lambda m:m['created'],reverse=True),bad

def save(root,data,image,name,tags='',notes='',options=None,guides=None,overlay=None):
 pose_data.validate(data);image=Path(image)
 if not image.is_file():raise ValueError('原截图不存在，不能保存完整姿势条目')
 name=name.strip()
 if not name or len(name)>100:raise ValueError('请输入 1–100 字的姿势名称')
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
 key=uuid.uuid4().hex;folder=entry(root,key);stage=root/('.saving_'+key);stage.mkdir()
 try:
  suffix=image.suffix.lower()
  if suffix not in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'):raise ValueError('截图文件类型不支持')
  image_name='image'+suffix;shutil.copy2(image,stage/image_name)
  doc=copy.deepcopy(data);doc['image']['path']=image_name;doc['image']['original_path']=image_name
  original=data['image'].get('original_path');original_name=None
  if original and Path(original).is_file() and Path(original).resolve()!=image.resolve() and Path(original).suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'):
   original_name='original'+Path(original).suffix.lower();shutil.copy2(original,stage/original_name);doc['image']['original_path']=original_name
  # Auxiliary hand reference images are optional but become self-contained.
  for side,hand in doc.get('orientation',{}).get('hands',{}).items():
   reference=hand.get('reference_image')
   if side not in ('L','R'):hand.pop('reference_image',None);continue
   if reference and Path(reference).is_file():
    dest='hand_'+side+Path(reference).suffix.lower();shutil.copy2(reference,stage/dest);hand['reference_image']=dest
   elif reference:hand.pop('reference_image',None)
  doc.pop('runtime',None)
  write(stage/'pose.json',doc)
  if overlay and Path(overlay).is_file():shutil.copy2(overlay,stage/'overlay.png')
  meta=dict(schema='screenshot_pose/library_v1',id=key,name=name,tags=tags[:300],notes=notes[:2000],created=datetime.datetime.now(datetime.timezone.utc).isoformat(),image=image_name,original=original_name,source=data.get('source',''),options=options or {},guides=guides or {},image_sha256=hashlib.sha256((stage/image_name).read_bytes()).hexdigest())
  write(stage/'entry.json',meta);stage.rename(folder)
  return meta
 except Exception:
  if stage.is_dir() and stage.parent==root:shutil.rmtree(stage)
  raise

def materialize(root,key,destination):
 folder=entry(root,key);meta=metadata(folder);doc=pose_data.read_pose(folder/'pose.json')
 destination=Path(destination)
 if destination.exists():raise ValueError('工作副本目录已存在')
 # Copy only declared assets; never execute or follow arbitrary entry files.
 names={meta['image'],'pose.json','entry.json'}
 if meta.get('original'):names.add(meta['original'])
 if (folder/'overlay.png').is_file():names.add('overlay.png')
 for hand in doc.get('orientation',{}).get('hands',{}).values():
  if hand.get('reference_image'):names.add(hand['reference_image'])
 sources={name:asset(folder,name) for name in names}
 destination.mkdir(parents=True)
 for name,path in sources.items():shutil.copy2(path,destination/name)
 doc['image']['path']=str((destination/meta['image']).resolve());doc['image']['original_path']=str((destination/(meta.get('original') or meta['image'])).resolve())
 for hand in doc.get('orientation',{}).get('hands',{}).values():
  if hand.get('reference_image'):hand['reference_image']=str((destination/hand['reference_image']).resolve())
 write(destination/'pose.json',doc)
 return meta,doc

def rename(root,key,name,tags,notes):
 folder=entry(root,key);meta=metadata(folder)
 if not name.strip() or len(name)>100:raise ValueError('请输入 1–100 字的姿势名称')
 meta.update(name=name.strip(),tags=tags[:300],notes=notes[:2000]);write(folder/'entry.json',meta)

def trash(root,key):
 folder=entry(root,key);metadata(folder)
 trash_root=Path(root).resolve()/'.trash';trash_root.mkdir(exist_ok=True)
 if trash_root.is_symlink():raise ValueError('回收目录不能是符号链接')
 target=trash_root/(key+'_'+uuid.uuid4().hex)
 if folder.parent!=Path(root).resolve() or target.parent.resolve()!=trash_root.resolve():raise ValueError('回收路径无效')
 folder.rename(target)
 return target
