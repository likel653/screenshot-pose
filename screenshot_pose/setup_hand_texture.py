"""Register and strictly validate the author's optional texture-supervised weights."""
import argparse,copy,datetime,hashlib,json,zipfile,shutil,uuid
from pathlib import Path


def verify_archive(path):
    """Reject incomplete modern torch archives before constructing a GPU model."""
    path=Path(path)
    with path.open('rb') as stream:signature=stream.read(4)
    if signature!=b'PK\x03\x04':return dict(format='legacy_or_unknown_deferred_to_torch')
    try:
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith('/data.pkl') or name=='data.pkl' for name in archive.namelist()):
                raise ValueError('此归档没有 PyTorch 检查点数据，请选择作者的 .ckpt 文件')
            bad=archive.testzip()
            if bad:raise ValueError('权重归档数据校验失败，需重新下载：'+bad)
            return dict(format='pytorch_zip',crc_checked=True,entries=len(archive.infolist()))
    except (zipfile.BadZipFile,EOFError) as exc:
        raise ValueError('权重下载不完整或归档损坏（缺少尾部目录/张量数据）；请在浏览器继续或重新下载完整 .ckpt，未启用增强模型') from exc


def cache_checkpoint(runtime,path,sha):
    folder=Path(runtime).resolve()/'hand3d/hamer_texture';folder.mkdir(parents=True,exist_ok=True)
    target=folder/('texture_supervised_'+sha[:12]+'.ckpt')
    if target.exists():
        with target.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=sha:raise ValueError('本地增强权重缓存内容不一致，未覆盖已有文件')
        return target
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part');shutil.copy2(path,temporary)
    with temporary.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=sha:raise ValueError('复制后的增强权重校验失败，原配置保留')
    temporary.replace(target);return target


def register(runtime,checkpoint):
    config=Path(runtime).resolve()/'hand3d/config.json';cfg=json.loads(config.read_text(encoding='utf8'))
    path=Path(checkpoint).resolve()
    if not path.is_file() or not 100_000_000<path.stat().st_size<8*1024**3:raise ValueError('请选择作者提供的完整增强版 .ckpt 文件')
    initial_stat=path.stat();archive=verify_archive(path)
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha==cfg['models']['HAMER'].get('sha256'):raise ValueError('这是原版 HaMeR 权重，不能标为纹理监督增强版')
    model=copy.deepcopy(cfg['models']['HAMER']);model.update(checkpoint=str(path),sha256=sha,source_module='hamer',
      source='https://github.com/gkarv/Hand-Texture-Module',variant='texture_supervised_author_checkpoint')
    cfg['models']['HAMER_TEXTURE']=model
    try:from .hand3d_runner import Runner
    except ImportError:from hand3d_runner import Runner
    runner=Runner(cfg);runner.load('HAMER_TEXTURE')
    cached=cache_checkpoint(runtime,path,sha)
    final_stat=path.stat()
    if (initial_stat.st_size,initial_stat.st_mtime_ns)!=(final_stat.st_size,final_stat.st_mtime_ns):
        raise ValueError('核验期间权重文件仍在变化，请等待下载完成后重新核验；原配置保留')
    cfg['models']['HAMER_TEXTURE'].update(checkpoint=str(cached),imported_from=str(path),network_validation=runner.checkpoint_report)
    # Only publish after the checkpoint fits the actual network architecture.
    backup=config.with_name('config.before_texture_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    backup.write_bytes(config.read_bytes());config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
    report=dict(ready=True,backend='HAMER_TEXTURE',sha256=sha,checkpoint=str(cached),strict_network_validation=True,archive=archive,network=runner.checkpoint_report)
    print(json.dumps(report),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--checkpoint',required=True)
    register(**vars(p.parse_args()))
