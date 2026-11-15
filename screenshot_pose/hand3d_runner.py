"""Original WiLoR/HaMeR inference in a dedicated environment, JSONL protocol."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('YOLO_AUTOINSTALL','false')

if os.name == 'nt':
    os.environ.setdefault('PYOPENGL_PLATFORM', 'win32')


def inference_weights(weights,backend):
    # Author's public forward_step uses backbone + mano_head + MANO. The
    # texture-supervised training checkpoint also saves its reconstruction
    # loss and texture decoder, neither present in that inference graph.
    auxiliary=('recoLoss.','texture_model.') if backend=='HAMER_TEXTURE' else ()
    ignored=[key for key in weights if key.startswith(auxiliary)]
    return {key:value for key,value in weights.items() if key not in ignored},ignored


class Runner:
    def __init__(self, config):
        self.config = config
        self.backend = None
        self.model = None
        self.detector = None

    def detections(self,image):
        import torch
        path=self.config.get('hand_detector')
        if not path or not Path(path).is_file():
            return []
        if self.detector is None:
            from ultralytics import YOLO,settings
            settings.update({'sync':False})
            # The official detector checkpoint contains a serialized YOLO
            # module. Restrict legacy loading compatibility to this call.
            original=torch.load
            def trusted_load(*args,**kwargs):
                kwargs.setdefault('weights_only',False)
                return original(*args,**kwargs)
            torch.load=trusted_load
            try:self.detector=YOLO(path).to('cuda')
            finally:torch.load=original
        prediction=self.detector.predict(image,conf=.4,verbose=False,device=0,save=False,show=False)[0]
        detections=[]
        for box in prediction.boxes:
            cls=int(box.cls.item());label=str(prediction.names[cls]).lower()
            if 'left' in label:right=0
            elif 'right' in label:right=1
            elif cls in (0,1):right=cls  # Original WiLoR demo class convention.
            else:continue
            detections.append(dict(box=box.xyxy[0].detach().cpu().tolist(),right=right,score=float(box.conf.item())))
        return detections

    def locate(self,image,locations,precision=False):
        import numpy as np
        from hand3d import associate_detection, regional_detection

        detections=self.detections(image)
        self.last_detections=detections
        results=[]
        for location in locations:
            region=location.get('regional_search') if precision else None
            if not region:
                results.append(associate_detection(location,detections))
                continue
            feet=np.asarray(region['feet'])
            def outside_feet(candidate):
                box=np.asarray(candidate['box'])
                return not len(feet) or np.linalg.norm(feet-(box[2:]+box[:2])/2,axis=1).min() > .8*max(box[2:]-box[:2])
            localized=associate_detection(location,[c for c in detections if outside_feet(c)])
            if localized is None:
                head=np.asarray(region['head'])
                center=(head.min(axis=0)+head.max(axis=0))/2
                extent=max(np.ptp(head,axis=0).max(), location['width']*.075)
                first,second=(self.locate_head_crop(image,location,center,extent*factor) for factor in (.78,1.05))
                if first and second:
                    a,b=np.asarray(first['box']),np.asarray(second['box'])
                    primary=max((first,second),key=lambda c:c['confidence'])
                    if primary['confidence'] >= .55 and np.max(np.abs(a-b)) < .3*max(a[2:]-a[:2]):
                        localized=primary
            results.append(localized)
        return results

    def locate_head_crop(self,image,location,center,extent):
        import numpy as np
        from hand3d import regional_detection

        lo=np.maximum(0,center-extent).astype(int)
        hi=np.minimum(image.shape[1::-1],center+extent).astype(int)
        x0,y0=lo; x1,y1=hi
        if min(hi-lo)<32:
            return None
        offset=np.tile(lo,2)
        candidates=[dict(c,box=(c['box']+offset).tolist()) for c in self.detections(image[y0:y1,x0:x1])]
        return regional_detection(location,candidates)

    def load(self, backend):
        import gc
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('WiLoR / HaMeR 需要可用的 PyTorch CUDA 环境')
        if self.backend == backend:
            return
        self.backend = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        model = self.config['models'][backend]
        sys.path.insert(0, model['repo'])
        source_module=model.get('source_module',backend.lower())
        module = importlib.import_module(source_module + '.models')
        configs = importlib.import_module(source_module + '.configs')
        cfg = configs.get_config(model['config'], update_cachedir=False)
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MANO.MODEL_PATH = self.config['mano']
        cfg.MANO.DATA_DIR = str(Path(self.config['mano']).parent)
        cfg.MANO.MEAN_PARAMS = model['mean_params']
        cfg.freeze()
        cls = getattr(module, 'WiLoR' if backend == 'WILOR' else 'HAMER')
        # Licensed MANO PKLs can contain legacy chumpy arrays. Compatibility
        # shims are limited to this isolated process, never the Blender runtime.
        import inspect
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
        for name, typ in [('bool', bool), ('int', int), ('float', float),
                          ('complex', complex), ('object', object), ('str', str), ('unicode', str)]:
            if name not in np.__dict__:
                setattr(np, name, typ)
        instance = cls(cfg, init_renderer=False)
        # Official PyTorch checkpoints include Lightning metadata and require
        # non-tensor deserialization. Only use the configured official assets.
        state = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        weights = state.get('state_dict', state)
        weights,ignored=inference_weights(weights,backend)
        result = instance.load_state_dict(weights, strict=False)
        allowed = ('discriminator.', 'renderer.', 'mesh_renderer.')
        bad = [key for key in result.missing_keys if not key.startswith(allowed)]
        unexpected = [key for key in result.unexpected_keys if not key.startswith(allowed)]
        if bad or unexpected:
            raise RuntimeError('网络权重与源码不匹配：' + str((bad[:8], unexpected[:8])))
        self.checkpoint_report=dict(inference_keys_verified=True,ignored_training_keys=len(ignored),
            ignored_training_prefixes=sorted(set(key.split('.')[0] for key in ignored)),
            missing_inference_keys=bad,unexpected_inference_keys=unexpected)
        self.model = instance.eval().to('cuda')
        self.cfg = cfg
        self.dataset = importlib.import_module(source_module + '.datasets.vitdet_dataset').ViTDetDataset
        self.backend = backend

    def infer(self, request):
        import cv2
        import numpy as np
        import torch
        backend = request['backend']
        if backend not in ('WILOR', 'HAMER','HAMER_TEXTURE'):
            raise ValueError('未知手部后端')
        self.load(backend)
        precision=request.get('screenshot_precision',False)
        scale=np.ones(2);sampling=None
        if precision:
            from hand_precision import image as precision_image
            image,scale,sampling=precision_image(request['image'],request.get('original_image'))
        else:image = cv2.imdecode(np.fromfile(request['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('手部参考图无法读取')
        locations=request.get('locations',[])
        if precision and locations and sampling['processed_size']!=[int(locations[0]['width']),int(locations[0]['height'])]:
            raise ValueError('缓存参考图尺寸已变化，请重新识别截图')
        if precision:
            from hand_precision import scaled_location
            locations=[scaled_location(v,scale) for v in locations]
        localized=self.locate(image,locations,precision) if locations else [None]*len(request['boxes'])
        boxes = np.asarray(request['boxes'], np.float32)*np.tile(scale,2)
        right = np.asarray(request['right'], np.float32)
        for i,detection in enumerate(localized):
            if detection:boxes[i]=detection['box']
        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not 1 <= len(boxes) <= 2 or right.shape != (len(boxes),):
            raise ValueError('手部裁剪参数无效')
        crop_count=3 if precision else 1
        if precision:
            # Independent predictions at three real crop extents, not enlarged
            # synthetic detail. The caller chooses using independent 2D evidence.
            boxes=np.repeat(boxes,3,axis=0);right=np.repeat(right,3)
            centers=(boxes[:,:2]+boxes[:,2:])/2
            half=(boxes[:,2:]-boxes[:,:2])/2*np.tile([1.,.8,1.2],len(boxes)//3)[:,None]
            boxes=np.column_stack((centers-half,centers+half))
            localized=[item for item in localized for _ in range(3)]
        dataset = self.dataset(self.cfg, image, boxes, right, rescale_factor=2.)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
        hands = []
        for batch_index,batch in enumerate(loader):
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.inference_mode():
                out = self.model(batch)
                if precision:
                    params=out['pred_mano_params'];n=len(batch['img'])
                    identity=torch.eye(3,device='cuda').reshape(1,1,3,3)
                    bind=self.model.mano(global_orient=identity.expand(n,1,3,3),hand_pose=identity.expand(n,15,3,3),betas=params['betas'],pose2rot=False)
                    rest_joints=bind.joints.detach().cpu().numpy()
                    local_rot=torch.cat((params['global_orient'],params['hand_pose']),dim=1)
                    parents=self.model.mano.parents.cpu().tolist();global_rot=[]
                    for j,parent in enumerate(parents):global_rot.append(local_rot[:,j] if parent<0 else global_rot[parent]@local_rot[:,j])
                    global_rot=torch.stack(global_rot,dim=1).detach().cpu().numpy()
            xyz = out['pred_keypoints_3d'].detach().float().cpu().numpy()
            cam = out['pred_cam'].detach().float().cpu().numpy()
            sizes = batch['box_size'].float().cpu().numpy()
            centers = batch['box_center'].float().cpu().numpy()
            flags = batch['right'].float().cpu().numpy()
            h, w = image.shape[:2]
            f = float(self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(w, h))
            for i, joints in enumerate(xyz):
                sign = 2 * flags[i] - 1
                physical = joints.copy()
                physical[:, 0] *= sign
                bs = sizes[i] * cam[i, 0]
                if not np.isfinite(bs) or bs <= 1e-6:
                    raise ValueError('手部相机估计无效')
                t = np.array([sign * cam[i, 1] + 2 * (centers[i, 0] - w / 2) / bs,
                              cam[i, 2] + 2 * (centers[i, 1] - h / 2) / bs, 2 * f / bs])
                view = physical + t
                if np.any(view[:, 2] <= 0):
                    raise ValueError('手部预测位于相机后方')
                xy = view[:, :2] / view[:, 2:] * f + [w / 2, h / 2]
                localization=localized[batch_index+i]
                if localization:
                    localization=dict(localization,box=(np.asarray(localization['box'])/np.tile(scale,2)).tolist(),
                                      wrist_region_gap_px=localization['wrist_region_gap_px']/float(np.mean(scale)))
                result=dict(joints_cv=joints.tolist(), landmarks_px=(xy/scale).tolist(), device='cuda',localization=localization)
                if precision:result.update(rest_joints_cv=rest_joints[i].tolist(),joint_rotations_cv=global_rot[i].tolist(),input_sampling=sampling,
                                          crop_expansion=[2.,1.6,2.4][(batch_index+i)%crop_count])
                hands.append(result)
        if precision:
            hands=[dict(hands[i],crop_candidates=hands[i+1:i+3]) for i in range(0,len(hands),3)]
        return hands


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--check-imports', action='store_true')
    args = p.parse_args()
    runner = Runner(json.loads(Path(args.config).read_text(encoding='utf8')))
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(args.config).resolve().parent/'ultralytics'))
    if args.check_imports:
        import torch
        if runner.config.get('hand_detector'):
            import ultralytics
            import dill
        for backend, value in runner.config['models'].items():
            sys.path.insert(0, value['repo'])
            module=value.get('source_module',backend.lower())
            importlib.import_module(module + '.models')
            importlib.import_module(module + '.datasets.vitdet_dataset')
        print(json.dumps(dict(imports_ok=True, cuda=torch.cuda.is_available(), torch=torch.__version__,
                              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))
        return
    for line in sys.stdin.buffer:
        try:
            if len(line) > 65536:
                raise ValueError('请求过大')
            with contextlib.redirect_stdout(sys.stderr):
                hands = runner.infer(json.loads(line))
            answer = dict(ok=True, hands=hands)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            answer = dict(ok=False, error=str(exc)[:600])
        print(json.dumps(answer, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
