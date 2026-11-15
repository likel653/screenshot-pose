"""Pose interchange, validation and camera-to-Blender conversion (no bpy)."""
import json
import math
from pathlib import Path

SCHEMA = 'screenshot_pose/v1'
JOINT_INDICES = {
    'shoulder.L': 11, 'shoulder.R': 12, 'elbow.L': 13, 'elbow.R': 14,
    'wrist.L': 15, 'wrist.R': 16, 'hip.L': 23, 'hip.R': 24,
    'knee.L': 25, 'knee.R': 26, 'ankle.L': 27, 'ankle.R': 28,
    'toe.L': 31, 'toe.R': 32,
}
SEGMENTS = [
    ('pelvis', 'chest'), ('chest', 'neck'), ('neck', 'head'),
    ('shoulder.L', 'shoulder.R'), ('hip.L', 'hip.R'),
] + [(a + side, b + side) for side in ('.L', '.R') for a, b in
     [('chest', 'shoulder'), ('shoulder', 'elbow'), ('elbow', 'wrist'),
      ('wrist', 'hand'), ('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'toe')]]
# The chest is central, not sided.
SEGMENTS = [(a.replace('chest.L', 'chest').replace('chest.R', 'chest'), b)
            for a, b in SEGMENTS]


def validate(data):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('不是 Screenshot Pose v1 数据文件')
    for key in ('landmarks', 'world_landmarks'):
        points = data.get(key)
        if not isinstance(points, list) or len(points) != 33:
            raise ValueError(f'{key} 必须包含 33 个关节点')
        for p in points:
            if not isinstance(p, dict):
                raise ValueError('关节点格式无效')
            for axis in ('x', 'y', 'z'):
                value = p.get(axis)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('关节点必须使用有限数值坐标')
            for key2 in ('visibility', 'presence'):
                value = p.get(key2, 1.0)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('关节点置信度必须介于 0 和 1')
    info = data.get('image', {})
    for key in ('width', 'height'):
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 100000:
            raise ValueError('图片尺寸无效')
    return data


def read_pose(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('姿势 JSON 文件过大')
    return validate(json.loads(path.read_text(encoding='utf-8-sig')))


def _mean(*values):
    return tuple(sum(v[i] for v in values) / len(values) for i in range(3))


def to_joints(data, mode='WORLD', depth=1.0, mirror=False, yaw=0.0, threshold=0.35):
    """Return coordinates in a camera-aligned Blender frame and low-confidence names.

    X = image right, Y = away from the camera, Z = up. WORLD is the learned
    metric estimate; IMAGE uses the screenshot's projection with inferred depth.
    Mirror reflects X AND swaps anatomical sides so handedness remains correct.
    """
    validate(data)
    source = data['world_landmarks'] if mode == 'WORLD' else data['landmarks']
    aspect = data['image']['height'] / data['image']['width']
    points = [(p['x'], p['z'] * depth, -p['y'] * (aspect if mode == 'IMAGE' else 1.0))
              for p in source]
    if mode == 'HYBRID':
        screen, world = data['landmarks'], data['world_landmarks']
        # Match the image projection while retaining the independently adjustable
        # inferred depth. Torso length supplies an approximate metric scale.
        sh = _mean(*[(screen[i]['x'], screen[i]['y']*aspect, 0) for i in (11,12)])
        hp = _mean(*[(screen[i]['x'], screen[i]['y']*aspect, 0) for i in (23,24)])
        sw = _mean(*[(world[i]['x'],world[i]['y'],world[i]['z']) for i in (11,12)])
        hw = _mean(*[(world[i]['x'],world[i]['y'],world[i]['z']) for i in (23,24)])
        norm = lambda a,b: math.sqrt(sum((a[k]-b[k])**2 for k in range(3)))
        scale = norm(sw,hw)/max(norm(sh,hp),.05)
        points = [(p['x']*scale, world[i]['z']*depth, -p['y']*aspect*scale)
                  for i,p in enumerate(screen)]
        source = screen
    center = _mean(points[23], points[24])
    points = [tuple(p[i] - center[i] for i in range(3)) for p in points]
    joints = {name: points[index] for name, index in JOINT_INDICES.items()}
    joints['pelvis'] = (0.0, 0.0, 0.0)
    joints['chest'] = _mean(points[11], points[12])
    ear_center = _mean(points[7], points[8])
    joints['neck'] = _mean(joints['chest'], ear_center)
    joints['head'] = ear_center
    joints['hand.L'] = _mean(points[17], points[19])
    joints['hand.R'] = _mean(points[18], points[20])
    confidence = lambda i: min(source[i].get('visibility', 1), source[i].get('presence', 1))
    low = [name for name, idx in JOINT_INDICES.items() if confidence(idx) < threshold]
    if mirror:
        swap = lambda n: n[:-2] + ('.R' if n.endswith('.L') else '.L') if n.endswith(('.L', '.R')) else n
        joints = {swap(n): (-p[0], p[1], p[2]) for n, p in joints.items()}
        low = [swap(n) for n in low]
    c, s = math.cos(yaw), math.sin(yaw)
    joints = {n: (c*p[0]-s*p[1], s*p[0]+c*p[1], p[2]) for n, p in joints.items()}
    return joints, low


def feet_overlap(data):
    p = data['landmarks']
    aspect = data['image']['height'] / data['image']['width']
    def distance(a, b):
        return math.hypot(p[a]['x']-p[b]['x'], (p[a]['y']-p[b]['y'])*aspect)
    pairs = ((27,28),(29,30),(31,32))
    span = min(distance(29,31),distance(30,32))
    return span > .025 and all(
        not p[i].get('outside_frame') and min(p[i].get('visibility',1),p[i].get('presence',1)) >= .5
        for pair in pairs for i in pair
    ) and max(distance(a,b) for a,b in pairs) < span*.18


def diagnostics(data):
    """Flag ambiguous detections, without pretending to automatically fix them."""
    validate(data)
    p=data['landmarks']; aspect=data['image']['height']/data['image']['width']
    dist=lambda a,b: math.hypot(p[a]['x']-p[b]['x'],(p[a]['y']-p[b]['y'])*aspect)
    shoulder=max(dist(11,12),.05)
    warnings=list(data.get('backend_warnings',[]))
    warnings.extend(data.get('orientation',{}).get('warnings',[]))
    if dist(15,16)<shoulder*.18:
        warnings.append('双手腕几乎重合：可能是合掌，也可能左右手串位，请在图片校正中核对')
    if feet_overlap(data):
        warnings.append('两侧脚踝、脚跟和脚尖都几乎重合，请在图片校正中核对是否把两条腿接到了同一只脚')
    for side,shoulder_i,elbow,wrist in [('左',11,13,15),('右',12,14,16)]:
        if min(p[elbow].get('visibility',1),p[wrist].get('visibility',1))<.5:
            warnings.append(side+'肘或手腕存在遮挡，建议校正后应用')
    return warnings


def lower_body_cropped(data):
    """Detect the bottom-edge pileup from a crop using independent 3D scores."""
    if data.get('source') not in ('sam3d_mhr', 'rtmw3d_wholebody'):
        return False
    if set(data.get('edited_indices', ())) & set(range(23, 33)):
        return False
    screen, world = data['landmarks'], data['world_landmarks']
    if all(screen[i].get('outside_frame') for i in (25, 26, 27, 28)):
        return True
    for knee, ankle, toe in ((25, 27, 31), (26, 28, 32)):
        if min(screen[i]['y'] for i in (knee, ankle, toe)) < .9:
            return False
        for index in (ankle, toe):
            point = world[index]
            if min(point.get('visibility', 1.), point.get('presence', 1.)) >= .35:
                return False
    return True


def template_pose(image_path='', width=1000, height=1000):
    """Editable approximate human template; never represented as a detection."""
    points = [dict(x=.5, y=.18, z=0., visibility=1., presence=1.) for _ in range(33)]
    coords = {
        0:(.5,.13),7:(.54,.14),8:(.46,.14),
        11:(.60,.27),12:(.40,.27),13:(.68,.40),14:(.32,.40),
        15:(.73,.52),16:(.27,.52),17:(.75,.55),18:(.25,.55),
        19:(.76,.54),20:(.24,.54),21:(.74,.53),22:(.26,.53),
        23:(.56,.52),24:(.44,.52),25:(.57,.72),26:(.43,.72),
        27:(.58,.91),28:(.42,.91),29:(.58,.92),30:(.42,.92),
        31:(.62,.94),32:(.38,.94),
    }
    for i,(x,y) in coords.items():
        points[i].update(x=x, y=y)
    world = [dict(p, x=(p['x']-.5)*2, y=(p['y']-.52)*2*height/width) for p in points]
    return {'schema':SCHEMA, 'source':'manual_template',
            'image':{'path':str(image_path),'width':int(width),'height':int(height)},
            'person_index':0,'pose_count':0,'landmarks':points,'world_landmarks':world}
