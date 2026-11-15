"""Motion-stage observations and proxy metrics, with no Blender dependency.

Agreement with a model is not human ground truth or a naturalness score.
Positions are retained in world/camera axes; displays may center the pelvis.
"""
import math

LIMBS={'upper_arm':('shoulder','elbow'),'forearm':('elbow','wrist'),
       'thigh':('hip','knee'),'shin':('knee','ankle')}
EDGES=[('pelvis','chest'),('chest','neck'),('neck','head'),
       ('hip.L','hip.R'),('shoulder.L','shoulder.R')]
for _side in ('L','R'):
    for _chain in (('shoulder','elbow','wrist','hand'),('hip','knee','ankle','toe')):
        EDGES.extend((a+'.'+_side,b+'.'+_side) for a,b in zip(_chain,_chain[1:]))

def sub(a,b):return [x-y for x,y in zip(a,b)]
def norm(a):return math.sqrt(sum(x*x for x in a))
def mean(a,b):return [(x+y)*.5 for x,y in zip(a,b)]
def angle(a,b):
    denominator=norm(a)*norm(b)
    if denominator<1e-10:return None
    return math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(a,b))/denominator))))

def quaternion_angle(a,b):
    denominator=norm(a)*norm(b)
    if denominator<1e-10:raise ValueError('Invalid rotation')
    dot=min(1.,abs(sum(x*y for x,y in zip(a,b)))/denominator)
    return math.degrees(2*math.acos(dot))

def stats(values):
    values=sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:return dict(count=0,median=None,p95=None,max=None)
    def quantile(f):
        x=(len(values)-1)*f;i=int(x);j=min(i+1,len(values)-1)
        return values[i]*(1-(x-i))+values[j]*(x-i)
    return dict(count=len(values),median=quantile(.5),p95=quantile(.95),max=values[-1])

def joint_points(heads,tails):
    points={}
    for side in ('L','R'):
        for role,joint in (('upper_arm','shoulder'),('forearm','elbow'),('hand','wrist'),('thigh','hip'),('shin','knee'),('foot','ankle')):
            key=role+'.'+side
            if key in heads:points[joint+'.'+side]=list(heads[key])
        for role,joint in (('hand','hand'),('foot','toe')):
            if role+'.'+side in tails:points[joint+'.'+side]=list(tails[role+'.'+side])
    for joint,role in (('neck','neck'),('head','head')):
        if role in heads:points[joint]=list(heads[role])
    for joint,base in (('pelvis','hip'),('chest','shoulder')):
        if all(base+'.'+s in points for s in ('L','R')):points[joint]=mean(points[base+'.L'],points[base+'.R'])
    return points

def source_packet(document):
    skeleton=document.get('full_skeleton',{})
    if skeleton.get('schema')!='screenshot_pose/skeleton_v1':raise ValueError('Full source skeleton required')
    heads={};tails={};rest_heads={};rest_tails={}
    for role,name in skeleton['role_map'].items():
        bone=skeleton['bones'][name]
        heads[role]=bone['pose_head'];tails[role]=bone['pose_tail']
        rest_heads[role]=bone['rest_head'];rest_tails[role]=bone['rest_tail']
    return dict(points=joint_points(heads,tails),rest_points=joint_points(rest_heads,rest_tails),
                joint_count=len(skeleton['bones']),root_camera_position=heads['hips'],
                root_position_is_world_ground_truth=False)

def limb_agreement(source,target):
    result={}
    for side in ('L','R'):
        for role,(a,b) in LIMBS.items():
            a+='.'+side;b+='.'+side
            if all(k in p for p in (source,target) for k in (a,b)):
                result[role+'.'+side]=angle(sub(source[b],source[a]),sub(target[b],target[a]))
    return result

def leg_length(points):
    lengths=[norm(sub(points['hip.'+s],points['knee.'+s]))+norm(sub(points['knee.'+s],points['ankle.'+s])) for s in ('L','R')]
    value=sum(lengths)/len(lengths)
    if value<1e-8:raise ValueError('Invalid rest leg length')
    return value

def display_points(points,scale):
    center=points['pelvis']
    return {n:[x/scale for x in sub(p,center)] for n,p in points.items()}

def span(points):
    return [max(p[k] for p in points)-min(p[k] for p in points) for k in range(3)] if points else [0.,0.,0.]

def summarize(frames):
    by_stage={};changes={};names=[]
    for frame in frames:
        source=frame['source']['points'];previous=None
        for name,packet in frame['stages'].items():
            if name not in names:names.append(name)
            row=by_stage.setdefault(name,dict(agreement={},roots=[],knee_angles={'L':[],'R':[]}))
            row['roots'].append(packet['root_world'])
            for role,value in limb_agreement(source,packet['points']).items():row['agreement'].setdefault(role,[]).append(value)
            for side in ('L','R'):
                p=packet['points'];row['knee_angles'][side].append(angle(sub(p['knee.'+side],p['hip.'+side]),sub(p['ankle.'+side],p['knee.'+side])))
            if previous is not None:
                old_name,old=previous
                key=old_name+' -> '+name
                for role,q in packet['rotations'].items():
                    if role in old['rotations']:changes.setdefault(key,{}).setdefault(role,[]).append(quaternion_angle(old['rotations'][role],q))
            previous=(name,packet)
    return dict(schema='screenshot_pose/motion_audit_v1',human_ground_truth=False,naturalness_verified=False,
        valid_samples=len(frames),stages={name:dict(root_translation_span=span(row['roots']),
            limb_agreement_with_sam_degrees={role:stats(values) for role,values in row['agreement'].items()},
            knee_flex_degrees={side:stats(values) for side,values in row['knee_angles'].items()}) for name,row in by_stage.items()},
        adjacent_stage_rotation_changes_degrees={name:{role:stats(values) for role,values in roles.items()} for name,roles in changes.items()})
