"""Create a conventional T-rest source armature from corrected landmarks."""
import bpy
from mathutils import Vector,Matrix
from . import preview, retarget,pose_data,orientation_math as om,full_rig

def create_complete(context,settings,document):
    source=document['full_skeleton'];bones=source['bones']
    rotations={n:(Matrix(om.proper_rotation(v['pose_rotation']).tolist()),Matrix(om.proper_rotation(v['rest_rotation']).tolist())) for n,v in bones.items()}
    if context.object and context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    data=bpy.data.armatures.new('Screenshot Pose · Complete Rest')
    obj=bpy.data.objects.new('Screenshot Pose · Complete Source',data);context.scene.collection.objects.link(obj)
    obj['sp_source']=True;obj['sp_note']='All original source joints and parent relationships; keyed native pose, before target fitting.'
    obj.show_in_front=True;context.view_layer.objects.active=obj;obj.select_set(True)
    hips=[source.get('role_map',full_rig.SOURCE)['thigh.'+side] for side in ('L','R')]
    pose_center=sum((Vector(bones[n]['pose_head']) for n in hips),Vector())/2
    rest_center=sum((Vector(bones[n]['rest_head']) for n in hips),Vector())/2
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for name,item in bones.items():
            bone=data.edit_bones.new(name);bone.head=Vector(item['rest_head'])-rest_center;bone.tail=Vector(item['rest_tail'])-rest_center
            if bone.length<1e-6:bone.tail=bone.head+Vector((0,.005,0))
            bone.align_roll(rotations[name][1].col[2])
        for name,item in bones.items():
            if item.get('parent') and item['parent'] in data.edit_bones:data.edit_bones[name].parent=data.edit_bones[item['parent']]
    finally:bpy.ops.object.mode_set(mode='OBJECT')
    targets={}
    for name,item in bones.items():
        posed,rest=rotations[name];matrix=(posed@rest.transposed()@data.bones[name].matrix_local.to_3x3()).to_4x4()
        matrix.translation=Vector(item['pose_head'])-pose_center;targets[name]=matrix
    ordered=sorted(bones,key=lambda n:len(obj.pose.bones[n].parent_recursive))
    for name in ordered:
        obj.pose.bones[name].rotation_mode='QUATERNION';obj.pose.bones[name].matrix=targets[name];context.view_layer.update()
    for name in ordered:
        p=obj.pose.bones[name];p.keyframe_insert(data_path='location',frame=context.scene.frame_current)
        p.keyframe_insert(data_path='rotation_quaternion',frame=context.scene.frame_current)
    obj.location.x+=2.
    return obj,{'applied':ordered,'warnings':[],'complete':True}


def create(context, settings):
    if settings.json_path:
        document=pose_data.read_pose(bpy.path.abspath(settings.json_path))
        if document.get('full_skeleton') and not document.get('edited_indices'):return create_complete(context,settings,document)
    root = preview.root_for(settings)
    if not root:
        raise ValueError('请先识别截图或创建关节点')
    joints = {o['sp_joint']:o.location.copy() for o in preview.marker_objects(settings)}
    required = ['pelvis','chest','head'] + [n+s for s in ('.L','.R') for n in ('hip','knee','ankle','toe','shoulder','elbow','wrist','hand')]
    if any(n not in joints for n in required):
        raise ValueError('关节点不完整，无法生成源骨架')
    def length(a,b):
        return max((joints[b]-joints[a]).length,.005)
    body=length('pelvis','chest')
    neck=max(length('chest','head')*.5,body*.15)
    origin=Vector((0,0,0))
    spine=Vector((0,0,body*.35))
    chest=Vector((0,0,body*.7))
    shoulders=Vector((0,0,body))
    specs={
        'hips':(origin,spine,None),
        'spine':(spine,chest,'hips'),
        'chest':(chest,shoulders,'spine'),
        'neck':(shoulders,shoulders+Vector((0,0,neck)),'chest'),
        'head':(shoulders+Vector((0,0,neck)), shoulders+Vector((0,0,neck*2)),'neck'),
    }
    hip_width=length('hip.L','hip.R')*.5
    shoulder_width=length('shoulder.L','shoulder.R')*.5
    for side,sign in (('.L',1),('.R',-1)):
        shoulder=shoulders+Vector((sign*shoulder_width,0,0))
        elbow=shoulder+Vector((sign*length('shoulder'+side,'elbow'+side),0,0))
        wrist=elbow+Vector((sign*length('elbow'+side,'wrist'+side),0,0))
        hand=wrist+Vector((sign*length('wrist'+side,'hand'+side),0,0))
        hip=Vector((sign*hip_width,0,0))
        knee=hip-Vector((0,0,length('hip'+side,'knee'+side)))
        ankle=knee-Vector((0,0,length('knee'+side,'ankle'+side)))
        toe=ankle+Vector((0,-length('ankle'+side,'toe'+side),0))
        for role,start,end,parent in (
            ('upper_arm',shoulder,elbow,'chest'),('forearm',elbow,wrist,'upper_arm'+side),
            ('hand',wrist,hand,'forearm'+side),('thigh',hip,knee,'hips'),
            ('shin',knee,ankle,'thigh'+side),('foot',ankle,toe,'shin'+side)):
            specs[role+side]=(start,end,parent)
    if context.object and context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    data=bpy.data.armatures.new('Screenshot Pose · T Rest')
    obj=bpy.data.objects.new('Screenshot Pose · Source',data)
    context.scene.collection.objects.link(obj)
    obj.matrix_world=root.matrix_world.copy()
    obj['sp_source']=True
    obj['sp_note']='T-rest source with a keyed screenshot pose. Align rest poses before using external retarget tools.'
    context.view_layer.objects.active=obj
    obj.select_set(True)
    obj.show_in_front=True
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for name,(head,tail,parent) in specs.items():
            bone=data.edit_bones.new(name)
            bone.head=head;bone.tail=tail
            if parent:
                bone.parent=data.edit_bones[parent]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    result=retarget.apply_pose(obj,{r:r for r in retarget.ROLE_KEYS},joints,
                               keyframe=True,frame=context.scene.frame_current,rigify_fk=False)
    return obj,result
