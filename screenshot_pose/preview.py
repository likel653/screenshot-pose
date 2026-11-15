"""Editable landmark guides in the native Blender viewport."""
import math
import json
import bpy
from mathutils import Vector, Matrix
from . import pose_data


def root_for(settings):
    obj = bpy.data.objects.get(settings.preview_name)
    return obj if obj and obj.get('screenshot_pose_preview') else None


def marker_objects(settings):
    root = root_for(settings)
    return [o for o in root.children if 'sp_joint' in o] if root else []


def remove_preview(settings):
    root = root_for(settings)
    if root:
        for child in list(root.children_recursive):
            data = child.data if child.type == 'MESH' and child.get('sp_lines') else None
            bpy.data.objects.remove(child, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
        bpy.data.objects.remove(root, do_unlink=True)
    settings.preview_name = ''


def make_preview(context, settings, data):
    if getattr(settings,'candidate_state',''):
        from .pose_tools import cancel
        cancel(settings,True)
    from .pose_tools import effective_document
    data=effective_document(settings,data)
    settings.quality_warnings = json.dumps(pose_data.diagnostics(data),ensure_ascii=False) if pose_data.diagnostics(data) else ''
    joints, low = pose_data.to_joints(data, settings.coordinate_mode, settings.depth,
                                    settings.mirror, settings.yaw, settings.threshold)
    remove_preview(settings)
    root = bpy.data.objects.new('Screenshot Pose · 可编辑关节点', None)
    context.scene.collection.objects.link(root)
    root['screenshot_pose_preview'] = True
    root['sp_target'] = settings.target.name if settings.target else ''
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = .12
    target = settings.target
    if target and len(target.data.bones):
        body_height = max(b.head_local.z for b in target.data.bones) - min(b.head_local.z for b in target.data.bones)
        source_height = max(p[2] for p in joints.values()) - min(p[2] for p in joints.values())
        scale = max(body_height, .1) / max(source_height, .1)
        # The editor stays aligned to world X/Z, so Front View and G Y also
        # work with a rotated character object. Conversion happens on apply.
        world_scale = max(abs(v) for v in target.matrix_world.to_scale())
        scale *= world_scale
        root.location = target.matrix_world.translation + Vector((max(body_height*world_scale, 1)*.9, 0, body_height*world_scale*.52))
        root.scale = (scale,scale,scale)
    else:
        root.location = (2,0,1)
    settings.preview_name = root.name
    for name, point in joints.items():
        obj = bpy.data.objects.new('SP · ' + name, None)
        context.scene.collection.objects.link(obj)
        obj.parent = root
        obj.location = point
        obj['sp_joint'] = name
        obj['sp_low_confidence'] = name in low
        obj.empty_display_type = 'SPHERE' if name not in low else 'CUBE'
        obj.empty_display_size = .016 if settings.coordinate_mode == 'IMAGE' else .028
        obj.show_in_front = True
        obj.show_name = settings.show_names
        obj.color = (1,.35,.1,1) if name in low else ((.2,.6,1,1) if name.endswith('.L') else (.9,.2,.35,1))
    # In IMAGE mode the reference and markers share exactly the same projection.
    path = data['image'].get('path','')
    if settings.coordinate_mode in ('IMAGE','HYBRID') and path and not settings.mirror and abs(settings.yaw) < 1e-6:
        try:
            image = bpy.data.images.load(path, check_existing=True)
            ref = bpy.data.objects.new('SP · 截图参考', None)
            context.scene.collection.objects.link(ref)
            ref.parent = root
            ref.empty_display_type = 'IMAGE'
            ref.data = image
            aspect = data['image']['height']/data['image']['width']
            factor=1.
            if settings.coordinate_mode=='HYBRID':
                left,right=data['landmarks'][11],data['landmarks'][12]
                span=math.hypot(left['x']-right['x'],(left['y']-right['y'])*aspect)
                a,b=joints['shoulder.L'],joints['shoulder.R']
                factor=math.hypot(a[0]-b[0],a[2]-b[2])/max(span,1e-8)
            center_x = (data['landmarks'][23]['x'] + data['landmarks'][24]['x']) / 2
            center_y = (data['landmarks'][23]['y'] + data['landmarks'][24]['y']) / 2
            ref.location = ((.5-center_x)*factor, .1, (center_y-.5)*aspect*factor)
            ref.rotation_euler = (math.pi/2, 0, 0)
            ref.empty_display_size = max(1.,aspect)*factor
            ref.empty_image_depth = 'BACK'
            ref.color[3] = .45
            ref.hide_select = True
        except (RuntimeError, OSError):
            pass
    refresh_lines(context, settings)
    return low


def refresh_lines(context, settings):
    root = root_for(settings)
    if not root:
        return
    for obj in list(root.children):
        if obj.get('sp_lines'):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    joints = {o['sp_joint']:o.location.copy() for o in marker_objects(settings)}
    names = list(joints)
    mesh = bpy.data.meshes.new('SP Guide Lines')
    mesh.from_pydata([joints[n] for n in names],
                     [(names.index(a),names.index(b)) for a,b in pose_data.SEGMENTS if a in joints and b in joints], [])
    obj = bpy.data.objects.new('SP · 骨架连线（刷新后更新）', mesh)
    context.scene.collection.objects.link(obj)
    obj.parent = root
    obj['sp_lines'] = True
    obj.show_in_front = True
    obj.hide_render = True
    obj.hide_select = True


def read_joints(settings):
    bpy.context.view_layer.update()
    target = settings.target
    root = root_for(settings)
    if not root:
        raise ValueError('请先识别图片或创建手动关节点')
    transform = target.matrix_world.inverted_safe() if target else Matrix.Identity(4)
    joints = {o['sp_joint']:transform @ o.matrix_world.translation for o in marker_objects(settings)}
    # Absolute guide placement is irrelevant to rotation-only retargeting.
    center = joints.get('pelvis', Vector((0,0,0)))
    return {name:point-center for name,point in joints.items()}
