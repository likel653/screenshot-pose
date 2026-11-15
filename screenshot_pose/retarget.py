"""Independent, rotation-only landmark retargeting for Blender armatures.

Joint vectors use armature-local coordinates. No bone, object, mesh, constraint
or rest-pose edits are made. An image cannot determine axial twist reliably;
the rest roll, transported with the torso, supplies that missing information.
"""

import math
import json
import re

import bpy
from mathutils import Matrix, Quaternion, Vector


ROLE_KEYS = ["hips", "spine", "chest", "neck", "head"] + [
    role + "." + side
    for side in ("L", "R")
    for role in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")
]

_ALIASES = {
    "hips": ("c_root_master.x", "hips", "pelvis", "hip", "腰", "下半身"),
    "spine": ("c_spine_01.x", "spine_fk", "spine_fk.001", "spine", "spine1", "腰椎", "上半身"),
    "chest": ("c_spine_03.x", "chest", "spine_fk.003", "spine_fk.002", "spine2", "spine3", "upperchest", "胸", "上半身2", "上半身２"),
    "neck": ("c_neck.x", "neck", "neck1", "首", "脖子"),
    "head": ("c_head.x", "head", "頭", "头"),
    "upper_arm": ("c_arm_fk", "upper_arm_fk", "upperarm_fk", "upper_arm", "upperarm", "arm", "腕", "肩臂", "上臂"),
    "forearm": ("c_forearm_fk", "forearm_fk", "forearm", "lowerarm", "elbow", "ひじ", "肘", "前臂"),
    "hand": ("c_hand_fk", "hand_fk", "hand", "wrist", "手首", "手腕", "手"),
    "thigh": ("c_thigh_fk", "thigh_fk", "thigh", "upleg", "upperleg", "足", "大腿"),
    "shin": ("c_leg_fk", "shin_fk", "shin", "calf", "leg", "lowerleg", "knee", "ひざ", "膝", "小腿"),
    "foot": ("c_foot_fk", "foot_fk", "foot", "ankle", "足首", "脚踝", "脚"),
}

_JOINT_PAIRS = {
    "neck": ("chest", "neck"), "head": ("neck", "head"),
    "upper_arm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist"),
    "hand": ("wrist", "hand"), "thigh": ("hip", "knee"),
    "shin": ("knee", "ankle"), "foot": ("ankle", "toe"),
}

_ROTATION_CONSTRAINTS = {
    "COPY_ROTATION", "COPY_TRANSFORMS", "DAMPED_TRACK", "LOCKED_TRACK",
    "TRACK_TO", "IK", "SPLINE_IK", "STRETCH_TO", "TRANSFORM",
    "ACTION", "ARMATURE", "FOLLOW_PATH", "LIMIT_ROTATION", "SHRINKWRAP",
}
_FK_PROPERTIES = ("IK_FK", "ik_fk_switch", "bend_all", "fingers_grasp")


def _normal(name):
    return re.sub(r"[\s_.:\-]", "", name.casefold())


def _name_parts(name):
    # Namespace removal supports mixamorig:Hips and FBX namespace prefixes.
    value = name.split(":")[-1].casefold()
    for prefix in ("mixamorig", "mixamo"):
        if value.startswith(prefix):
            value = value[len(prefix):].lstrip("_:.")
    category = ""
    for prefix in ("def-", "mch-", "org-"):
        if value.startswith(prefix):
            category, value = prefix[:-1], value[len(prefix):]
            break
    side = None
    for marker, label in (("left", "L"), ("right", "R"), ("左", "L"), ("右", "R")):
        if value.startswith(marker):
            side, value = label, value[len(marker):].lstrip("_. -")
            break
        if value.endswith(marker):
            side, value = label, value[:-len(marker)].rstrip("_. -")
            break
    if side is None:
        match = re.search(r"[_. -]([lr])$", value)
        if match:
            side, value = match.group(1).upper(), value[:match.start()]
        elif re.match(r"^[lr][_. -]", value):
            side, value = value[0].upper(), value[2:]
    return _normal(value), side, category


def mhx_profile(rig):
    return bool(rig and rig.type=='ARMATURE' and rig.get('MhxRig') and
                'MhaArmIk_L' in rig and 'upper_arm.fk.L' in rig.pose.bones)


def object_fk_switches(rig,mapping):
    if not mhx_profile(rig):return {}
    values={}
    if any(r in mapping for r in ('spine','chest')) and 'MhaSpineControl' in rig:values['MhaSpineControl']=False
    if any(r in mapping for r in ('head','neck')) and 'MhaNeckControl' in rig:values['MhaNeckControl']=False
    for side in ('L','R'):
        for roles,key in [(('upper_arm','forearm','hand'),'MhaArmIk_'),(('thigh','shin','foot'),'MhaLegIk_')]:
            if any(mapping.get(r+'.'+side,'').endswith('.fk.'+side) for r in roles) and key+side in rig:values[key+side]=0.
        if mapping.get('hand.'+side) and 'MhaFingerIk_'+side in rig:values['MhaFingerIk_'+side]=0.
    return values


def object_property_path(key):return '['+json.dumps(key)+']'


def _active_rotation_constraints(pbone):
    result = []
    for constraint in pbone.constraints:
        if constraint.mute or constraint.influence <= 1e-5:
            continue
        if mhx_profile(pbone.id_data):
            if constraint.type=='LIMIT_ROTATION':continue  # writable, still clamps in Blender
            if (constraint.type in ('COPY_ROTATION','COPY_TRANSFORMS') and constraint.mix_mode in ('AFTER','BEFORE','ADD','OFFSET') and constraint.owner_space=='LOCAL' and constraint.target==pbone.id_data):continue
        if constraint.type in _ROTATION_CONSTRAINTS:
            result.append(constraint)
        elif constraint.type == "CHILD_OF" and any(
            getattr(constraint, "use_rotation_" + axis, True) for axis in "xyz"
        ):
            result.append(constraint)
    return result


def centralized_fk_profile(armature):
    """Juno-style centralized switches; require controller structure, not rig name."""
    bones=armature.pose.bones
    return ('Properties_IK_FK' in bones and 'ik_left_upperarm' in bones['Properties_IK_FK']
            and all(n in bones for n in ('FK-UpperArm.L','FK-Forearm.L','FK-Wrist.L','FK-Thigh.L','FK-Knee.L','FK-Foot.L')))


def slider_fk_profile(rig):
    """Recognize translated FK sliders by their controls and driver wiring."""
    if not rig or rig.type!='ARMATURE' or not rig.animation_data:return False
    bones=rig.pose.bones
    required=['pelvis','hips','spine lower','spine upper','head neck lower','head neck upper']
    for long in ('left','right'):
        required.extend(('arm shoulder 2 '+long,'arm elbow '+long,'arm '+long+' wrist',
                         'leg thigh '+long,'leg knee '+long,'leg '+long+' ankle'))
    if not all(n in bones for n in required):return False
    for kind,part in (('Arm','arm elbow '),('Leg','leg knee ')):
        for long in ('Left','Right'):
            name=kind+'IKSwitch'+long
            if name not in bones:return False
            limits=[c for c in bones[name].constraints if c.type=='LIMIT_LOCATION' and c.owner_space=='LOCAL' and c.use_min_x and c.use_max_x]
            # max_x is itself driven by the global slider, so it varies at runtime.
            if not any(abs(c.min_x+.02)<1e-5 for c in limits):return False
            paths={c.path_from_id('influence') for c in bones[part+long.lower()].constraints if c.type=='IK' and c.target==rig}
            if not any(f.data_path in paths and any(v.type=='TRANSFORMS' and any(t.id==rig and t.bone_target==name and t.transform_type=='LOC_X' and t.transform_space=='LOCAL_SPACE' for t in v.targets) for v in f.driver.variables) for f in rig.animation_data.drivers):return False
    return True


def location_fk_switches(rig,mapping):
    if not slider_fk_profile(rig):return []
    switches=[]
    for side,long in (('L','Left'),('R','Right')):
        for kind,roles,names in (
            ('Arm',('upper_arm','forearm','hand'),('arm shoulder 2 '+long.lower(),'arm elbow '+long.lower(),'arm '+long.lower()+' wrist')),
            ('Leg',('thigh','shin','foot'),('leg thigh '+long.lower(),'leg knee '+long.lower(),'leg '+long.lower()+' ankle'))):
            if any(mapping.get(role+'.'+side)==name for role,name in zip(roles,names)):
                switches.append((rig.pose.bones[kind+'IKSwitch'+long],0,-.02))
    return switches


def location_switch_paths(rig,mapping):
    return {b.path_from_id('location') for b,index,value in location_fk_switches(rig,mapping)}


def snapshot_properties(armature,pbone):
    keys=set(_FK_PROPERTIES)
    if centralized_fk_profile(armature):
        if pbone.name=='Properties_IK_FK':keys.update(('ik_spine1','ik_left_upperarm','ik_right_upperarm','ik_left_thigh','ik_right_thigh'))
        if pbone.name=='Properties':keys.update('ik_'+side+'_finger_'+finger+'1' for side in ('left','right') for finger in ('thumb','index','middle','ring','pinky'))
    return {key:float(pbone[key]) for key in keys if key in pbone}


def auto_map(armature):
    """Map common Rigify, Auto-Rig Pro, Mixamo, generic and MMD bone names.

    Exact names win; ambiguous substring guesses are intentionally avoided.
    Callers can expose this mapping for manual correction before applying it.
    """
    if not armature or armature.type != "ARMATURE" or armature.pose is None:
        return {}
    result, used = {}, set()
    for role in ROLE_KEYS:
        base, _, side = role.partition(".")
        candidates = []
        aliases = [_normal(alias) for alias in _ALIASES[base]]
        for pbone in armature.pose.bones:
            if pbone.name in used:
                continue
            stem, found_side, category = _name_parts(pbone.name)
            if found_side != (side or None):
                continue
            if stem not in aliases:
                continue
            rank = aliases.index(stem)
            score = 1000 - rank * 8
            score -= {"def": 300, "org": 450, "mch": 600}.get(category, 0)
            score -= 160 if _active_rotation_constraints(pbone) else 0
            # Controllers take precedence over equally named deform bones.
            score += 80 if pbone.name.casefold().startswith("c_") else 0
            score += 60 if "_fk" in pbone.name.casefold() else 0
            candidates.append((score, pbone.name))
        if candidates:
            result[role] = max(candidates)[1]
            used.add(result[role])
    if centralized_fk_profile(armature):
        profile={'hips':'HIP-Spine1','spine':'FK-Spine1','chest':'FK-Spine3','neck':'FK-Neck','head':'FK-Head'}
        for side in ('L','R'):
            for role,name in [('upper_arm','UpperArm'),('forearm','Forearm'),('hand','Wrist'),('thigh','Thigh'),('shin','Knee'),('foot','Foot')]:profile[role+'.'+side]='FK-'+name+'.'+side
        result.update({role:name for role,name in profile.items() if name in armature.pose.bones})
    if slider_fk_profile(armature):
        # pelvis is the shared parent of torso and legs; hips only drives legs.
        result.update({'hips':'pelvis','spine':'spine lower','chest':'spine upper','neck':'head neck lower','head':'head neck upper'})
        for side,long in (('L','left'),('R','right')):
            for role,name in [('upper_arm','arm shoulder 2 '+long),('forearm','arm elbow '+long),('hand','arm '+long+' wrist'),('thigh','leg thigh '+long),('shin','leg knee '+long),('foot','leg '+long+' ankle')]:result[role+'.'+side]=name
    return result


def _update(armature):
    armature.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()


def absolute_targets(armature, targets, *, batch=False, passes=3, offset_controls=()):
    """Video-only depth batches: evaluate parents before solving their children."""
    ordered=sorted(targets,key=lambda n:len(armature.pose.bones[n].parent_recursive))
    groups=[]
    for name in ordered:
        depth=len(armature.pose.bones[name].parent_recursive)
        if not groups or not batch or groups[-1][0]!=depth:groups.append((depth,[]))
        groups[-1][1].append(name)
    for _ in range(passes):
        for depth,names in groups:
            for name in names:
                bone=armature.pose.bones[name]
                _set_rotation(bone,_to_local_rotation(bone,targets[name],compensate_offset=name in offset_controls))
            _update(armature)
    return ordered


def _as_vector(value):
    try:
        vector = Vector(value)
        if len(vector) == 3 and all(math.isfinite(v) for v in vector):
            return vector
    except (TypeError, ValueError):
        pass
    return None


def _direction(joints, start, end):
    a, b = _as_vector(joints.get(start)), _as_vector(joints.get(end))
    if a is None or b is None or (b - a).length < 1e-7:
        return None
    return (b - a).normalized()


def _frame(across, up):
    if across is None or up is None or across.length < 1e-7 or up.length < 1e-7:
        return None
    x = across.normalized()
    z = up - x * up.dot(x)
    if z.length < 1e-7:
        return None
    z.normalize()
    y = z.cross(x).normalized()
    return Matrix((x, y, x.cross(y))).transposed().to_quaternion()


def _body_frames(joints):
    up = _direction(joints, "pelvis", "chest")
    hips = _direction(joints, "hip.R", "hip.L")
    shoulders = _direction(joints, "shoulder.R", "shoulder.L")
    return _frame(hips if hips is not None else shoulders, up), _frame(
        shoulders if shoulders is not None else hips, up
    )


def _rest_joints(armature, mapping):
    points = {}
    for side in ("L", "R"):
        for role, start, end in (
            ("upper_arm", "shoulder", "elbow"), ("forearm", "elbow", "wrist"),
            ("hand", "wrist", "hand"), ("thigh", "hip", "knee"),
            ("shin", "knee", "ankle"), ("foot", "ankle", "toe"),
        ):
            bone = armature.data.bones.get(mapping.get(role + "." + side, ""))
            if bone:
                points[start + "." + side] = bone.head_local.copy()
                points[end + "." + side] = bone.tail_local.copy()
    for joint, role in (("pelvis", "hips"), ("chest", "chest"), ("neck", "neck"), ("head", "head")):
        bone = armature.data.bones.get(mapping.get(role, ""))
        if bone:
            points[joint] = bone.head_local.copy()
    for joint, pair in (("pelvis", "hip"), ("chest", "shoulder")):
        if pair + ".L" in points and pair + ".R" in points:
            points[joint] = (points[pair + ".L"] + points[pair + ".R"]) * 0.5
    return points


def snapshot_pose(armature):
    """Return JSON-serializable state for the next local pose application.

    Action references are stored by name, and the slot handle is retained for
    Blender 4.4+. Existing Actions remain intact because keyframing uses a copy.
    """
    if not armature or armature.type != "ARMATURE":
        raise ValueError("请选择骨架对象。")
    animation = armature.animation_data
    slot = getattr(animation, "action_slot", None) if animation else None
    return {
        "version": 1, "armature": armature.name,
        "had_animation_data": animation is not None,
        "action": animation.action.name if animation and animation.action else None,
        "action_fake_user": bool(animation.action.use_fake_user) if animation and animation.action else False,
        "action_slot_handle": slot.handle if slot else None,
        "object_properties": {k:armature[k] for k in object_fk_switches(armature,auto_map(armature))},
        "bones": {
            pbone.name: {
                "rotation_mode": pbone.rotation_mode,
                "matrix_basis": [list(row) for row in pbone.matrix_basis],
                "location": list(pbone.location), "scale": list(pbone.scale),
                "rotation_euler": list(pbone.rotation_euler),
                "rotation_quaternion": list(pbone.rotation_quaternion),
                "rotation_axis_angle": list(pbone.rotation_axis_angle),
                "properties": snapshot_properties(armature,pbone),
            }
            for pbone in armature.pose.bones
        },
    }


def _assign_action(armature, action, slot_handle=None):
    animation = armature.animation_data_create()
    animation.action = action
    if action and slot_handle is not None and hasattr(action, "slots"):
        for slot in action.slots:
            if slot.handle == slot_handle:
                animation.action_slot = slot
                break


def restore_pose(armature, snapshot, *, update=True):
    """Restore a snapshot without altering the rig's constraints or rest pose."""
    if not armature or armature.type != "ARMATURE":
        raise ValueError("请选择骨架对象。")
    if snapshot.get("armature") != armature.name:
        raise ValueError("备份属于另一骨架，未执行恢复。")
    action_name = snapshot.get("action")
    action = bpy.data.actions.get(action_name) if action_name else None
    if action_name and action is None:
        raise ValueError("原动作已被删除，无法完整恢复：" + action_name)
    if armature.animation_data or snapshot.get("had_animation_data"):
        _assign_action(armature, action, snapshot.get("action_slot_handle"))
    if action:
        action.use_fake_user = snapshot.get('action_fake_user', action.use_fake_user)
    for key,value in snapshot.get("object_properties",{}).items():armature[key]=value
    for name, state in snapshot.get("bones", {}).items():
        pbone = armature.pose.bones.get(name)
        if pbone is None:
            continue
        pbone.rotation_mode = state["rotation_mode"]
        pbone.matrix_basis = Matrix(state["matrix_basis"])
        # Restore all channels explicitly, including unused Euler/quaternion
        # channels, so mode changes and Euler winding survive a round trip.
        for channel in ("location", "scale", "rotation_euler", "rotation_quaternion", "rotation_axis_angle"):
            if channel in state:
                setattr(pbone, channel, state[channel])
        for key, value in state.get("properties", {}).items():
            pbone[key] = value
    if update:_update(armature)
    return {"restored": len(snapshot.get("bones", {}))}


def _set_rotation(pbone, rotation):
    rotation.normalize()
    if pbone.rotation_mode == "QUATERNION":
        pbone.rotation_quaternion = rotation
    elif pbone.rotation_mode == "AXIS_ANGLE":
        axis, angle = rotation.to_axis_angle()
        pbone.rotation_axis_angle = (angle, *axis)
    else:
        pbone.rotation_euler = rotation.to_euler(pbone.rotation_mode, pbone.rotation_euler)


def _to_local_rotation(pbone, absolute, *, compensate_offset=False):
    desired = absolute.to_matrix().to_4x4()
    desired.translation = pbone.matrix.translation
    kwargs = {}
    if pbone.parent:
        kwargs = {
            "parent_matrix": pbone.parent.matrix,
            "parent_matrix_local": pbone.parent.bone.matrix_local,
        }
    if compensate_offset or (mhx_profile(pbone.id_data) and any(not c.mute and c.influence>1e-5 and c.type in ('COPY_ROTATION','COPY_TRANSFORMS') and c.mix_mode in ('AFTER','BEFORE','ADD','OFFSET') and c.owner_space=='LOCAL' for c in pbone.constraints)):
        pre=pbone.bone.convert_local_to_pose(pbone.matrix_basis,pbone.bone.matrix_local,**kwargs).to_quaternion()
        q=absolute @ pbone.matrix.to_quaternion().inverted() @ pre
        desired=q.to_matrix().to_4x4();desired.translation=pbone.matrix.translation
    return pbone.bone.convert_local_to_pose(
        desired, pbone.bone.matrix_local, invert=True, **kwargs
    ).to_quaternion()


def _fk_switches(armature, mapping):
    """Only touch switches for limbs that actually map to known FK controls."""
    switches = []
    if centralized_fk_profile(armature):
        settings=armature.pose.bones['Properties_IK_FK']
        if any(mapping.get(r,'').startswith('FK-Spine') for r in ('spine','chest')) and 'ik_spine1' in settings:switches.append((settings,'ik_spine1',0.))
        for side,long in (('L','left'),('R','right')):
            for roles,key in [(('upper_arm','forearm','hand'),'upperarm'),(('thigh','shin','foot'),'thigh')]:
                if any(mapping.get(r+'.'+side,'').startswith('FK-') for r in roles):
                    prop='ik_'+long+'_'+key
                    if prop in settings:switches.append((settings,prop,0.))
            if mapping.get('hand.'+side,'').startswith('FK-'):
                fingers=armature.pose.bones.get('Properties')
                if fingers:
                    for finger in ('thumb','index','middle','ring','pinky'):
                        prop='ik_'+long+'_finger_'+finger+'1'
                        if prop in fingers:switches.append((fingers,prop,0.))
    names = {name.casefold(): pbone for name, pbone in ((p.name, p) for p in armature.pose.bones)}
    for side in ("L", "R"):
        suffix = "." + side.casefold()
        for role, rigify_parent, arp_parent in (
            ("upper_arm", "upper_arm_parent", "c_hand_ik"),
            ("thigh", "thigh_parent", "c_foot_ik"),
        ):
            limb_roles = ("upper_arm", "forearm", "hand") if role == "upper_arm" else ("thigh", "shin", "foot")
            controls = [mapping.get(part + "." + side, "").casefold() for part in limb_roles]
            if not any("_fk" in name for name in controls):
                continue
            for parent_name, prop in ((rigify_parent, "IK_FK"), (arp_parent, "ik_fk_switch")):
                pbone = names.get(parent_name + suffix)
                if pbone and prop in pbone:
                    switches.append((pbone, prop, 1.0))
    return switches


def _driver_controls_rotation(armature, pbone):
    animation = armature.animation_data
    if not animation:
        return False
    paths = {pbone.path_from_id(channel) for channel in (
        "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "matrix_basis"
    )}
    return any(not driver.mute and driver.data_path in paths for driver in animation.drivers)


def apply_pose(armature, mapping, joints, *, strength=1.0, keyframe=False, frame=None, rigify_fk=True, roles=None):
    """Apply absolute anatomical directions as local rotation channels.

    A strength of zero is a strict no-op. Unmapped bones keep their local
    channels. Mapped constrained/driven rotation channels are skipped and
    reported. FK switching covers Rigify and Auto-Rig Pro despite the legacy
    ``rigify_fk`` option name. Keyframes copy the current action first.
    """
    if not armature or armature.type != "ARMATURE":
        raise ValueError("请选择骨架对象。")
    if armature.mode == "EDIT":
        raise ValueError("请先退出骨架编辑模式。")
    if armature.library and not armature.override_library:
        raise ValueError("链接骨架只读，请先创建 Library Override。")
    if not math.isfinite(float(strength)):
        raise ValueError("姿势强度必须是有限数值。")
    strength = max(0.0, min(1.0, float(strength)))
    result = {"applied": [], "skipped": [], "warnings": []}
    if strength == 0:
        return result
    selected_mapping={r:n for r,n in mapping.items() if roles is None or r in roles}
    object_switches=object_fk_switches(armature,selected_mapping)
    location_switches=location_fk_switches(armature,selected_mapping)
    if rigify_fk:
        for key,value in object_switches.items():armature[key]=value
        for bone,index,value in location_switches:bone.location[index]=value
        if location_switches:result['touched']=[b.name for b,index,value in location_switches]
        if object_switches or location_switches:_update(armature)
    elif any(armature[k]!=v for k,v in object_switches.items()):result['warnings'].append('MHX IK/FK 对象开关尚未切换，外观可能不跟随')
    if not rigify_fk and any(abs(b.location[index]-value)>1e-5 for b,index,value in location_switches):result['warnings'].append('滑块 IK/FK 尚未切换，外观可能不跟随')
    if armature.data.pose_position == "REST":
        result["warnings"].append("骨架目前显示静置姿势；请切换为 Pose Position 查看结果。")
    valid, used = {}, set()
    for role in ROLE_KEYS:
        if roles is not None and role not in roles:
            continue
        name = mapping.get(role)
        if not name:
            result["skipped"].append(role)
            continue
        pbone = armature.pose.bones.get(name)
        if pbone is None or name in used:
            result["skipped"].append(role)
            result["warnings"].append(f"{role}：骨骼不存在或被重复映射（{name}）。")
            continue
        used.add(name)
        blocked = _active_rotation_constraints(pbone)
        if blocked or _driver_controls_rotation(armature, pbone):
            result["skipped"].append(role)
            reason = ", ".join(c.name for c in blocked) if blocked else "旋转驱动器"
            result["warnings"].append(f"{role} → {name}：由 {reason} 控制，请改映射到 FK 控制骨。")
            continue
        valid[role] = pbone
    if not valid:
        result["warnings"].append("没有可应用的骨骼；请检查骨骼映射。")
        return result

    rest_joints = _rest_joints(armature, mapping)
    rest_hips, rest_chest = _body_frames(rest_joints)
    pose_hips, pose_chest = _body_frames(joints)
    identity = Quaternion((1, 0, 0, 0))
    hips_delta = pose_hips @ rest_hips.inverted() if pose_hips and rest_hips else None
    chest_delta = pose_chest @ rest_chest.inverted() if pose_chest and rest_chest else hips_delta
    targets = {}
    for role, pbone in valid.items():
        base, _, side = role.partition(".")
        rest_orientation = pbone.bone.matrix_local.to_quaternion()
        if base in ("hips", "spine", "chest"):
            delta = hips_delta if base == "hips" else chest_delta
            if base == "spine" and hips_delta and chest_delta:
                delta = hips_delta.slerp(chest_delta, 0.5)
            if delta is None:
                result["skipped"].append(role)
                result["warnings"].append(f"{role}：缺少双肩、双髋或躯干数据，保持原姿势。")
                continue
            target = delta @ rest_orientation
        else:
            start, end = _JOINT_PAIRS[base]
            if side:
                start, end = start + "." + side, end + "." + side
            direction = _direction(joints, start, end)
            if direction is None:
                result["skipped"].append(role)
                result["warnings"].append(f"{role}：缺少有效 {start} / {end} 关键点，保持原姿势。")
                continue
            delta = (hips_delta if base in ("thigh", "shin", "foot") else chest_delta) or identity
            rest_direction = (pbone.bone.tail_local - pbone.bone.head_local).normalized()
            # Head and neck controls may have decorative/backward axes too.
            # Use their anatomical reference when available, retaining roll.
            if base in ("neck", "head"):
                rest_direction = _direction(rest_joints, start, end) or rest_direction
            transported = delta @ rest_direction
            target = transported.rotation_difference(direction) @ delta @ rest_orientation
        targets[role] = target.normalized()

    if not targets:
        return result
    # Capture before changing any parents; strength blends absolute pose space.
    _update(armature)
    initial_rotations = {role: valid[role].matrix.to_quaternion() for role in targets}
    switches = _fk_switches(armature, {role: valid[role].name for role in targets}) if rigify_fk else []
    if rigify_fk:
        for pbone, prop, value in switches:
            pbone[prop] = value
        _update(armature)
    else:
        for pbone, prop, value in _fk_switches(armature, {role: valid[role].name for role in targets}):
            if abs(float(pbone[prop]) - value) > 1e-5:
                result["warnings"].append(f"{pbone.name} 的 {prop} 尚未切换为 FK，肢体外观可能不跟随。")

    ordered = sorted(targets, key=lambda role: len(valid[role].parent_recursive))
    expected = {}
    for role in ordered:
        pbone = valid[role]
        target = initial_rotations[role].slerp(targets[role], strength)
        _set_rotation(pbone, _to_local_rotation(pbone, target))
        _update(armature)
        expected[role] = target
        result["applied"].append(role)
    # Some controller rigs route a hidden parent through a constraint to the
    # neck/chest rather than through the visible bone parent tree. Reconcile
    # after every mapped parent has evaluated, retaining the original targets.
    for _ in range(2):
        corrected = False
        for role, target in expected.items():
            pbone = valid[role]
            difference = pbone.matrix.to_quaternion().rotation_difference(target).angle
            difference = min(difference, abs(2 * math.pi - difference))
            if difference > math.radians(.05):
                _set_rotation(pbone, _to_local_rotation(pbone, target))
                _update(armature)
                corrected = True
        if not corrected:
            break
    for role, target in expected.items():
        actual = valid[role].matrix.to_quaternion()
        difference = actual.rotation_difference(target).angle
        difference = min(difference, abs(2 * math.pi - difference))
        if difference > math.radians(3):
            result["warnings"].append(f"{role} → {valid[role].name}：求值方向偏差 {math.degrees(difference):.1f}°，请检查父级约束或驱动。")

    if keyframe and result["applied"]:
        animation = armature.animation_data
        old_action = animation.action if animation else None
        if old_action:
            # Keep the snapshot's original action across save/reopen even when
            # assigning a private copy leaves the original with zero users.
            old_action.use_fake_user = True
        old_slot = getattr(animation, "action_slot", None) if animation else None
        slot_handle = old_slot.handle if old_slot else None
        action = old_action.copy() if old_action else bpy.data.actions.new(armature.name + " Screenshot Pose")
        action.name = (old_action.name if old_action else armature.name) + " Screenshot Pose"
        _assign_action(armature, action, slot_handle)
        at_frame = bpy.context.scene.frame_current if frame is None else float(frame)
        for role in result["applied"]:
            pbone = valid[role]
            channel = "rotation_quaternion" if pbone.rotation_mode == "QUATERNION" else (
                "rotation_axis_angle" if pbone.rotation_mode == "AXIS_ANGLE" else "rotation_euler"
            )
            pbone.keyframe_insert(data_path=channel, frame=at_frame, group=pbone.name)
        for key in object_switches:armature.keyframe_insert(data_path=object_property_path(key),frame=at_frame,group="MHX FK switches")
        if rigify_fk:
            for bone,index,value in location_switches:bone.keyframe_insert(data_path='location',index=index,frame=at_frame,group=bone.name)
        for pbone, prop, value in switches:
            pbone.keyframe_insert(data_path='["' + prop + '"]', frame=at_frame, group=pbone.name)
        if rigify_fk and location_switches:
            from .animation_preview import curves
            switch_paths=location_switch_paths(armature,selected_mapping)
            for curve in curves(action):
                if curve.data_path in switch_paths:
                    for point in curve.keyframe_points:point.interpolation='CONSTANT'
        _update(armature)
    return result
