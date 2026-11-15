bl_info = {
    'name': 'Screenshot Pose · 截图转姿势', 'author': 'Screenshot Pose contributors',
    'version': (0, 39, 0), 'blender': (4, 2, 0),
    'location': '3D View > Sidebar > Screenshot Pose',
    'description': 'Local image pose estimation, editable landmarks and character retargeting',
    'category': 'Animation',
}

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import pose_data, preview, retarget, source_rig, contact_solver, hand_presets, projection_fit, orientation, foot_pose, full_rig, finger_pose
from . import screenshot_quality, screenshot_diagnostics, review_tools
from . import camera_tools, rgb_contact

from . import pose_tools, multiview_tools, hand_tools, hand_preview, character_tools, direct_tools, frame_tools, contact_targets, foot_tools, animation_preview, pose_library, video_tools, video_review
import importlib as _tools_importlib
_tools_importlib.reload(pose_tools)
_tools_importlib.reload(camera_tools)
_tools_importlib.reload(multiview_tools)
_tools_importlib.reload(hand_tools)
_tools_importlib.reload(review_tools)
_tools_importlib.reload(hand_preview)
_tools_importlib.reload(character_tools)
_tools_importlib.reload(direct_tools)
_tools_importlib.reload(frame_tools)
_tools_importlib.reload(contact_targets)
_tools_importlib.reload(foot_tools)

PACKAGE = __package__
HERE = Path(__file__).resolve().parent
_JOB = None


def _local_runtime_defaults():
    # Optional workstation configuration. On any other computer these paths
    # fail validation and the normal first-run setup is offered instead.
    try:
        cfg = json.loads((HERE/'local_runtime.json').read_text(encoding='utf-8'))
        if Path(cfg['python']).is_file() and (Path(cfg['runtime'])/'pose_landmarker_full.task').is_file():
            return cfg
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


LOCAL_RUNTIME = _local_runtime_defaults()
ROLE_LABELS = {'hips':'骨盆', 'spine':'脊柱', 'chest':'胸部', 'neck':'脖子', 'head':'头部',
               'upper_arm':'上臂', 'forearm':'前臂', 'hand':'手掌', 'thigh':'大腿', 'shin':'小腿', 'foot':'脚'}


def label(role):
    base = role.split('.')[0]
    return ROLE_LABELS.get(base, base) + (' · 左' if role.endswith('.L') else ' · 右' if role.endswith('.R') else '')


def preferences():
    entry = bpy.context.preferences.addons.get(PACKAGE)
    return entry.preferences if entry else None


def runtime_dir():
    prefs = preferences()
    path = prefs.runtime_dir if prefs else ''
    return Path(bpy.path.abspath(path)) if path else Path(bpy.utils.user_resource('SCRIPTS'))/'screenshot_pose_runtime'


def runtime_python(backend=None):
    prefs = preferences()
    if backend is None:
        backend=getattr(getattr(bpy.context.scene,'screenshot_pose',None),'backend','MEDIAPIPE')
    if backend in ('RTMW3D','SAM3D'):
        configured=prefs.gpu_python if prefs else LOCAL_RUNTIME.get('gpu_python','')
        return Path(bpy.path.abspath(configured)) if configured else runtime_dir()/'gpu_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if prefs and prefs.worker_python:
        return Path(bpy.path.abspath(prefs.worker_python))
    root = runtime_dir()/'python_env'
    return root/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def sam_root_args():
    prefs=preferences()
    root=prefs.sam_root if prefs else LOCAL_RUNTIME.get('sam_root','')
    return ['--native-root-path',bpy.path.abspath(root)] if root else []


def setup_python():
    prefs = preferences()
    if prefs and prefs.python_path:
        p = Path(bpy.path.abspath(prefs.python_path))
        if p.is_file():
            return p
        raise ValueError('Python 路径不存在，请在插件偏好设置中选择 python.exe')
    candidates = []
    for name in ('python', 'python3'):
        path = shutil.which(name)
        if path and 'WindowsApps' not in path:
            candidates.append(Path(path))
    candidates.extend([Path(sys.prefix)/'bin'/'python.exe', Path(sys.prefix)/'bin'/'python3',
                       Path(sys.prefix)/'python.exe'])
    for path in candidates:
        if path.is_file() and path.name.lower() not in ('blender.exe','blender'):
            return path
    raise ValueError('没有找到独立 Python，请在插件偏好设置中指定 Python 3.12 / 3.13')


def _redraw():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def stop_job():
    global _JOB
    job, _JOB = _JOB, None
    if not job:
        return
    if 'tick' in job:
        job['cancel']()
        return
    try:
        if job['process'].poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill','/PID',str(job['process'].pid),'/T','/F'],
                               capture_output=True,timeout=10,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            else:
                job['process'].terminate()
            # taskkill may return before the process handle signals, or be
            # unavailable in a restricted desktop. Always reap our own child.
            if job['process'].poll() is None:
                job['process'].terminate()
            try:
                job['process'].wait(timeout=3)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait(timeout=3)
    finally:
        job['output'].close()
        if job.get('record_folder'):
            try:(Path(job['record_folder'])/'screen.partial.avi').unlink(missing_ok=True)
            except OSError:pass


@bpy.app.handlers.persistent
def on_load_pre(_dummy):
    video_review.clean()
    animation_preview.cleanup()
    foot_tools.clean()
    direct_tools.stop_modals()
    direct_tools.restore_view_flags()
    pose_tools.cancel_all()
    stop_job()
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)


def start_job(scene, command, purpose, callback):
    global _JOB
    video_review.guard()
    if _JOB:
        raise ValueError('已有任务运行，请等待或取消')
    if scene.screenshot_pose.candidate_state:raise ValueError('请先确认或取消姿势候选预览')
    jobs = runtime_dir()/'jobs'
    jobs.mkdir(parents=True,exist_ok=True)
    log = jobs/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.log')
    output = log.open('w', encoding='utf-8')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    try:
        process = subprocess.Popen([str(x) for x in command], stdout=output, stderr=subprocess.STDOUT,
                                   env=env, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except Exception:
        output.close()
        raise
    _JOB = dict(process=process, output=output, scene=scene, callback=callback,
                purpose=purpose, log=log, started=time.monotonic())
    scene.screenshot_pose.status = purpose + '…'
    scene.screenshot_pose.last_log = str(log)
    bpy.app.timers.register(poll_job, first_interval=.4)
    _redraw()


def poll_job():
    global _JOB
    job = _JOB
    if not job:
        return None
    try:
        scene = job['scene']
        settings = scene.screenshot_pose
        if 'tick' in job:
            if job['tick']():
                _redraw()
                return .02
            _JOB = None
            _redraw()
            return None
        code = job['process'].poll()
        if code is None:
            settings.status = job['purpose'] + f"… {int(time.monotonic()-job['started'])} 秒"
            if job.get('progress') and job['progress'].is_file():
                try:
                    progress=json.loads(job['progress'].read_text(encoding='utf8'))
                    settings.status=progress['message']+f" · {int(time.monotonic()-job['started'])} 秒"
                except (OSError,ValueError,KeyError):pass
            _redraw()
            return .5
        job['output'].close()
        _JOB = None
        if code:
            lines = job['log'].read_text(encoding='utf-8', errors='replace').splitlines()
            from .job_errors import failure_message
            message = failure_message('\n'.join(lines), code)
            settings.status = '失败：' + message[:180]
        else:
            job['callback'](scene)
        _redraw()
    except Exception as exc:
        stop_job()
        print('Screenshot Pose:', exc)
        try:
            job['scene'].screenshot_pose.status = '失败：' + str(exc)[:180]
        except ReferenceError:
            pass
    return None


def set_mapping(settings):
    settings.mapping.clear()
    mapping = retarget.auto_map(settings.target) if settings.target else {}
    for role in retarget.ROLE_KEYS:
        item = settings.mapping.add()
        item.role = role
        item.bone = mapping.get(role,'') or ''
        item.enabled = True
    return sum(bool(row.bone) for row in settings.mapping)


def _apply_current_once(context, settings, joints_override=None, include_details=True, data_override=None, stage_observer=None):
    video_review.guard()
    target = settings.target
    if not target or target.type != 'ARMATURE':
        raise ValueError('请先选定目标角色骨架')
    if context.mode == 'EDIT_ARMATURE':
        raise ValueError('请先退出骨架编辑模式')
    root = preview.root_for(settings)
    if root and root.get('sp_target', '') != target.name:
        raise ValueError('目标角色已变更，请按参数重建关节点后应用')
    if not settings.mapping:
        set_mapping(settings)
    mapping = {row.role:row.bone for row in settings.mapping if row.enabled and row.bone}
    if not mapping:
        raise ValueError('没有匹配到骨骼，请展开骨骼映射手动选择')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('同一根骨骼不能映射到多个部位，请修正映射')
    joints = joints_override if joints_override is not None else preview.read_joints(settings)
    snapshot = retarget.snapshot_pose(target)
    # Store the original until explicit restore, even across repeated adjustments.
    if not target.get('screenshot_pose_backup'):
        target['screenshot_pose_backup'] = json.dumps(snapshot, ensure_ascii=False)
    else:
        # Older backups did not capture finger controls. Preserve their pre-edit values.
        backup=json.loads(target['screenshot_pose_backup'])
        for name,state in snapshot['bones'].items():
            if name in backup['bones']:
                props=backup['bones'][name].setdefault('properties',{})
                for prop,value in state['properties'].items():
                    props.setdefault(prop,value)
        for key,value in snapshot.get('object_properties',{}).items():backup.setdefault('object_properties',{}).setdefault(key,value)
        target['screenshot_pose_backup']=json.dumps(backup,ensure_ascii=False)
    try:
        data=data_override if data_override is not None else pose_data.read_pose(bpy.path.abspath(settings.json_path)) if settings.json_path else {}
        cropped=frame_tools.preserve_lower(settings,data)
        roles=set(frame_tools.upper_mapping(mapping)) if cropped else None
        result = retarget.apply_pose(target, mapping, joints, strength=settings.strength,
                                     keyframe=settings.keyframe, frame=context.scene.frame_current,
                                     rigify_fk=settings.switch_fk, roles=roles)
        if cropped:result['warnings'].append('图片裁掉了小腿与双脚，保留当前骨盆及腿脚，应用可见上身')
        if stage_observer:stage_observer('landmarks',target,result)
        if settings.strength>0 and full_rig.enabled(data,settings):
            from mathutils import Vector
            canonical,_=pose_data.to_joints(data,settings.coordinate_mode,settings.depth,settings.mirror,settings.yaw,settings.threshold)
            edited=any((o.location-Vector(canonical[o['sp_joint']])).length>1e-5 for o in preview.marker_objects(settings))
            if edited:result['warnings'].append('关节点已有手动调整，使用关节点驱动，跳过完整源骨架旋转')
            else:
                complete=full_rig.apply(target,mapping,settings,data,roles_to_apply=roles)
                result['full_rig']=complete;result['warnings'].extend(complete['warnings'])
                if settings.keyframe:contact_solver.key_touched(target,complete['touched'],context.scene.frame_current)
        if stage_observer:stage_observer('body',target,result)
        foot_state=None
        if settings.strength>0 and settings.foot_pose and not cropped:
            foot_state=foot_pose.apply(target,mapping,settings,data,calibration=getattr(settings,'video_sole_calibration',None))
        if settings.strength>0 and settings.ground_feet and not settings.contact_enabled:
            if not settings.switch_fk:
                raise ValueError('双脚接地需要开启自动切换 FK')
            contact=contact_solver.ground_feet(target,mapping,floor=settings.floor_height,
                    pelvis_offset=settings.pelvis_offset,clearance_ratio=settings.ankle_clearance,
                    root_name=settings.root_bone,flatten=settings.flatten_feet,sole_anchors=foot_state['sole_anchors'] if foot_state else None,sides=contact_solver.grounded_sides(settings))
            result['contact']=contact
            result['warnings'].extend(contact['warnings'])
            if settings.keyframe:
                contact_solver.key_touched(target,contact['touched'],context.scene.frame_current)
        if settings.strength>0 and settings.fit_projection:
            if not settings.switch_fk:
                raise ValueError('贴合原图需要开启自动切换 FK')
            protected=bool(getattr(settings,'preserve_source_pose',False) and result.get('full_rig') and not getattr(settings,'video_batch_updates',False))
            if cropped:
                fit=projection_fit.optimize(target,mapping,settings,data,upper_only=True)
            elif protected:
                from . import source_fit
                fit=source_fit.optimize(target,mapping,settings,data)
            else:fit=projection_fit.optimize(target,mapping,settings,data)
            if getattr(settings,'fit_torso',False) and not protected and not cropped:
                from . import torso_fit
                try:fit['torso']=torso_fit.refine(target,mapping,settings,fit)
                except (ValueError,FloatingPointError) as exc:fit['torso']=dict(enabled=True,accepted=False,reason=str(exc))
            result['fit']=fit;result['warnings'].extend(fit['warnings'])
            if settings.keyframe and fit['touched']:
                contact_solver.key_touched(target,fit['touched'],context.scene.frame_current)
        if foot_state:
            final_feet=foot_pose.apply(target,mapping,settings,data,result.get('fit'),foot_state['calibration'])
            final_feet['contacts']=foot_pose.contact_report(target,mapping,final_feet['calibration'],settings.floor_height)
            result['feet']=final_feet;result['warnings'].extend(final_feet['warnings'])
            if settings.keyframe:contact_solver.key_touched(target,final_feet['touched'],context.scene.frame_current)
        if stage_observer:stage_observer('feet',target,result)
        if include_details and settings.strength>0 and (settings.orient_head or settings.orient_hand_l or settings.orient_hand_r):
            detail=orientation.apply(target,mapping,settings,data,result.get('fit'))
            result['orientation']=detail;result['warnings'].extend(detail['warnings'])
            if settings.keyframe:
                contact_solver.key_touched(target,detail['touched'],context.scene.frame_current,detail['properties'])
        if stage_observer:stage_observer('orientation',target,result)
        if include_details and settings.strength>0 and ((settings.finger_joints and settings.auto_finger_curl) or any(hand_presets.manual_fingers(data,side) for side in ('L','R'))):
            fingers=finger_pose.apply(target,mapping,settings,data,orientation.calibration(target,mapping,settings))
            result['fingers']=fingers;result['warnings'].extend(fingers['warnings'])
            if settings.keyframe:contact_solver.key_touched(target,fingers['touched'],context.scene.frame_current,fingers['properties'])
        for side in ('L','R'):
            preset=hand_presets.selected(settings,data,side)
            if include_details and settings.strength>0 and preset!='NONE':
                hand=hand_presets.apply(target,side,preset,settings.strength,settings.hand_spread,getattr(settings,'hand_grip_'+side.lower(),.75),hand_presets.manual_fingers(data,side))
                if settings.keyframe:
                    contact_solver.key_touched(target,hand['touched'],context.scene.frame_current,hand['properties'])
        whole_active=bool(include_details and not cropped and getattr(settings,'whole_body',False) and settings.fit_projection and settings.strength>0 and result.get('fit') and (not result['fit'].get('source_guard') or settings.contact_enabled))
        prepared_contacts=None
        if include_details and settings.contact_enabled:
            if whole_active:prepared_contacts=contact_targets.prepare(target,mapping,settings,calibration=foot_state['calibration'] if foot_state else None)
            contacts=contact_targets.apply(target,mapping,settings,data,calibration=foot_state['calibration'] if foot_state else None,prepared=prepared_contacts)
            result['contacts']=contacts;result['warnings'].extend(contacts['warnings'])
            if settings.keyframe:contact_solver.key_touched(target,contacts['touched'],context.scene.frame_current,contacts['properties'])
            if result.get('fit') and contacts['touched']:
                contact_targets.refresh_fit(target,mapping,result['fit'])
        if whole_active:
            if result['fit'].get('source_guard') and not prepared_contacts:whole_active=False
        if whole_active:
            from . import whole_body_fit
            try:unified=whole_body_fit.refine(target,mapping,settings,data,result['fit'],prepared_contacts,result.get('contacts'))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:unified=dict(enabled=True,accepted=False,reason=str(exc),touched=[])
            result['fit']['whole_body']=unified
            if unified.get('accepted'):
                if unified.get('final_contacts'):
                    stale=set(result.get('contacts',{}).get('warnings',[]));result['warnings']=[note for note in result['warnings'] if note not in stale]
                    result['contacts']=unified['final_contacts'];result['warnings'].extend(result['contacts']['warnings'])
                if settings.keyframe:contact_solver.key_touched(target,unified['touched'],context.scene.frame_current,result.get('contacts',{}).get('properties',[]))
        if foot_state and result.get('feet'):
            result['feet']['contacts']=foot_pose.contact_report(target,mapping,foot_state['calibration'],settings.floor_height)
        if result.get('fit',{}).get('source_guard'):
            from . import camera_data,source_fit_math
            fit=result['fit'];actual=projection_fit.points(target,mapping);pred=camera_data.project(actual,fit['camera'])
            fit.update(points=actual.tolist(),after_projection=pred.tolist(),after_px=camera_data.error(pred,np.array(fit['observed']),np.array(fit['weights']))*fit['image']['width'])
            fit['source_guard']['final_max_link_change_deg']=max(source_fit_math.deviation(actual,np.array(fit['source_guard']['reference_points'])))
        if include_details and not getattr(settings,'video_batch_updates',False):
            result['hand_transfer']=screenshot_diagnostics.hands(target,mapping,settings,data,orientation.calibration(target,mapping,settings))
            if data_override is None and settings.json_path:result['observation_signature']=pose_tools.signature(settings)
        if settings.quality_warnings:
            result['warnings'].extend(json.loads(settings.quality_warnings))
        if stage_observer:stage_observer('final',target,result)
    except Exception:
        retarget.restore_pose(target, snapshot)
        raise
    preview.refresh_lines(context, settings)
    result['warnings']=list(dict.fromkeys(str(note) for note in result.get('warnings',[])))
    settings.last_result = json.dumps(result, ensure_ascii=False, default=str)
    count = len(result.get('applied',[]))
    notes = result.get('warnings',[])
    settings.status = f'已应用 {count} 个部位' + ('，已插入关键帧' if settings.keyframe else '（未插入关键帧）')
    if notes:
        settings.status += '；有提示，请查看结果'
    return result


def apply_current(context,settings):
    result=frame_tools.apply(context,settings,_solve_current)
    result['review_context']=review_tools.context_stamp(settings)
    settings.last_result=json.dumps(result,ensure_ascii=False,default=str)
    if result.get('fit',{}).get('source_guard'):settings.status+='；源三维姿势保护'
    return result

def _solve_current(context,settings):
    if settings.candidate_state:raise ValueError('请先确认或取消姿势候选预览')
    data=pose_data.read_pose(bpy.path.abspath(settings.json_path)) if settings.json_path else {}
    if settings.chosen_candidate:
        if settings.chosen_signature==pose_tools.signature(settings):return pose_tools.apply_chosen(context,settings,data)
        settings.chosen_candidate='';settings.chosen_signature=''
    if frame_tools.preserve_lower(settings,data):
        return _apply_current_once(context,settings)
    if full_rig.enabled(data,settings):
        if pose_tools.edited(settings,data):return _apply_current_once(context,settings)
        if getattr(settings,'preserve_source_pose',False) and not getattr(settings,'video_batch_updates',False):
            result=_apply_current_once(context,settings)
            result['full_source_comparison']={'chosen':'full_source_rotation','reason':'source_pose_protection','metric':'bounded_source_prior_not_accuracy'}
            settings.last_result=json.dumps(result,ensure_ascii=False,default=str)
            return result
        if settings.auto_full_quality and not settings.fit_projection:
            result=_apply_current_once(context,settings)
            result['full_source_comparison']={'chosen':'full_source_rotation','reason':'no_projection_keep_complete_source',
                'metric':'preserve_source_rotation_no_image_accuracy_claim'}
            settings.last_result=json.dumps(result,ensure_ascii=False,default=str)
            return result
        if settings.auto_full_quality and settings.fit_projection and settings.strength>0:
            target=settings.target
            if not target:return _apply_current_once(context,settings)
            initial=retarget.snapshot_pose(target);backup=target.get('screenshot_pose_backup');keyframe=settings.keyframe
            previous_result=settings.last_result;previous_status=settings.status
            from . import depth_selection
            mapping=pose_tools.mapping_for(settings);reference={};scores={}
            def observe(stage,rig,result):
                if stage=='body':reference.update(screenshot_diagnostics.rotations(rig,mapping))
            try:
                settings.keyframe=False
                native=_apply_current_once(context,settings,include_details=False,stage_observer=observe)
                native_error=float(native.get('fit',{}).get('after_px',1e6))
                native_points=projection_fit.points(target,mapping);common=native['fit']
                jobs=contact_targets.prepare(target,mapping,settings) if settings.contact_enabled else []
                pins=[((5 if job['kind']=='FOOT' else 2)+(6 if job['side']=='R' else 0),list(job['target'])) for job in jobs]
                native_metrics=depth_selection.measure(native_points,native_points,common['observed'],common['weights'],common['camera'],data['image']['width'],pins)
                degree,errors=screenshot_diagnostics.rotation_error(target,mapping,reference)
                scores['full_source_rotation']={**native_metrics,'valid':True,'source_guard_passed':native_metrics['valid'],'own_camera_px':native_error,'rotation_deg':degree,'per_role_deg':errors}
                retarget.restore_pose(target,initial)
                if backup is None:
                    if 'screenshot_pose_backup' in target:del target['screenshot_pose_backup']
                else:target['screenshot_pose_backup']=backup
                settings.full_source_rotation=False
                # Both trials use the identical body-only pipeline. Head,
                # fingers, presets and contacts are applied once, after choice.
                compatible=_apply_current_once(context,settings,include_details=False)
                compatible_error=float(compatible.get('fit',{}).get('after_px',1e6))
                compatible_metrics=depth_selection.measure(projection_fit.points(target,mapping),native_points,common['observed'],common['weights'],common['camera'],data['image']['width'],pins)
                degree,errors=screenshot_diagnostics.rotation_error(target,mapping,reference)
                scores['landmark_fit']={**compatible_metrics,'own_camera_px':compatible_error,'rotation_deg':degree,'per_role_deg':errors}
            finally:
                settings.full_source_rotation=True;settings.keyframe=keyframe;retarget.restore_pose(target,initial)
                settings.last_result=previous_result;settings.status=previous_status
                if backup is None:
                    if 'screenshot_pose_backup' in target:del target['screenshot_pose_backup']
                else:target['screenshot_pose_backup']=backup
            chosen,reason=screenshot_quality.body_choice(scores['full_source_rotation'],scores['landmark_fit'],data['image']['width'])
            choose_native=chosen=='full_source_rotation'
            try:
                settings.full_source_rotation=choose_native
                result=_apply_current_once(context,settings)
            finally:settings.full_source_rotation=True
            result['full_source_comparison']={'chosen':'full_source_rotation' if choose_native else 'landmark_fit',
                'full_projection_px':native_error,'landmark_projection_px':compatible_error,
                'metric':'fixed_camera_joint_depth_contact_and_source_rotation_priors_not_accuracy',
                'scores':scores,'reason':reason,'trial_details_applied':False}
            settings.last_result=json.dumps(result,ensure_ascii=False,default=str)
            return result
        return _apply_current_once(context,settings)
    candidates=[c for c in data.get('body_candidates',[]) if not c.get('manual_only')]
    if not (settings.auto_depth_select and settings.fit_projection and settings.strength>0 and len(candidates)>1 and not data.get('edited_indices')):
        return _apply_current_once(context,settings)
    from mathutils import Vector
    target=settings.target
    if not target:return _apply_current_once(context,settings)
    canonical,_=pose_data.to_joints(data,settings.coordinate_mode,settings.depth,settings.mirror,settings.yaw,settings.threshold)
    guides=preview.marker_objects(settings)
    if any((o.location-Vector(canonical[o['sp_joint']])).length>1e-5 for o in guides):
        result=_apply_current_once(context,settings);result['warnings'].append('关节点已有手动调整，跳过深度候选比较');settings.last_result=json.dumps(result,ensure_ascii=False);return result
    initial=retarget.snapshot_pose(target);backup=target.get('screenshot_pose_backup');keyframe=settings.keyframe
    from . import depth_selection
    trials=[];solutions=[];reference=None;common_camera=None;observed=None;weights=None;contact_priors=[]
    try:
        settings.keyframe=False
        for candidate in candidates[:3]:
            retarget.restore_pose(target,initial)
            if backup is None:
                if 'screenshot_pose_backup' in target:del target['screenshot_pose_backup']
            else:target['screenshot_pose_backup']=backup
            candidate_data=dict(data,world_landmarks=candidate['world_landmarks'])
            points,_=pose_data.to_joints(candidate_data,settings.coordinate_mode,settings.depth,settings.mirror,settings.yaw,settings.threshold)
            transform=target.matrix_world.inverted_safe().to_3x3()
            joints={name:transform@Vector(p) for name,p in points.items()}
            try:
                result=_apply_current_once(context,settings,joints,include_details=False,data_override=candidate_data)
                actual=projection_fit.points(target,pose_tools.mapping_for(settings));fit=result['fit']
                if reference is None:
                    reference=np.array(actual,copy=True);common_camera=fit['camera'];observed=fit['observed'];weights=fit['weights']
                    if settings.contact_enabled:
                        jobs=contact_targets.prepare(target,pose_tools.mapping_for(settings),settings)
                        contact_priors=[((5 if job['kind']=='FOOT' else 2)+(6 if job['side']=='R' else 0),list(job['target'])) for job in jobs]
                score=depth_selection.measure(actual,reference,observed,weights,common_camera,data['image']['width'],contact_priors)
                trials.append(dict(source=candidate['source'],own_camera_px=fit['after_px'],**score));solutions.append((candidate['source'],joints,candidate_data))
            except Exception as exc:
                trials.append({'source':candidate['source'],'valid':False,'error':str(exc)[:180]});solutions.append(None)
    finally:
        settings.keyframe=keyframe;retarget.restore_pose(target,initial)
        if backup is None:
            if 'screenshot_pose_backup' in target:del target['screenshot_pose_backup']
        else:target['screenshot_pose_backup']=backup
    index,decision=depth_selection.choose(trials,data['image']['width']);chosen,joints,document=solutions[index]
    result=_apply_current_once(context,settings,joints,data_override=document);result['observation_signature']=pose_tools.signature(settings)
    result['depth_selection']=dict(chosen=chosen,candidates=trials,**decision,metric='fixed_camera_plus_joint_depth_contact_priors')
    settings.last_result=json.dumps(result,ensure_ascii=False);return result


class SPPreferences(AddonPreferences):
    pose_library_dir: StringProperty(name='本地姿势库目录',subtype='DIR_PATH',default='',description='留空使用 Documents/ScreenshotPoseLibrary；整个文件夹可备份或移交')
    bl_idname = PACKAGE
    runtime_dir: StringProperty(name='识别环境目录', subtype='DIR_PATH',default=LOCAL_RUNTIME.get('runtime',''), description='独立安装环境和模型，留空使用 Blender 用户脚本目录')
    python_path: StringProperty(name='安装用 Python', subtype='FILE_PATH', description='通常可留空；可指定 Python 3.12 / 3.13 的 python.exe')
    gpu_python: StringProperty(name='GPU 识别环境 Python',subtype='FILE_PATH',default=LOCAL_RUNTIME.get('gpu_python',''),description='独立 CUDA 环境，与 CPU 识别环境分开')
    worker_python: StringProperty(name='已有识别环境 Python（可选）', subtype='FILE_PATH',default=LOCAL_RUNTIME.get('python',''), description='已有 MediaPipe 环境时可直接选择其 python.exe，免去再次安装')
    sam_root: StringProperty(name='SAM 原生运行器目录',subtype='DIR_PATH',default=LOCAL_RUNTIME.get('sam_root',''),description='包含 native_build/vulkan/bin 的目录；权重位于识别环境的 sam3d_gguf 子目录')
    rgb_contact_root: StringProperty(name='RGB MoCap 程序目录', subtype='DIR_PATH', description='视频支撑精修使用包含 mocap/correction.py 的 RGB MoCap 目录')
    rgb_contact_python: StringProperty(name='RGB MoCap Python', subtype='FILE_PATH', description='与 RGB MoCap 接触模块匹配，已安装 NumPy 和 SciPy 的 Python')

    def draw(self, context):
        self.layout.label(text='首次准备会下载 MediaPipe、依赖及官方模型。识别在本机完成。')
        self.layout.prop(self,'runtime_dir')
        self.layout.prop(self,'python_path')
        self.layout.prop(self,'worker_python')
        self.layout.prop(self,'gpu_python')
        self.layout.prop(self,'sam_root')
        self.layout.prop(self,'rgb_contact_root')
        self.layout.prop(self,'rgb_contact_python')
        self.layout.operator('screenshot_pose.setup',icon='IMPORT')


class SPMapping(PropertyGroup):
    role: StringProperty()
    bone: StringProperty(name='目标骨骼')
    enabled: BoolProperty(default=True)


def target_poll(self, obj):
    return obj.type == 'ARMATURE' and not obj.get('sp_source')


def target_changed(self, context):
    video_review.cancel()
    if getattr(self,'candidate_state',''):pose_tools.cancel(self,True)
    set_mapping(self)
    character_tools.changed(self)
    self.direct_undo_state=''
    self.last_result=''
    if self.preview_name:
        self.status='目标角色已变更：映射已更新，请重建关节点'


def ui_part_changed(self,context):
    if self.ui_part!='BODY':self.local_scope=self.ui_part
    if self.ui_part.startswith(('HAND_','ARM_')):self.hand_edit_side=self.ui_part[-1];self.hand_adopt_side=self.ui_part[-1]

class SPSettings(PropertyGroup):
    screenshot_hands:EnumProperty(name='截图手部精修',items=[('AUTO','自动 GPU · HaMeR','已准备时自动精修双手；缺少模型或失败保留原观测，不自动下载'),('OFF','关闭自动精修','仅使用身体识别与现有局部观测'),('HAMER','HaMeR · CUDA','每次截图后用专用手部模型精修'),('HAMER_TEXTURE','增强 HaMeR · CUDA','需准备用户提供的增强版权重')],default='AUTO')
    preserve_source_pose:BoolProperty(name='保护源三维姿势',default=True,description='完整骨架模式优先保留源深度和躯干，只允许小幅图片微调；手动关节点和显式支撑优先。关闭可恢复自由拟合。')
    hand_adopt_side:EnumProperty(name='采用到',items=[('L','左手','人物自身左手'),('R','右手','人物自身右手'),('BOTH','双手','两个结果都通过检查后采用')],default='L')
    hand_adopt_component:EnumProperty(name='采用范围',items=[('ALL','整手','掌向与手指一同采用'),('PALM','仅掌向','保留现有手指的掌内形状'),('FINGERS','仅手指','保留现有掌向')],default='ALL')
    ui_section:EnumProperty(name='工作区',items=[('GENERATE','生成','截图和图片转姿势'),('CORRECT','修正','按部位修正与复核'),('ADVANCED','设置','模型、映射与校准'),('LIBRARY','姿势库','本地姿势素材'),('VIDEO','视频','视频识别与动画')],default='GENERATE',options={'SKIP_SAVE'})
    ui_part:EnumProperty(name='修正部位',items=review_tools.review_data.PARTS,default='BODY',update=ui_part_changed)
    video_issues: CollectionProperty(type=video_review.SPVideoIssue)
    video_issue_index: IntProperty(default=0,min=0)
    video_review_markers: StringProperty(options={'HIDDEN'})
    video_review_status: StringProperty(default='生成动作后刷新问题列表；也可在任意帧开始局部修正')
    video_reference_path: StringProperty(options={'HIDDEN'})
    video_reference_info: StringProperty(options={'HIDDEN'})
    video_continuous: BoolProperty(name='识别阶段连续跟踪',default=True,description='共享人物框，参考相邻帧选择手掌候选；失跟或切镜头时重置')
    video_lock_part: EnumProperty(name='接触部位',items=[('FOOT_L','左脚',''),('FOOT_R','右脚',''),('BOTH_FEET','双脚',''),('HAND_L','左手',''),('HAND_R','右手','')],default='FOOT_L')
    video_lock_start: IntProperty(name='接触开始帧',default=1,min=-1040000,max=1040000)
    video_lock_end: IntProperty(name='接触结束帧',default=24,min=-1040000,max=1040000)
    video_lock_fade: IntProperty(name='区间外过渡帧',default=4,min=1,max=60)
    video_lock_pelvis: BoolProperty(name='允许骨盆有界升降',default=False)
    video_lock_pelvis_limit: FloatProperty(name='骨盆升降上限',default=.15,min=0.,max=2.,subtype='DISTANCE')
    video_edit_scope: EnumProperty(name='修正部位',items=pose_tools.SCOPES+[('WRIST_L','仅左手腕','只传播左手腕修正，保留前臂与手指通道'),('WRIST_R','仅右手腕','只传播右手腕修正，保留前臂与手指通道')],default='HAND_R')
    video_edit_start: IntProperty(name='起始衔接帧',default=1,min=-1048574,max=1048574)
    video_edit_end: IntProperty(name='结束衔接帧',default=17,min=-1048574,max=1048574)
    video_input: EnumProperty(name='视频来源',items=[('FILE','本地视频','选择已有视频文件'),('SCREEN','屏幕录制','框选正在播放的视频区域并按指定时长录制')],default='FILE')
    video_record_seconds: FloatProperty(name='录制时长（秒）',default=5.,min=.5,max=120.)
    video_record_fps: IntProperty(name='录制帧率',default=15,min=1,max=60)
    video_record_identify: BoolProperty(name='录完自动识别',default=True)
    video_capture_summary: StringProperty(options={'HIDDEN'})
    video_path: StringProperty(name='本地视频',subtype='FILE_PATH')
    video_backend: EnumProperty(name='视频识别模式',items=[('SAM3D','SAM · 高质量 GPU','逐帧使用现有 SAM 全身和局部模型，速度较慢'),('MEDIAPIPE','连续跟踪 · 快速 CPU','使用视频跟踪模式；不提供独立手指精修')],default='SAM3D')
    video_start: FloatProperty(name='开始秒',default=0.,min=0.)
    video_end: FloatProperty(name='结束秒（0 为结尾）',default=0.,min=0.)
    video_fps: FloatProperty(name='每秒识别帧数',default=12.,min=0.,max=240.,description='0 表示识别视频每一帧；动画时长不受采样率影响')
    video_sampling: EnumProperty(name='识别采样',items=[('ADAPTIVE','运动自适应','根据画面运动在快速动作区增加真实识别采样'),('FIXED','固定频率','按每秒识别帧数均匀采样')],default='ADAPTIVE')
    video_sampling_max: FloatProperty(name='快速动作最高采样率',default=30.,min=1.,max=120.)
    video_interpolation: EnumProperty(name='采样间运动',items=[('HERMITE','有界曲线 · 逐帧烘焙','按真实时间戳、有界切线和局部四元数曲线重新求值'),('SLERP','球面线性 · 逐帧烘焙','按真实时间戳做最短弧球面插值'),('LEGACY','原采样关键帧 · 对照','保留旧稀疏关键帧模式')],default='HERMITE')
    video_root_motion: EnumProperty(name='骨盆运动',items=[('HEIGHT','相对高度','根据画面骨盆高度变化估计起伏，不恢复绝对世界轨迹'),('SCREEN','相对画面位移','根据固定镜头画面估计左右与高度运动；不估计前后位移'),('IN_PLACE','原地 · 对照','保留原位置')],default='HEIGHT')
    video_root_gain: FloatProperty(name='骨盆运动幅度',default=1.,min=0.,max=2.)
    video_auto_support: BoolProperty(name='自动脚部支撑',default=True,description='从连续低速、接近画面地面的脚推断支撑，逐帧 FK 求解；跳跃/抬脚不强制接地，仍需回放复核')
    video_support_limit: FloatProperty(name='支撑骨盆调整上限（腿长比）',default=.25,min=0.,max=.4)
    video_cache: StringProperty(name='识别缓存',subtype='FILE_PATH')
    video_start_frame: IntProperty(name='动画起始帧',default=1,min=1,max=1000000)
    video_temporal_mode: EnumProperty(name='连续性处理',items=[('ADAPTIVE','自适应 · 保留快动作','按运动速度和转折减弱平滑，按时间戳居中拟合，减少抖动与动作削弱'),('LEGACY','固定平滑 · 对照','相邻姿势加权平均，供对照'),('OFF','关闭平滑','仍可独立开启异常跳变复核')],default='ADAPTIVE')
    video_smooth: FloatProperty(name='平滑范围（秒）',default=.12,min=0.,max=.5,description='离线使用前后帧处理，不增加时间轴延迟；范围过大会削弱动作细节；0 为关闭')
    video_max_gap: FloatProperty(name='允许补间中断（秒）',default=.3,min=0.,max=2.,description='只允许短暂漏检的间隔插值；超过此范围会停止生成')
    video_native: BoolProperty(name='使用 SAM 完整身体旋转',default=True,description='整段使用同一来源，不逐帧切换身体候选')
    video_fit: BoolProperty(name='逐帧贴合原图',default=False,description='可能改善单帧轮廓，但会增加耗时和帧间抖动')
    video_set_range: BoolProperty(name='设置播放范围',default=True)
    video_report: StringProperty(options={'HIDDEN'})
    video_stabilize: BoolProperty(name='复核异常跳变',default=True,description='仅修补可靠前后帧之间的低置信度孤立翻转；其余疑点只记录，不替换真实大动作')
    video_hand_continuity: BoolProperty(name='手指缺失连续衔接',default=True,description='按实际手指更新状态补间短暂缺失，有效观测不作为缺失补间；不跨已标记的镜头切换或失跟')
    video_palm_candidates: BoolProperty(name='实验掌面候选路径',default=False,options={'HIDDEN'},description='实验开关，尚未通过动作稳定性验收；默认关闭')
    video_hand_gap: FloatProperty(name='手指缺失补间上限（秒）',default=.6,min=0.,max=1.,description='仅对可靠前后姿势夹住的局部手指缺失补间，不跨人物失跟、镜头切换或长缺失；估计结果单独记录')
    video_wrist_repair: BoolProperty(name='手腕可疑短段修正',default=True,description='仅修正可靠前后帧夹住、二维掌面变化较小的短时误翻；可靠三维转腕保留，不跨失跟或人工手势')
    video_wrist_gap: FloatProperty(name='手腕修正跨度（秒）',default=.35,min=0.,max=.5)
    video_local_memory: BoolProperty(name='短遮挡局部姿势记忆',default=True,description='手腕相对前臂、手指相对手掌短暂保留最近可靠姿势，超时平滑退出；不跨失跟、镜头切换或人工编辑')
    video_local_memory_seconds: FloatProperty(name='局部记忆上限（秒）',default=.35,min=.05,max=.5)
    video_optimize: BoolProperty(name='加速角色适配',default=True,description='复用本次角色鞋底校准，按骨架层级批量更新；不改变单图处理')
    library_prompt: BoolProperty(name='识别完成后询问入库',default=False)
    library_items: CollectionProperty(type=pose_library.SPLibraryItem)
    library_index: IntProperty(default=0)
    library_search: StringProperty(name='搜索名称 / 标签')
    library_status: StringProperty(default='点击刷新读取本地姿势库')
    fresh_pose_start: BoolProperty(name='新图 / 新帧从角色基准开始',default=True,description='清除上一姿势的骨盆位移和手指残留；关闭后在当前动画姿势基础上叠加')
    ground_mode: EnumProperty(name='接地脚',items=[('BOTH','双脚',''),('L','仅左脚',''),('R','仅右脚','')],default='BOTH')
    foot_side: EnumProperty(name='脚部',items=[('L','人物左脚',''),('R','人物右脚','')],default='L')
    foot_refine_report: StringProperty(options={'HIDDEN'})
    contact_enabled: BoolProperty(name='独立手脚接触',default=False,description='在图片拟合后求解用户指定的接触目标；脚部支撑替代旧接地选项')
    support_l: EnumProperty(name='左脚',items=[('FREE','自由','保留识别姿势'),('FLOOR','地面','鞋底位于地面高度'),('ELEVATED','高处支撑','鞋底位于指定高度')],default='FREE')
    support_r: EnumProperty(name='右脚',items=[('FREE','自由','保留识别姿势'),('FLOOR','地面','鞋底位于地面高度'),('ELEVATED','高处支撑','鞋底位于指定高度')],default='FREE')
    support_height_l: FloatProperty(name='支撑高度',default=.5,subtype='DISTANCE')
    support_height_r: FloatProperty(name='支撑高度',default=.5,subtype='DISTANCE')
    support_target_l: PointerProperty(name='支撑目标',type=bpy.types.Object,description='可选：使用物体世界位置的 XY 和支撑高度 Z；支撑面保持水平')
    support_target_r: PointerProperty(name='支撑目标',type=bpy.types.Object,description='可选：使用物体世界位置的 XY 和支撑高度 Z；支撑面保持水平')
    knee_target_l: PointerProperty(name='膝盖朝向目标',type=bpy.types.Object,description='可选：用独立物体指定膝盖弯曲平面')
    knee_target_r: PointerProperty(name='膝盖朝向目标',type=bpy.types.Object,description='可选：用独立物体指定膝盖弯曲平面')
    support_move_pelvis: BoolProperty(name='必要时调整骨盆高度',default=True,description='完整应用时才移动骨盆；单只手脚更新不会移动骨盆')
    support_pelvis_limit: FloatProperty(name='骨盆调整上限',default=.25,min=0,subtype='DISTANCE')
    hand_contact_l: EnumProperty(name='左手接触',items=[('FREE','自由','保留识别手势'),('SURFACE','扶面','定位掌心并使用张开手势'),('GRIP','握物','定位掌心并使用握持手势')],default='FREE')
    hand_contact_r: EnumProperty(name='右手接触',items=[('FREE','自由','保留识别手势'),('SURFACE','扶面','定位掌心并使用张开手势'),('GRIP','握物','定位掌心并使用握持手势')],default='FREE')
    hand_target_l: PointerProperty(name='掌心目标',type=bpy.types.Object,description='物体位置为掌心中心，局部 Y 为手指方向，Z 为掌面法向；不自动识别物体形状')
    hand_target_r: PointerProperty(name='掌心目标',type=bpy.types.Object,description='物体位置为掌心中心，局部 Y 为手指方向，Z 为掌面法向；不自动识别物体形状')
    direct_undo_state: StringProperty(options={'HIDDEN'})
    direct_endpoint: EnumProperty(name='拖动部位',items=direct_tools.ENDPOINTS,default='wrist.R')
    direct_lock_height: BoolProperty(name='脚踝保持当前高度',default=False,description='固定脚踝当前高度，适合已接地的脚；关闭可上下拖动，此时可能离地')
    direct_residual: FloatProperty(name='目标残差',default=0,subtype='DISTANCE')
    reference_opacity: FloatProperty(name='原图显示强度',default=.35,min=0,max=1,update=direct_tools.update_overlay)
    profile_status: StringProperty(default='尚未保存此角色配置')
    diagnosis: StringProperty(options={'HIDDEN'})
    rest_palm_flip_l: BoolProperty(name='校准：反转左掌静置法线',default=False,description='仅修正此角色静置掌面定义；不是当前截图的手掌翻转')
    rest_palm_flip_r: BoolProperty(name='校准：反转右掌静置法线',default=False,description='仅修正此角色静置掌面定义；不是当前截图的手掌翻转')
    hand_edit_side: EnumProperty(name='编辑哪只手',items=[('L','左手','人物自身左手'),('R','右手','人物自身右手')],default='R')
    hand_palm_options: CollectionProperty(type=pose_tools.SPCandidateItem)
    hand_palm_index: IntProperty(default=0,min=0)
    hand_palm_active_index: IntProperty(default=-1)
    hand_palm_side: StringProperty()
    hand_palm_signature: StringProperty(options={'HIDDEN'})
    hand_reference_l: StringProperty(name='左手参考图（留空使用主图）',subtype='FILE_PATH')
    hand_reference_r: StringProperty(name='右手参考图（留空使用主图）',subtype='FILE_PATH')
    hand_person_l: IntProperty(name='人物序号',default=1,min=1,max=4)
    hand_person_r: IntProperty(name='人物序号',default=1,min=1,max=4)
    hand_grip_l: FloatProperty(name='握持收拢',default=.75,min=0,max=1)
    hand_grip_r: FloatProperty(name='握持收拢',default=.75,min=0,max=1)
    hand_semantics: BoolProperty(name='采用可靠的自动手势建议',default=False)
    hand_gpu_2d: BoolProperty(name='尝试 GPU 二维手部定位（可选）',default=False,description='RTMPose 手部专用模型；不产生深度，低质量结果会回退；二次元戴手套截图未必更好')
    hand3d_backend: EnumProperty(name='手部精修模型',items=[('LOCAL','现有局部模型','MediaPipe 与已有手部来源'),('WILOR','WiLoR · CUDA 三维','独立手形重建；需要 MANO'),('HAMER','HaMeR · CUDA 三维','Transformer 手形重建；需要 MANO'),('COMPARE','WiLoR / HaMeR 双模型对比','分别保存结果，不自动选择或应用'),('HAMER_TEXTURE','纹理监督增强 HaMeR','需要单独下载作者增强权重并核验；未就绪不回退冒充'),('COMPARE_TEXTURE','原版 / 增强版 HaMeR 对比','独立比较两份权重，不自动应用')],default='LOCAL')
    hand_texture_checkpoint: StringProperty(name='增强版 HaMeR 权重',subtype='FILE_PATH')
    hand3d_video_backend: EnumProperty(name='视频手部模型',items=[('LOCAL','现有手部流程','保持当前视频手部识别'),('WILOR','WiLoR · CUDA 三维','按采样帧重建手部，使用同一常驻模型'),('HAMER','HaMeR · CUDA 三维','按采样帧重建手部，使用同一常驻模型')],default='LOCAL')
    hand3d_mano: StringProperty(name='MANO_RIGHT.pkl',subtype='FILE_PATH',description='按 MANO 官方授权下载的文件；不会随插件分发')
    hand3d_precision: BoolProperty(name='截图精修使用原图与原生旋转',default=True,description='核验原始图片后保留手部像素，保存 MANO 关节旋转；仅影响截图手部精修')
    hand_native_roll: BoolProperty(name='采用原生指骨扭转',default=True,description='有 MANO 原生数据时保留指根等可转动关节的轴向旋转；自动处理中间和末端指节仍限制为弯曲')
    hand3d_comparison: StringProperty(subtype='FILE_PATH')
    hand_refine_report: StringProperty(subtype='FILE_PATH')
    multiview_enabled: BoolProperty(name='启用多图辅助',default=False)
    reference_images: CollectionProperty(type=multiview_tools.SPReferenceImage)
    multiview_strength: FloatProperty(name='辅助影响',default=.35,min=.05,max=.6)
    multiview_report: StringProperty(subtype='FILE_PATH')
    multiview_output: StringProperty(subtype='FILE_PATH')
    local_scope: EnumProperty(name='更新部位',items=pose_tools.SCOPES,default='HAND_R')
    pose_candidates: CollectionProperty(type=pose_tools.SPCandidateItem)
    candidate_index: IntProperty(default=0,min=0)
    candidate_active_index: IntProperty(default=-1)
    candidate_state: StringProperty(options={'HIDDEN'})
    candidate_signature: StringProperty(options={'HIDDEN'})
    chosen_candidate: StringProperty(options={'HIDDEN'})
    chosen_signature: StringProperty(options={'HIDDEN'})
    target: PointerProperty(name='角色骨架', type=bpy.types.Object, poll=target_poll,update=target_changed,description='从列表选择要应用姿势的角色骨架，不依赖视图中选中的对象')
    backend: EnumProperty(name='识别模型',items=[('MEDIAPIPE','快速 · MediaPipe','本地轻量模型'),('RTMW','高精度 · RTMW 大模型','133 点全身识别；深度仍由 MediaPipe 估计'),('RTMW3D','GPU 三维 · RTMW3D','使用 CUDA 推理三维身体、手掌与脚部'),('SAM3D','SAM 三维 · GPU','SAM 127 关节身体和头部；结合 RTMW 与局部手部模型')],default='SAM3D' if LOCAL_RUNTIME.get('sam_root') else 'MEDIAPIPE')
    fit_projection: BoolProperty(name='贴合原图并匹配相机',default=False,description='按选定的投影模型匹配图片；保留骨长与已接地脚踝')
    fit_model: EnumProperty(name='投影模型',items=[('ORTHO','原有近似','保留现有弱透视流程'),('PERSPECTIVE','标定透视（试验）','使用当前截图的 AnyCalib 内参，联合优化透视相机与肢体；失败时保留原拟合')],default='ORTHO')
    camera_refresh_body: BoolProperty(name='同时重新识别 SAM 身体（试验）',default=False,description='用估计内参重新运行身体模型；当前样本并未稳定胜出，默认只标定并保留现有身体和手部观测')
    fit_iterations: IntProperty(name='拟合迭代上限',default=45,min=5,max=100)
    fit_torso: BoolProperty(name='骨盆、脊柱与肩部联合修正',default=True,description='保持匹配相机，受限调整躯干；实际骨架未改善或破坏支撑时撤回；视频单独保持原流程')
    whole_body: BoolProperty(name='全身与支撑协调',default=True,description='在原结果上一起调整骨盆XYZ、躯干和四肢；固定明确的手脚接触，验证实际骨架与关节范围后采用；视频不自动启用')
    whole_body_root_limit: FloatProperty(name='骨盆各轴位移上限（比例）',default=.10,min=0,max=.25,description='相对当前身体包围尺度；显式支撑时同时受骨盆调整上限限制')
    whole_body_rotation_limit: FloatProperty(name='躯干修正角度上限',default=12.,min=0,max=25.,description='每个可写躯干控制的各轴角度上限，单位度；0保留原躯干旋转')
    fit_regularization: FloatProperty(name='保留初始姿态',default=.025,min=.005,max=.2)
    fit_camera: StringProperty()
    camera_backup: StringProperty()
    orient_head: BoolProperty(name='使用局部头部朝向',default=False)
    orient_hand_l: BoolProperty(name='使用左掌完整朝向',default=False)
    orient_hand_r: BoolProperty(name='使用右掌完整朝向',default=False)
    auto_finger_curl: BoolProperty(name='自动手指姿势',default=True)
    finger_joints: BoolProperty(name='逐节手指与张指',default=True,description='有三维手点时分别驱动每节指骨；显式手势预设优先')
    full_source_rotation: BoolProperty(name='使用完整源骨架旋转',default=True,description='数据包含完整骨架时保留肩部、脊柱链和关节扭转；关闭可比较旧关节点方案')
    auto_full_quality: BoolProperty(name='比较图像贴合与关节旋转',default=True,description='同一起点比较二维贴合和源姿势旋转保留；没有明显图像证据时优先完整骨架。未开启拟合也保留完整旋转')
    orientation_threshold: FloatProperty(name='局部质量下限',default=.45,min=0,max=1)
    head_strength: FloatProperty(name='头部朝向强度',default=1,min=0,max=1)
    head_consensus: BoolProperty(name='比较身体与局部头向',default=True,description='以独立脸部观测检查候选和头颈角度，证据不足时保留身体头向；仍可手动微调')
    head_gpu: BoolProperty(name='加入独立头向 GPU 候选',default=False,description='可选 6DRepNet360；需重新识别头部，不保证优于 SAM，仍经脸点复核后采用')
    hand_strength: FloatProperty(name='手掌朝向强度',default=1,min=0,max=1)
    head_yaw: FloatProperty(name='头部左右微调',default=0,subtype='ANGLE',min=-math.pi,max=math.pi)
    head_pitch: FloatProperty(name='头部抬低微调',default=0,subtype='ANGLE',min=-math.pi,max=math.pi)
    head_roll: FloatProperty(name='头部歪斜微调',default=0,subtype='ANGLE',min=-math.pi,max=math.pi)
    neck_share: FloatProperty(name='颈部分担',default=.25,min=0,max=.6)
    forearm_share: FloatProperty(name='前臂分担扭转',default=.7,min=0,max=1)
    wrist_limit: FloatProperty(name='手腕弯曲上限',default=math.radians(85),min=math.radians(20),max=math.radians(150),subtype='ANGLE')
    palm_roll_l: FloatProperty(name='左掌旋转微调',default=0,min=-math.pi,max=math.pi,subtype='ANGLE')
    palm_roll_r: FloatProperty(name='右掌旋转微调',default=0,min=-math.pi,max=math.pi,subtype='ANGLE')
    palm_flip_l: BoolProperty(name='翻转左掌',default=False)
    palm_flip_r: BoolProperty(name='翻转右掌',default=False)
    hand_candidate_l: EnumProperty(name='左掌几何候选',items=[('0','自动选择',''),('1','候选 1',''),('2','候选 2','')],default='0')
    hand_candidate_r: EnumProperty(name='右掌几何候选',items=[('0','自动选择',''),('1','候选 1',''),('2','候选 2','')],default='0')
    head_forward: EnumProperty(name='角色静置面朝向',items=[('-Y','-Y（默认正面）',''),('Y','+Y',''),('X','+X',''),('-X','-X','')],default='-Y')
    index_l: StringProperty(name='左食指根骨（自动）')
    pinky_l: StringProperty(name='左小指根骨（自动）')
    index_r: StringProperty(name='右食指根骨（自动）')
    pinky_r: StringProperty(name='右小指根骨（自动）')
    show_orientation: BoolProperty(name='头部 / 手掌朝向',default=True)
    show_orientation_calibration: BoolProperty(name='角色方向校准与高级选项',default=False)
    capture_delay: IntProperty(name='截图前等待（旧设置，已停用）',default=0,min=0,max=10,description='点击后留出切换到参考图片的时间')
    image_path: StringProperty(name='截图', subtype='FILE_PATH')
    json_path: StringProperty(name='姿势数据', subtype='FILE_PATH')
    overlay_path: StringProperty(subtype='FILE_PATH')
    preview_name: StringProperty()
    mapping: CollectionProperty(type=SPMapping)
    coordinate_mode: EnumProperty(name='重建方式',items=[('WORLD','估计三维','优先使用模型推测的三维关节'),('HYBRID','图片优先 + 估计深度','优先匹配截图轮廓，深度使用可调整的三维估计'),('IMAGE','按图片校正','关节点按截图投影排布，适合二次元与手动修正')],default='WORLD')
    depth: FloatProperty(name='深度强度',default=1,min=0,max=2)
    yaw: FloatProperty(name='朝向修正',subtype='ANGLE',default=0,min=-math.pi,max=math.pi)
    mirror: BoolProperty(name='左右镜像',default=False)
    threshold: FloatProperty(name='可见度阈值',default=.35,min=0,max=1)
    confidence: FloatProperty(name='检测阈值',default=.5,min=.1,max=.95)
    person_index: IntProperty(name='人物序号（从 1 开始）',default=1,min=1,max=4)
    strength: FloatProperty(name='应用强度',default=1,min=0,max=1)
    keyframe: BoolProperty(name='在当前帧插入关键帧',default=False,description='复制现有 Action 后写入，保留原动作')
    switch_fk: BoolProperty(name='自动切换已识别绑定到 FK',default=True,description='支持 Rigify / Auto-Rig Pro；恢复时还原开关')
    show_names: BoolProperty(name='显示关节名称',default=True)
    show_mapping: BoolProperty(name='骨骼映射',default=False)
    show_advanced: BoolProperty(name='识别与深度选项',default=False)
    status: StringProperty(default='选择角色和截图开始；也可直接使用手动关节点')
    last_log: StringProperty()
    last_result: StringProperty()
    quality_warnings: StringProperty()
    auto_depth_select: BoolProperty(name='自动比较三维深度候选',default=True,description='在共同相机下结合图像、关节结构、源深度与明确接触目标评估；优势不足保留默认候选，有歧义需复核')
    foot_pose: BoolProperty(name='三维脚向与鞋底参考',default=False,description='读取目标鞋底参考；保持脚跟、前脚掌与地面的关系')
    shoe_mesh: PointerProperty(name='鞋子网格（留空自动）',type=bpy.types.Object,poll=lambda self,obj:obj.type=='MESH')
    foot_threshold: FloatProperty(name='脚部质量下限',default=.4,min=0,max=1)
    ground_feet: BoolProperty(name='双脚接地（站立 / 深蹲）',default=False,description='移动骨盆并做两段腿部 IK 求解；坐姿/跳跃请关闭')
    coordinate_pelvis: BoolProperty(name='协调骨盆高度与双腿',default=False,description='双脚接地并贴合原图时，比较小范围骨盆高度调整；保留接地点和骨长。真实骨架未改善时保留原方案；不处理碰撞或物理平衡')
    root_bone: StringProperty(name='骨盆平移控制骨',description='留空自动选择 ARP root / Rigify torso / 映射的骨盆')
    floor_height: FloatProperty(name='地面高度',default=0,subtype='DISTANCE')
    pelvis_offset: FloatProperty(name='骨盆高度微调',default=0,min=-10,max=10,subtype='DISTANCE')
    ankle_clearance: FloatProperty(name='踝部余量（腿长比例）',default=.055,min=0,max=.3)
    flatten_feet: BoolProperty(name='脚部方向放平',default=True)
    hand_left: EnumProperty(name='左手预设',items=[('NONE','自动 / 保持','优先使用可靠的自动手指结果；缺失时报告并保持原手指'),('V','V 手势约束（ARP）','食指、中指伸直，其余手指收拢'),('GRIP','握持约束（ARP）','近似握持手机；不模拟物体接触'),('FIST','握拳（ARP）',''),('OPEN','张开（ARP）','')],default='NONE')
    hand_right: EnumProperty(name='右手预设',items=[('NONE','自动 / 保持','优先使用可靠的自动手指结果；缺失时报告并保持原手指'),('V','V 手势约束（ARP）','食指、中指伸直，其余手指收拢'),('GRIP','握持约束（ARP）','近似握持手机；不模拟物体接触'),('FIST','握拳（ARP）',''),('OPEN','张开（ARP）','')],default='NONE')
    hand_spread: FloatProperty(name='V 手指张开角',default=.16,min=0,max=.6,subtype='ANGLE')


class SP_OT_setup(Operator):
    bl_idname = 'screenshot_pose.setup'
    bl_label = '准备本地识别环境（首次联网）'
    bl_description = '只下载安装依赖和模型；不会上传截图'

    def execute(self, context):
        try:
            def done(scene):
                prefs = preferences()
                if prefs:
                    prefs.worker_python = ''
                scene.screenshot_pose.status = '本地识别环境已准备好'
            start_job(context.scene,[setup_python(), HERE/'setup_runtime.py','--runtime', runtime_dir()], '准备识别环境',done)
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_setup_gpu(Operator):
    bl_idname='screenshot_pose.setup_gpu'
    bl_label='准备 GPU 三维模型与环境（首次联网）'
    def execute(self,context):
        try:
            def done(scene):
                if preferences():preferences().gpu_python=str(runtime_dir()/'gpu_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
                scene.screenshot_pose.status='GPU 三维环境已准备好，并已验证 CUDA 执行'
            start_job(context.scene,[setup_python(),HERE/'setup_gpu.py','--runtime',runtime_dir()], '准备 CUDA 三维环境',done)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_setup_high(Operator):
    bl_idname='screenshot_pose.setup_high'
    bl_label='准备高精度模型（首次联网）'
    def execute(self,context):
        try:
            if not runtime_python().is_file():raise ValueError('请先准备基础识别环境')
            def done(scene):scene.screenshot_pose.status='RTMW 高精度模型已准备好'
            start_job(context.scene,[runtime_python(),HERE/'setup_high.py','--runtime',runtime_dir()], '准备 RTMW 大模型',done)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_match_camera(Operator):
    bl_idname='screenshot_pose.match_camera'
    bl_label='创建并切换到匹配相机'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:projection_fit.create_camera(context,context.scene.screenshot_pose)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_restore_camera(Operator):
    bl_idname='screenshot_pose.restore_camera'
    bl_label='恢复原相机与画幅'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        projection_fit.restore_camera(context,context.scene.screenshot_pose)
        return {'FINISHED'}


class SP_OT_setup_details(Operator):
    bl_idname='screenshot_pose.setup_details'
    bl_label='准备头部 / 手部模型（首次联网）'
    def execute(self,context):
        try:
            if not runtime_python().is_file():raise ValueError('请先准备基础识别环境')
            def done(scene):scene.screenshot_pose.status='头部与手部局部模型已准备好'
            start_job(context.scene,[runtime_python(),HERE/'setup_details.py','--runtime',runtime_dir()], '准备局部朝向模型',done)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_details(Operator):
    bl_idname='screenshot_pose.details'
    bl_label='分析头部 / 手掌朝向'
    head_only: BoolProperty(default=False)
    def execute(self,context):
        try:
            s=context.scene.screenshot_pose;source=Path(bpy.path.abspath(s.json_path))
            if not source.is_file():raise ValueError('请先识别身体或导入姿势 JSON')
            names=('face_landmarker.task',) if self.head_only else ('face_landmarker.task','hand_landmarker.task')
            if not all((runtime_dir()/n).is_file() for n in names):raise ValueError('请先准备头部 / 手部模型')
            import hashlib
            fingerprint=hashlib.sha256(source.read_bytes()).hexdigest();head_only=self.head_only
            output=runtime_dir()/'poses'/('details_'+uuid.uuid4().hex+'.json');output.parent.mkdir(parents=True,exist_ok=True)
            def done(scene):
                state=scene.screenshot_pose
                if Path(bpy.path.abspath(state.json_path)).resolve()!=source.resolve() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=fingerprint:
                    state.status='参考数据已变化，本次头部结果未自动采用';return
                state.json_path=str(output);state.orient_head=True
                if not head_only:state.orient_hand_l=True;state.orient_hand_r=True
                d=pose_data.read_pose(output);o=d['orientation'];state.status=f"局部识别完成：头部 {'成功' if o.get('head') else '失败'}，{len(o['hands'])} 只手；应用后查看朝向"
                if head_only:state.status='头部候选已更新，身体和手部观测保留；点击仅更新头部或应用到角色'
                if d.get('head3d_runtime',{}).get('actual_inference') is False:state.status+='；独立 GPU 候选失败，已保留其他候选，详见任务日志'
                if bpy.context.scene==scene:preview.make_preview(bpy.context,state,d)
            extra=['--head-only'] if head_only else []
            if head_only and s.head_gpu:
                cfg=runtime_dir()/'head3d/config.json'
                if not cfg.is_file():raise ValueError('请先准备独立头向 GPU 模型，或关闭 GPU 候选')
                extra+=['--head3d-config',cfg]
            start_job(context.scene,[runtime_python(),HERE/'local_pose_worker.py','--input',source,'--output',output,'--model-dir',runtime_dir()]+extra, '分析头部候选' if head_only else '分析头部与手掌',done)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_calibrate_orientation(Operator):
    bl_idname='screenshot_pose.calibrate_orientation'
    bl_label='校准角色方向（读取静置骨架）'
    def execute(self,context):
        try:
            s=context.scene.screenshot_pose
            if not s.target:raise ValueError('请先选择角色骨架')
            mapping={r.role:r.bone for r in s.mapping if r.enabled and r.bone}
            character_tools.guard(s)
            result=orientation.store_calibration(s.target,mapping,s)
            character_tools.save(s)
            self.report({'INFO'},f"已读取头部参考和 {len(result['hands'])} 只手掌参考")
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_cancel(Operator):
    bl_idname = 'screenshot_pose.cancel'
    bl_label = '取消当前任务'

    def execute(self,context):
        if _JOB:
            cancelled_job=_JOB
            stop_job()
            context.scene.screenshot_pose.status = '任务已取消'
            folder=cancelled_job.get('record_folder')
            if folder and (folder/'result.json').is_file():
                try:
                    saved=json.loads((folder/'result.json').read_text(encoding='utf8'))
                    if not saved.get('cancelled'):
                        video_tools.recording_done(cancelled_job['scene'],folder)
                        cancelled_job['scene'].screenshot_pose.status='后续识别已取消，录制视频已保留，可重新识别'
                except (OSError,ValueError,KeyError):pass
        return {'FINISHED'}


class SP_OT_target(Operator):
    bl_idname = 'screenshot_pose.use_selected'
    bl_label = '使用当前选中的角色'
    bl_options = {'REGISTER','UNDO'}

    def execute(self,context):
        obj = context.object
        target = obj if obj and obj.type == 'ARMATURE' else obj.find_armature() if obj else None
        if not target or target.get('sp_source'):
            self.report({'ERROR'},'请先选中角色身体或它的控制骨架'); return {'CANCELLED'}
        settings = context.scene.screenshot_pose
        settings.target = target
        count = set_mapping(settings)
        settings.status = f'目标：{target.name}；自动匹配 {count} 个部位'
        return {'FINISHED'}


class SP_OT_map(Operator):
    bl_idname = 'screenshot_pose.auto_map'
    bl_label = '自动匹配骨骼'
    bl_options = {'REGISTER','UNDO'}

    def execute(self,context):
        settings=context.scene.screenshot_pose
        if not settings.target:
            self.report({'ERROR'},'请先设置目标角色'); return {'CANCELLED'}
        settings.status = f'已匹配 {set_mapping(settings)} 个部位；请核对左右和控制骨'
        return {'FINISHED'}


class SP_OT_detect(Operator):
    bl_idname = 'screenshot_pose.detect'
    bl_label = '识别截图 → 生成关节点'

    def execute(self,context):
        settings = context.scene.screenshot_pose
        try:
            source = Path(bpy.path.abspath(settings.image_path))
            if not source.is_file():
                raise ValueError('请先选择有效的截图文件')
            model = runtime_dir()/'pose_landmarker_full.task'
            if settings.backend in ('RTMW3D','SAM3D') and (not runtime_python().is_file() or not (runtime_dir()/'rtmw3d_pose.onnx').is_file()):raise ValueError('请先点击“准备 GPU 三维模型与环境”')
            if not runtime_python().is_file() or not model.is_file():
                raise ValueError('请先点击“准备本地识别环境”')
            out = runtime_dir()/'poses'/uuid.uuid4().hex
            out.mkdir(parents=True,exist_ok=True)
            result, overlay = out/'pose.json',out/'overlay.png'
            if settings.json_path and Path(bpy.path.abspath(settings.json_path)).is_file():
                try:
                    if pose_data.read_pose(bpy.path.abspath(settings.json_path)).get('source') == 'manual_template':
                        settings.coordinate_mode='WORLD'; settings.depth=1
                except (ValueError, OSError):
                    pass
            def done(scene):
                s = scene.screenshot_pose
                frame_tools.accept(s)
                s.json_path, s.overlay_path = str(result), str(overlay)
                s.orient_head=False;s.orient_hand_l=False;s.orient_hand_r=False;s.foot_pose=False
                data = pose_data.read_pose(result)
                if data.get('source') in ('rtmw3d_wholebody','sam3d_mhr'):s.foot_pose=True;s.orient_head=True;s.orient_hand_l=True;s.orient_hand_r=True
                if data.get('suggested_mode')=='HYBRID':s.coordinate_mode='HYBRID';s.depth=1;s.mirror=False;s.yaw=0
                # A file switch while inference is running must not modify a different scene.
                if bpy.context.scene != scene:
                    s.status = '识别完成，请切回此场景并点击重建关节点'
                    hand_tools.screenshot_status(s,data,result)
                    return
                low = preview.make_preview(bpy.context,s,data)
                s.status = f"识别完成：{data['pose_count']} 人，已选择第 {data['person_index']+1} 人；{len(low)} 个低可见度关节"
                hand_tools.screenshot_status(s,data,result)
                pose_library.detected(scene)
            start_job(context.scene,[runtime_python(),HERE/'worker_screenshot.py','--backend',settings.backend,'--hand-mode',settings.screenshot_hands,'--progress',out/'progress.json','--image',source,'--model',model,
                      '--output',result,'--overlay',overlay,'--person-index',str(settings.person_index-1),
                      '--max-poses','4','--confidence',str(settings.confidence)]+(sam_root_args() if settings.backend=='SAM3D' else []),'识别截图',done)
            if _JOB:_JOB['progress']=out/'progress.json'
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_clipboard(Operator):
    # Keep the operator ID for existing shortcuts; the UI now starts region capture.
    bl_idname='screenshot_pose.clipboard'
    bl_label='截图并识别'
    bl_description='立即框选参考图片，保持屏幕原亮度；松手自动识别，Esc 或右键取消'

    def execute(self,context):
        s=context.scene.screenshot_pose
        try:
            if os.name!='nt':raise ValueError('直接截图目前支持 Windows，请使用选择图片文件')
            model=runtime_dir()/'pose_landmarker_full.task'
            if not runtime_python().is_file() or not model.is_file():raise ValueError('请先准备本地识别环境')
            if s.backend in ('RTMW3D','SAM3D') and not (runtime_dir()/'rtmw3d_pose.onnx').is_file():raise ValueError('请先准备 GPU 三维模型与环境')
            if s.backend=='RTMW' and not all((runtime_dir()/n).is_file() for n in ('rtmw_detector.onnx','rtmw_pose.onnx')):
                raise ValueError('请先准备高精度模型，或选择快速模式')
            out=runtime_dir()/'captures'/uuid.uuid4().hex;out.mkdir(parents=True,exist_ok=True)
            image,result,overlay,status=out/'capture.png',out/'pose.json',out/'overlay.png',out/'capture_status.json'
            def done(scene):
                state=scene.screenshot_pose
                if json.loads(status.read_text(encoding='utf-8')).get('cancelled'):
                    state.status='已取消截图，保留当前姿势数据';return
                frame_tools.accept(state)
                state.image_path=str(image);state.json_path=str(result);state.overlay_path=str(overlay)
                state.orient_head=False;state.orient_hand_l=False;state.orient_hand_r=False;state.foot_pose=False
                data=pose_data.read_pose(result)
                if data.get('source') in ('rtmw3d_wholebody','sam3d_mhr'):state.foot_pose=True;state.orient_head=True;state.orient_hand_l=True;state.orient_hand_r=True
                if data.get('suggested_mode')=='HYBRID':state.coordinate_mode='HYBRID';state.depth=1;state.mirror=False;state.yaw=0
                elif state.coordinate_mode=='IMAGE':state.coordinate_mode='WORLD';state.depth=1
                if bpy.context.scene!=scene:
                    state.status='截图识别完成，请切回此场景并重建关节点';hand_tools.screenshot_status(state,data,result);return
                low=preview.make_preview(bpy.context,state,data)
                state.status=f"截图识别完成：{data['pose_count']} 人；{len(low)} 个低可见度关节"
                hand_tools.screenshot_status(state,data,result)
                pose_library.detected(scene)
            start_job(context.scene,[runtime_python(),HERE/'capture_worker.py','--hand-mode',s.screenshot_hands,'--progress',out/'progress.json','--image-output',image,'--output',result,'--overlay',overlay,'--status',status,'--model',model,'--backend',s.backend,'--delay','0','--person-index',str(s.person_index-1),'--confidence',str(s.confidence)]+(sam_root_args() if s.backend=='SAM3D' else []), '框选截图并识别',done)
            if _JOB:_JOB['progress']=out/'progress.json'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_template(Operator):
    bl_idname='screenshot_pose.template'
    bl_label='手动关节点（无需识别）'
    bl_description='为二次元或检测失败的截图创建可拖动模板，不代表识别结果'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        s=context.scene.screenshot_pose
        try:
            path=bpy.path.abspath(s.image_path) if s.image_path else ''
            width,height=1000,1000
            if path:
                img=bpy.data.images.load(path,check_existing=True)
                width,height=img.size[:]
            data=pose_data.template_pose(path,width,height)
            out=runtime_dir()/'poses'/uuid.uuid4().hex
            out.mkdir(parents=True,exist_ok=True)
            dest=out/'manual.json'
            dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            s.json_path=str(dest)
            s.coordinate_mode='IMAGE'; s.depth=0; s.mirror=False; s.yaw=0
            preview.make_preview(context,s,data)
            s.status='手动模板已创建：正视图对齐截图，拖动关节点后应用'
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_rebuild(Operator):
    bl_idname='screenshot_pose.rebuild'
    bl_label='按参数重建关节点（替换手动调整）'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        s=context.scene.screenshot_pose
        try:
            low=preview.make_preview(context,s,pose_data.read_pose(bpy.path.abspath(s.json_path)))
            s.status=f'关节点已重建；{len(low)} 个低可见度关节用方框标记'
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_refresh(Operator):
    bl_idname='screenshot_pose.refresh'
    bl_label='刷新骨架连线'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        preview.refresh_lines(context,context.scene.screenshot_pose)
        return {'FINISHED'}


class SP_OT_focus(Operator):
    bl_idname='screenshot_pose.focus'
    bl_label='查看并编辑关节点'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.screenshot_pose
        points=preview.marker_objects(s)
        if not points or context.area.type != 'VIEW_3D':
            self.report({'ERROR'},'请先生成关节点，并在 3D 视图操作'); return {'CANCELLED'}
        if context.object and context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in points:
            obj.select_set(True)
        context.view_layer.objects.active=points[0]
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected(use_all_regions=False)
        s.status='选单个关节按 G 移动；正视图调轮廓，G Y 调前后深度；完成后应用'
        return {'FINISHED'}


class SP_OT_apply(Operator):
    bl_idname='screenshot_pose.apply'
    bl_label='应用到当前角色'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            result=apply_current(context,context.scene.screenshot_pose)
            if not result.get('applied'):
                self.report({'WARNING'},'没有可应用的部位，请查看映射和结果提示')
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_restore(Operator):
    bl_idname='screenshot_pose.restore'
    bl_label='恢复应用前的角色姿势'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        if context.scene.screenshot_pose.candidate_state:pose_tools.cancel(context.scene.screenshot_pose,True)
        target=context.scene.screenshot_pose.target
        try:
            if not target or not target.get('screenshot_pose_backup'):
                raise ValueError('此角色没有恢复快照')
            animation_preview.end(target)
            retarget.restore_pose(target,json.loads(target['screenshot_pose_backup']))
            del target['screenshot_pose_backup']
            context.scene.screenshot_pose.status='已恢复应用前的姿势、FK 状态和动作'
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_import(Operator,ImportHelper):
    bl_idname='screenshot_pose.import_json'
    bl_label='导入姿势 JSON'
    filename_ext='.json'
    filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            data=pose_data.read_pose(self.filepath)
            s=context.scene.screenshot_pose
            frame_tools.accept(s)
            s.json_path=self.filepath
            if data.get('orientation'):s.orient_head=True;s.orient_hand_l=True;s.orient_hand_r=True
            if data.get('source')=='rtmw3d_wholebody':s.foot_pose=True
            if data.get('suggested_mode') == 'HYBRID':
                s.coordinate_mode='HYBRID';s.mirror=False;s.yaw=0
            preview.make_preview(context,s,data)
            s.status='姿势数据已导入'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_export(Operator,ExportHelper):
    bl_idname='screenshot_pose.export_json'
    bl_label='导出识别 / 图片校正 JSON'
    filename_ext='.json'
    filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
    def execute(self,context):
        try:
            data=pose_data.read_pose(bpy.path.abspath(context.scene.screenshot_pose.json_path))
            Path(self.filepath).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            self.report({'INFO'},'已导出识别及图片校正数据；3D 视图内移动的关节点保存在 .blend 工程中')
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_result(Operator):
    bl_idname='screenshot_pose.show_result'
    bl_label='查看应用结果'
    def execute(self,context):
        s=context.scene.screenshot_pose
        def draw(menu,_context):
            menu.layout.label(text=s.status[:100])
            if s.quality_warnings:
                for note in json.loads(s.quality_warnings):
                    menu.layout.label(text=note[:130],icon='ERROR')
            if s.last_result:
                result=json.loads(s.last_result)
                selection=result.get('depth_selection')
                if selection:menu.layout.label(text='身体深度：'+('RTMW3D' if selection['chosen']=='rtmw3d_relative_depth' else '局部质量比较后保留原深度'))
                if result.get('feet'):menu.layout.label(text='脚部：已使用鞋底参考')
                comparison=result.get('full_source_comparison')
                if comparison:
                    menu.layout.label(text='身体：'+('完整源骨架' if comparison['chosen']=='full_source_rotation' else '关节点拟合'))
                    for label,score in comparison.get('scores',{}).items():
                        menu.layout.label(text=('完整骨架' if label=='full_source_rotation' else '关节点')+f"：投影 {score['projection_px']:.1f}px，源旋转偏差 {score['rotation_deg']:.1f}°")
                detail=result.get('orientation',{})
                if detail.get('head'):menu.layout.label(text='头部：已使用局部完整朝向')
                for side,hand in detail.get('hands',{}).items():
                    source={'wilor_hand3d':'WiLoR 三维手部','hamer_hand3d':'HaMeR 三维手部','hamer_texture_hand3d':'纹理监督增强 HaMeR','mediapipe_hand_world':'手部局部模型','rtmw3d_hand':'GPU 三维手部'}.get(hand['source'],'掌面几何估计')
                    menu.layout.label(text=('左掌' if side=='L' else '右掌')+'：'+source)
                for side,hand in result.get('hand_transfer',{}).items():
                    if hand.get('mean_direction_error_deg') is not None:
                        menu.layout.label(text=('左手' if side=='L' else '右手')+f"源→角色指节方向偏差：平均 {hand['mean_direction_error_deg']:.1f}° / 最大 {hand['max_direction_error_deg']:.1f}°")
                if result.get('hand_transfer'):menu.layout.label(text='上述偏差衡量转移保真，不代表原图识别准确率。',icon='INFO')
                for note in result.get('warnings',[]):
                    menu.layout.label(text=str(note)[:130],icon='INFO')
                for note in result.get('skipped',[]):
                    menu.layout.label(text='跳过：'+str(note)[:100])
        context.window_manager.popup_menu(draw,title='Screenshot Pose 结果')
        return {'FINISHED'}


class SP_OT_transfer_diagnosis(Operator):
    bl_idname='screenshot_pose.transfer_diagnosis'
    bl_label='查看源姿势与角色诊断'
    bl_description='离线对比识别源骨架、源手和当前角色，不改变角色'
    def execute(self,context):
        try:
            s=context.scene.screenshot_pose
            if _JOB or s.candidate_state:raise ValueError('请先结束识别任务或候选预览')
            if not s.target or not s.last_result:raise ValueError('请先应用当前截图姿势')
            result=json.loads(s.last_result)
            if result.get('observation_signature')!=pose_tools.signature(s):raise ValueError('参考数据已变化，请先应用当前姿势或局部修正')
            report=screenshot_diagnostics.export(context,s,result)
            bpy.ops.wm.path_open(filepath=str(report))
            s.status='已导出源姿势与当前角色诊断；不改变角色或动画'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_log(Operator):
    bl_idname='screenshot_pose.open_log'
    bl_label='打开任务日志'
    def execute(self,context):
        path=context.scene.screenshot_pose.last_log
        if path and Path(path).is_file():
            bpy.ops.wm.path_open(filepath=path)
        return {'FINISHED'}


class SP_OT_overlay(Operator):
    bl_idname='screenshot_pose.open_overlay'
    bl_label='查看图片上的识别结果'
    def execute(self,context):
        path=context.scene.screenshot_pose.overlay_path
        if path and Path(path).is_file():
            bpy.ops.wm.path_open(filepath=path)
            return {'FINISHED'}
        self.report({'ERROR'},'尚无识别预览，请先识别截图')
        return {'CANCELLED'}


class SP_OT_image_editor(Operator):
    bl_idname='screenshot_pose.edit_image'
    bl_label='在原图上校正关节点'
    bl_description='打开本地图像编辑窗；选择左右手腕并拖动或点击放置，保存后回到 Blender'
    def execute(self,context):
        s=context.scene.screenshot_pose
        try:
            source=Path(bpy.path.abspath(s.json_path))
            pose_data.read_pose(source)
            output=runtime_dir()/'poses'/uuid.uuid4().hex/'corrected.json'
            output.parent.mkdir(parents=True,exist_ok=True)
            def done(scene):
                state=scene.screenshot_pose
                if not output.is_file():
                    state.status='已关闭图片校正，未修改姿势';return
                data=pose_data.read_pose(output)
                state.json_path=str(output);state.overlay_path=str(output.with_suffix('.png'))
                state.coordinate_mode='HYBRID';state.mirror=False;state.yaw=0
                if bpy.context.scene==scene:
                    preview.make_preview(bpy.context,state,data)
                state.status='图片校正已保存，已切换图片优先模式；可继续调整深度并应用'
            start_job(context.scene,[runtime_python(),HERE/'image_editor.py','--input',source,'--output',output],
                      '图片校正窗口已打开，保存或关闭后返回',done)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_source(Operator):
    bl_idname='screenshot_pose.create_source'
    bl_label='生成源骨架（可检查完整关节）'
    bl_description='有完整源数据时保留全部关节和旋转；否则创建兼容 T 姿势源骨架'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            obj,result=source_rig.create(context,context.scene.screenshot_pose)
            context.scene.screenshot_pose.status=f'已创建 {obj.name}；可导出 FBX/BVH 或继续重定向'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_reset_orientation(Operator):
    bl_idname='screenshot_pose.reset_orientation'
    bl_label='重置头部与手掌方向微调'
    bl_description='清零头部角度、掌心翻转和手腕角度；保留识别结果和手势预设，重新应用后生效'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.screenshot_pose
        if s.candidate_state:
            self.report({'ERROR'},'请先确认或取消候选预览');return {'CANCELLED'}
        for name in ('head_yaw','head_pitch','head_roll','palm_roll_l','palm_roll_r'):setattr(s,name,0.)
        s.palm_flip_l=False;s.palm_flip_r=False;s.hand_candidate_l='0';s.hand_candidate_r='0'
        s.status='方向微调已重置；点击应用到角色后生效'
        return {'FINISHED'}


from . import ui
# Refresh the panel definitions as well when Blender reloads the parent package.
import importlib as _importlib
_importlib.reload(ui)
SP_PT_main=ui.SP_PT_main


CLASSES=video_review.CLASSES+(pose_library.SPLibraryItem,pose_tools.SPCandidateItem,multiview_tools.SPReferenceImage,SPPreferences,SPMapping,SPSettings,SP_OT_setup,SP_OT_setup_gpu,SP_OT_setup_high,SP_OT_setup_details,SP_OT_details,SP_OT_calibrate_orientation,SP_OT_match_camera,SP_OT_restore_camera,SP_OT_cancel,SP_OT_target,SP_OT_map,
         SP_OT_detect,SP_OT_clipboard,SP_OT_template,SP_OT_rebuild,SP_OT_refresh,SP_OT_focus,
         SP_OT_apply,SP_OT_restore,SP_OT_import,SP_OT_export,SP_OT_result,SP_OT_log,SP_OT_overlay,SP_OT_image_editor,SP_OT_source,SP_OT_reset_orientation)+pose_tools.OPERATORS+multiview_tools.OPERATORS+hand_tools.OPERATORS+hand_preview.OPERATORS+character_tools.OPERATORS+direct_tools.OPERATORS+contact_targets.OPERATORS+foot_tools.OPERATORS+pose_library.OPERATORS+video_tools.OPERATORS+rgb_contact.OPERATORS+review_tools.OPERATORS+ui.PANELS


CLASSES=CLASSES+(SP_OT_transfer_diagnosis,)+camera_tools.OPERATORS


def register():
    # Blender may reload only this package when installing/enabling an update.
    # Relative imports then reuse old objects in sys.modules (e.g. the previous
    # ground_feet signature), so refresh the pure helper modules together before
    # registering any operator. Keep sole anchors and their contact behavior.
    import importlib
    for name in ('pose_data','fit_math','orientation_math','hand_edit_data','retarget','preview',
                 'contact_solver','hand_presets','projection_fit','orientation',
                 'foot_pose','full_rig','finger_pose','source_rig','contact_math','foot_source',
                 'screenshot_quality','screenshot_diagnostics','review_data','depth_selection','hand_selection','source_fit_math','source_fit','camera_data','perspective_math',
                 'perspective_fit','head_math','torso_math','torso_fit','whole_body_math','whole_body_fit','hand3d','hand_geometry','hand_precision','diagnostic_report'):
        importlib.reload(importlib.import_module('.'+name,PACKAGE))
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.screenshot_pose=PointerProperty(type=SPSettings)
    if animation_preview.before_frame not in bpy.app.handlers.frame_change_pre:bpy.app.handlers.frame_change_pre.append(animation_preview.before_frame)
    if video_review.frame_change not in bpy.app.handlers.frame_change_pre:bpy.app.handlers.frame_change_pre.append(video_review.frame_change)
    if video_review.after_frame not in bpy.app.handlers.frame_change_post:bpy.app.handlers.frame_change_post.append(video_review.after_frame)
    if video_review.save_pre not in bpy.app.handlers.save_pre:bpy.app.handlers.save_pre.append(video_review.save_pre)
    if on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(on_load_pre)


def unregister():
    review_tools.clear()
    video_review.clean()
    for handlers,callback in ((bpy.app.handlers.frame_change_pre,video_review.frame_change),(bpy.app.handlers.frame_change_post,video_review.after_frame),(bpy.app.handlers.save_pre,video_review.save_pre)):
        if callback in handlers:handlers.remove(callback)
    pose_library.clear_icons()
    animation_preview.cleanup()
    if animation_preview.before_frame in bpy.app.handlers.frame_change_pre:bpy.app.handlers.frame_change_pre.remove(animation_preview.before_frame)
    foot_tools.clean()
    direct_tools.stop_modals()
    pose_tools.cancel_all()
    direct_tools.cleanup_overlays()
    stop_job()
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)
    if on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(on_load_pre)
    del bpy.types.Scene.screenshot_pose
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__=='__main__':
    register()
