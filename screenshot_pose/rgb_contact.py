"""Optional video Action postprocess with rollback and FK validation."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

import bpy
from bpy.types import Operator
from mathutils import Matrix, Vector
import numpy as np

from . import animation_preview, contact_solver, pose_tools, retarget


JOINTS = (0, 1, 2, 4, 5, 7, 8)
ROLES = ('hips', 'thigh.L', 'thigh.R', 'shin.L', 'shin.R', 'foot.L', 'foot.R')
SOURCE = 'sp_rgb_contact_source'


def addon():
    return importlib.import_module(__package__)


def runtime():
    a = addon()
    prefs = a.preferences()
    try:
        defaults = json.loads((a.runtime_dir() / 'rgb_contact_runtime.json').read_text(encoding='utf8'))
    except (OSError, ValueError):
        defaults = {}
    root = (prefs.rgb_contact_root if prefs else '') or defaults.get('root', '')
    python = (prefs.rgb_contact_python if prefs else '') or defaults.get('python', '')
    if os.name != 'nt' or not root or not python:
        raise ValueError('请在插件偏好设置配置 Windows RGB MoCap 目录和 Python')
    root, python = (Path(bpy.path.abspath(path)).resolve() for path in (root, python))
    if not (root / 'mocap/correction.py').is_file() or not python.is_file():
        raise ValueError('RGB MoCap 或其 Python 路径无效，请检查插件偏好设置')
    return root, python


def signature(action):
    digest = hashlib.sha256()
    for curve in animation_preview.curves(action):
        digest.update(repr((curve.data_path, curve.array_index, curve.mute, len(curve.modifiers))).encode())
        keys = curve.keyframe_points
        values = np.empty(len(keys) * 2, dtype=np.float32)
        for field in ('co', 'handle_left', 'handle_right'):
            keys.foreach_get(field, values)
            digest.update(values.tobytes())
        digest.update(''.join(key.interpolation[0] for key in keys).encode())
    return digest.hexdigest()


def toe_rest(rig, foot, side):
    names = (f'c_toes_fk.{side.lower()}', f'toe.{side}', f'toes.{side}',
             f'mixamorig:{"Left" if side == "L" else "Right"}ToeBase',
             f'{"Left" if side == "L" else "Right"}ToeBase')
    bone = next((rig.data.bones.get(name) for name in names if name in rig.data.bones), None)
    return bone.head_local if bone else foot.bone.tail_local


class Session:
    def __init__(self, context, settings):
        self.root_path, self.python = runtime()
        self.scene, self.settings = context.scene, settings
        self.rig = rig = settings.target
        animation = rig.animation_data if rig and rig.type == 'ARMATURE' else None
        if not animation or not animation.action:
            raise ValueError('请先为目标角色生成视频动画')
        if animation_preview.KEY in rig or settings.candidate_state or rig.mode not in ('OBJECT', 'POSE'):
            raise ValueError('请结束姿势预览和骨架编辑，再修正视频动作')
        if (animation.use_nla and any(not track.mute and len(track.strips) for track in animation.nla_tracks)) or animation.use_tweak_mode:
            raise ValueError('当前动画受 NLA 叠加影响，请先使用独立 Action')
        self.action = animation.action
        self.slot = getattr(animation, 'action_slot_handle', None)
        if len(getattr(self.action, 'slots', ())) > 1:
            raise ValueError('请先使用只绑定目标骨架的独立 Action')
        if self.action.get(SOURCE):
            raise ValueError('此动作已做过支撑精修，请切回原动作再尝试')
        mapping = pose_tools.mapping_for(settings) or retarget.auto_map(rig)
        self.bones = [rig.pose.bones.get(mapping.get(role, '')) for role in ROLES]
        if not all(self.bones) or len({bone.name for bone in self.bones}) != len(ROLES):
            raise ValueError('请核对骨盆和双腿的大腿、小腿、脚 FK 映射')
        if contact_solver.root_control(rig, mapping, settings.root_bone) != self.bones[0]:
            raise ValueError('该动作的平移控制与骨盆分离，当前精修需要同一个 FK 根控制')
        for index, bone in enumerate(self.bones[1:], 1):
            if retarget._driver_controls_rotation(rig, bone) or retarget._active_rotation_constraints(bone):
                raise ValueError('腿部旋转受约束或驱动控制：' + bone.name)
            expected = self.bones[(0, 0, 0, 1, 2, 3, 4)[index]]
            if expected not in bone.parent_recursive:
                raise ValueError('腿部 FK 骨架层级不连续：' + bone.name)
        self.world = rig.matrix_world.copy()
        scale = self.world.to_scale()
        if self.world.to_3x3().determinant() <= 0 or min(scale) < 1e-8 or max(scale) / min(scale) > 1.001:
            raise ValueError('支撑精修需要正向均匀骨架缩放')
        first, last = math.ceil(self.action.frame_range[0]), math.floor(self.action.frame_range[1])
        if self.scene.use_preview_range:
            first = max(first, self.scene.frame_preview_start)
            last = min(last, self.scene.frame_preview_end)
        self.frames = tuple(range(first, last + 1))
        self.fps = self.scene.render.fps / self.scene.render.fps_base
        if not max(6, round(.25 * self.fps)) <= len(self.frames) <= 10000:
            raise ValueError('请选择至少 0.25 秒、最多 10000 帧的连续动作范围')
        self.original_frame = self.scene.frame_current, self.scene.frame_subframe
        self.original_signature = signature(self.action)
        self.rest = np.zeros((22, 3))
        self.rest[list(JOINTS)] = [tuple(self.world @ bone.bone.head_local) for bone in self.bones]
        self.rest[10] = self.world @ toe_rest(rig, self.bones[5], 'L')
        self.rest[11] = self.world @ toe_rest(rig, self.bones[6], 'R')
        self.rest_rotation = np.array([np.asarray((self.world @ b.bone.matrix_local).to_quaternion().to_matrix()) for b in self.bones])
        self.matrix, self.parent, self.points, self.rotation = [], [], [], []
        self.process = self.log = self.result = self.candidate = None
        self.index = 0
        self.phase = 'sample'
        self.folder = addon().runtime_dir() / 'videos' / ('rgb_contact_' + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)

    def guard(self, action):
        if (bpy.context.scene != self.scene or self.settings.target != self.rig or self.rig.mode not in ('OBJECT', 'POSE')
                or not self.rig.animation_data or self.rig.animation_data.action != action
                or getattr(self.rig.animation_data, 'action_slot_handle', None) != self.slot):
            raise ValueError('场景、目标或动作已改变，取消本次支撑精修')

    def restore_frame(self):
        self.scene.frame_set(self.original_frame[0], subframe=self.original_frame[1])

    def capture(self):
        self.scene.frame_set(self.frames[self.index])
        if np.max(np.abs(np.asarray(self.rig.matrix_world) - np.asarray(self.world))) > .0001:
            raise ValueError('骨架对象有位移动画，请先把移动烘焙到骨盆控制骨')
        matrices = [b.matrix.copy() for b in self.bones]
        self.matrix.append(matrices)
        self.parent.append([b.parent.matrix.copy() if b.parent else None for b in self.bones])
        points = np.zeros((22, 3))
        points[list(JOINTS)] = [tuple(self.world @ m.translation) for m in matrices]
        self.points.append(points)
        self.rotation.append([np.asarray((self.world @ m).to_quaternion().to_matrix()) for m in matrices])

    def launch(self):
        self.restore_frame()
        np.savez_compressed(self.folder / 'input.npz', rest=self.rest, points=self.points,
                            rotation=self.rotation, rest_rotation=self.rest_rotation, fps=self.fps)
        log_path = self.folder / 'worker.log'
        self.log = log_path.open('w', encoding='utf8')
        self.settings.last_log = str(log_path)
        self.process = subprocess.Popen([str(self.python), '-I', '-X', 'utf8', str(Path(__file__).with_name('worker_rgb_contact.py')),
                                         str(self.root_path), str(self.folder)], stdout=self.log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.phase = 'solve'

    def accept(self):
        code = self.process.poll()
        if code is None:
            self.settings.status = '正在计算连续脚底支撑…'
            return True
        self.log.close()
        if code:
            lines = (self.folder / 'worker.log').read_text(encoding='utf8', errors='replace').splitlines()
            raise ValueError(lines[-1] if lines else 'RGB MoCap 接触求解失败')
        self.report = json.loads((self.folder / 'report.json').read_text(encoding='utf8'))
        result_path = self.folder / 'result.npz'
        if not result_path.is_file():
            self.settings.status = '未检测到可改善的支撑区间，保留当前动作'
            return False
        if signature(self.action) != self.original_signature:
            raise ValueError('原动作在运行期间被编辑，本次结果未应用')
        with np.load(result_path, allow_pickle=False) as result:
            self.result = {name: result[name] for name in ('points', 'rotations')}
        n = len(self.frames)
        if (self.result['points'].shape != (n, 7, 3) or self.result['rotations'].shape != (n, 7, 3, 3)
                or not all(np.isfinite(value).all() for value in self.result.values())):
            raise ValueError('接触修正输出格式无效')
        self.candidate = self.action.copy()
        self.candidate.name = self.action.name + ' · 支撑精修'
        retarget._assign_action(self.rig, self.candidate, self.slot)
        self.phase, self.index = 'bake', 0
        return True

    def bake(self):
        i = self.index
        frame = self.frames[i]
        world_inverse = self.world.inverted()
        world_rotation = self.world.to_quaternion().inverted()
        changes = {}
        for j, bone in enumerate(self.bones):
            old = self.matrix[i][j]
            location = world_inverse @ Vector(self.result['points'][i, j])
            rotation = world_rotation @ Matrix(self.result['rotations'][i, j]).to_quaternion()
            desired = Matrix.LocRotScale(location, rotation, old.to_scale())
            parent_matrix = self.parent[i][j]
            parent = bone.parent
            if parent is not None:
                ancestor = next((candidate.name for candidate in bone.parent_recursive if candidate.name in changes), None)
                if ancestor is not None:
                    parent_matrix = changes[ancestor] @ parent_matrix
            options = {} if parent is None else dict(parent_matrix=parent_matrix, parent_matrix_local=parent.bone.matrix_local)
            local = bone.bone.convert_local_to_pose(desired, bone.bone.matrix_local, invert=True, **options)
            changes[bone.name] = desired @ old.inverted()
            if j == 0:
                bone.location = local.translation
                bone.keyframe_insert(data_path='location', frame=frame, group=bone.name)
            else:
                q = local.to_quaternion()
                if bone.rotation_mode == 'QUATERNION' and bone.rotation_quaternion.dot(q) < 0:
                    q.negate()
                retarget._set_rotation(bone, q)
                channel = {'QUATERNION': 'rotation_quaternion', 'AXIS_ANGLE': 'rotation_axis_angle'}.get(bone.rotation_mode, 'rotation_euler')
                bone.keyframe_insert(data_path=channel, frame=frame, group=bone.name)

    def verify(self):
        i = self.index
        self.scene.frame_set(self.frames[i])
        actual = np.array([tuple(self.world @ bone.head) for bone in self.bones])
        expected = self.result['points'][i]
        leg = np.linalg.norm(self.rest[1] - self.rest[4]) + np.linalg.norm(self.rest[4] - self.rest[7])
        if np.linalg.norm(actual - expected, axis=1).max() > .006 * leg:
            raise ValueError('目标控制骨回放与求解结果不一致，已恢复原动作')
        for j, bone in enumerate(self.bones[1:], 1):
            actual = (self.world @ bone.matrix).to_quaternion()
            expected = Matrix(self.result['rotations'][i, j]).to_quaternion()
            if abs(actual.dot(expected)) < math.cos(math.radians(2) / 2):
                raise ValueError('腿部骨骼旋转由其他控制覆盖，已恢复原动作')

    def tick(self):
        self.guard(self.candidate or self.action)
        if self.phase == 'solve':
            return self.accept()
        stage = {'sample': self.capture, 'bake': self.bake, 'verify': self.verify}[self.phase]
        started = time.monotonic()
        while self.index < len(self.frames):
            stage()
            self.index += 1
            if time.monotonic() - started > .045:
                name = {'sample': '读取视频动作', 'bake': '写入支撑修正副本', 'verify': '核对角色回放'}[self.phase]
                self.settings.status = f'{name} {self.index}/{len(self.frames)}'
                return True
        if self.phase == 'sample':
            self.launch()
            return True
        if self.phase == 'bake':
            self.phase, self.index = 'verify', 0
            return True
        if signature(self.action) != self.original_signature:
            raise ValueError('原动作在运行期间被编辑，本次结果未应用')
        self.restore_frame()
        self.candidate[SOURCE] = self.action
        self.action.use_fake_user = True
        self.candidate.use_fake_user = True
        before, after = self.report['speed']
        self.settings.status = f'支撑精修完成：{len(self.report["segments"])} 段，估计支撑滑移减少 {(1-after/before)*100:.0f}%；已保留原动作'
        return False

    def rollback(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log and not self.log.closed:
            self.log.close()
        try:
            if self.candidate and self.rig.animation_data and self.rig.animation_data.action == self.candidate:
                retarget._assign_action(self.rig, self.action, self.slot)
            if self.candidate and self.candidate.users == 0:
                bpy.data.actions.remove(self.candidate)
            if bpy.context.scene == self.scene:
                self.restore_frame()
        except ReferenceError:
            pass


class SP_OT_rgb_contact(Operator):
    bl_idname = 'screenshot_pose.video_rgb_contact'
    bl_label = '精修脚底支撑（RGB MoCap）'
    bl_description = '复制当前视频 Action，只校正骨盆与腿脚。启用时间轴预览范围时仅处理该范围'

    def execute(self, context):
        a = addon()
        try:
            a.video_review.guard()
            if a._JOB:
                raise ValueError('请先结束当前任务')
            session = Session(context, context.scene.screenshot_pose)
            a._JOB = dict(tick=session.tick, cancel=session.rollback, scene=context.scene, purpose=self.bl_label)
            if not bpy.app.timers.is_registered(a.poll_job):
                bpy.app.timers.register(a.poll_job, first_interval=.05)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class SP_OT_rgb_original(Operator):
    bl_idname = 'screenshot_pose.video_rgb_original'
    bl_label = '查看精修前动作'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        a = addon()
        try:
            a.video_review.guard()
            if a._JOB:
                raise ValueError('请先结束当前任务')
            settings = context.scene.screenshot_pose
            rig = settings.target
            action = rig.animation_data.action if rig and rig.animation_data else None
            original = action.get(SOURCE) if action else None
            if not isinstance(original, bpy.types.Action):
                raise ValueError('当前动作没有支撑精修记录')
            retarget._assign_action(rig, original, getattr(rig.animation_data, 'action_slot_handle', None))
            retarget._update(rig)
            settings.status = '已切回精修前动作；精修版可在动作编辑器中选择'
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


OPERATORS = (SP_OT_rgb_contact, SP_OT_rgb_original)
