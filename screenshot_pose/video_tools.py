"""Cancellable video recognition and transactional, isolated animation baking."""
import importlib,json,math,time,uuid,copy
from functools import lru_cache
from pathlib import Path
import bpy
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion
from . import video_data,video_stability,video_temporal,video_hands,video_resample,video_motion,video_support,video_wrists

def addon():return importlib.import_module(__package__)

def recording_done(scene,folder):
    a=addon();folder=Path(folder).resolve();data=json.loads((folder/'result.json').read_text(encoding='utf8'));s=scene.screenshot_pose
    if data.get('cancelled'):
        s.status='屏幕录制已取消，原视频和姿势缓存保留';return
    video=video_data.child(folder,data.get('video'));record=data['record'];sequence=data.get('sequence')
    cache=video_data.child(folder,sequence) if sequence else None
    if cache:video_data.read(cache)
    s.video_path=str(video);s.video_start=0.;s.video_end=0.;s.video_cache=str(cache) if cache else ''
    s.video_capture_summary='最近录制：%.1f 秒 · %d×%d · %d fps'%(record['video_seconds'],record['width'],record['height'],record['fps'])
    if record.get('duplicates'):s.video_capture_summary+='；%d 帧重复补时'%record['duplicates']
    s.status='屏幕视频已录制并识别，可生成到所选角色' if cache else '屏幕录制已保存，可点击识别录制视频'
    if data.get('inference_error'):s.status='录制已保存，识别失败：'+data['inference_error']

class SP_OT_video_record(Operator):
    bl_idname='screenshot_pose.video_record';bl_label='框选屏幕视频并录制'
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose
        try:
            if a._JOB:raise ValueError('已有任务正在运行')
            python=a.runtime_python(s.video_backend if s.video_record_identify else 'MEDIAPIPE')
            if not python.is_file():python=a.runtime_python('MEDIAPIPE')
            if not python.is_file():raise ValueError('请先准备本地识别环境')
            out=a.runtime_dir()/'screen_videos'/uuid.uuid4().hex
            command=[python,a.HERE/'record_worker.py','--output',out,'--duration',str(s.video_record_seconds),'--record-fps',str(s.video_record_fps)]
            if s.video_record_identify:
                command+=['--identify','--model',a.runtime_dir()/'pose_landmarker_full.task','--backend',s.video_backend,'--sample-fps',str(s.video_fps),'--confidence',str(s.confidence)]
                if s.video_backend=='SAM3D':command+=a.sam_root_args()
            command+=['--sampling',s.video_sampling,'--max-sample-fps',str(s.video_sampling_max)]
            if not s.video_continuous:command+=['--no-continuity']
            a.start_job(context.scene,command,'框选屏幕视频并录制',lambda scene:recording_done(scene,out));a._JOB['progress']=out/'progress.json';a._JOB['record_folder']=out
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

class Bake:
    def __init__(self,context,s,path,*,stage_observer=None):
        a=addon()
        a.video_review.guard()
        if not s.target or s.target.type!='ARMATURE':raise ValueError('先选择目标角色骨架')
        if s.target.mode=='EDIT':raise ValueError('请先退出骨架编辑模式')
        if s.target.library and not s.target.override_library:raise ValueError('链接角色需要先创建库覆盖')
        if s.candidate_state:raise ValueError('请先确认或取消姿势预览')
        self.path=Path(path).resolve();self.data=video_data.read(self.path)
        self.stage_observer=stage_observer
        self.scene=context.scene;self.rig=s.target;self.s=s
        self.timed=video_data.timed_frames(self.data,self.scene.render.fps/self.scene.render.fps_base,s.video_start_frame,s.video_max_gap)
        self.interpolation=getattr(s,'video_interpolation','HERMITE')
        if self.interpolation!='LEGACY':
            fps=self.scene.render.fps/self.scene.render.fps_base;origin=self.data.get('start_seconds',self.data['frames'][0]['time'])
            self.timed=[(s.video_start_frame+round((item['time']-origin)*fps),item) for item in self.data['frames'] if item['status']=='ok']
        if not s.mapping:a.set_mapping(s)
        if not any(r.enabled and r.bone for r in s.mapping):raise ValueError('请先配置角色骨骼映射')
        # Validate payloads on cancellable solve ticks, with rollback on failure.
        a.animation_preview.end(self.rig)
        self.before=a.retarget.snapshot_pose(self.rig);self.backup=self.rig.get('screenshot_pose_backup')
        self.restore_record=self.rig.get('sp_video_restore')
        self.nla={t.name:t.mute for t in self.rig.animation_data.nla_tracks} if self.rig.animation_data else {}
        self.frame=self.scene.frame_current;self.range=(self.scene.frame_start,self.scene.frame_end)
        self.original=bpy.data.actions.get(self.before['action']) if self.before['action'] else None
        self.action=None;self.closed=False;self.committed=False;self.names=set();self.properties=set();self.warnings=set()
        self.folder=a.runtime_dir()/'videos'/('bake_'+uuid.uuid4().hex);self.folder.mkdir(parents=True)
        self.frames=[];self.smooth=s.video_smooth;self.temporal_mode=s.video_temporal_mode;self.review=s.video_stabilize;self.corrections={};self.issues=[];self.recognition_issues=[];self.cut_indices=[]
        self.started=time.perf_counter();self.stage_times={};self.optimize=s.video_optimize
        self.hand_continuity=getattr(s,'video_hand_continuity',True);self.hand_gap=getattr(s,'video_hand_gap',.6)
        self.palm_plan={};self.palm_events=[];self.finger_controls=set();self.hand_estimates=[]
        self.hand_cuts=set()
        self.motion_observations=[]
        self.names.update(a.frame_tools.controls(s))
        self.properties.update((b.name,k) for b,k,v in a.retarget._fk_switches(self.rig,a.pose_tools.mapping_for(s)))
        self.base=dict(self.before,action=None,action_slot_handle=None)
        # Re-baking this clip from a posed frame must not add the old root offset again.
        if self.original and self.original.get('sp_video_source') and Path(self.original['sp_video_source']).resolve()==self.path:
            try:
                root=a.contact_solver.root_control(self.rig,a.pose_tools.mapping_for(s),s.root_bone)
                origin=json.loads(self.original.get('sp_video_root_origin','{}'))
                if not origin and self.restore_record:
                    saved=json.loads(self.restore_record)['snapshot']['bones'][root.name]
                    origin=dict(bone=root.name,location=saved['location'])
                if origin.get('bone')==root.name and len(origin['location'])==3 and all(math.isfinite(v) for v in origin['location']):
                    self.base=copy.deepcopy(self.base);state=self.base['bones'][root.name];state['location']=list(origin['location'])
                    for i in range(3):state['matrix_basis'][i][3]=origin['location'][i]
            except (ValueError,TypeError,KeyError):self.warnings.add('未能读取旧动作的骨盆参考，将以当前姿势位置作为起点')
        self.proxy=a.pose_tools.SettingsProxy(s,preview_name='',keyframe=False,candidate_state='',chosen_candidate='',quality_warnings='',
            strength=1.,switch_fk=True,ground_feet=False,contact_enabled=False,fit_projection=s.video_fit,fit_video_observations=True,fit_model='ORTHO',fit_torso=False,head_consensus=False,whole_body=False,
              preserve_source_pose=False,auto_full_quality=False,auto_depth_select=False,hand_left='NONE',hand_right='NONE',palm_flip_l=False,palm_flip_r=False,
            head_yaw=0.,head_pitch=0.,head_roll=0.,palm_roll_l=0.,palm_roll_r=0.,hand_candidate_l='0',hand_candidate_r='0',
            full_source_rotation=s.video_native,orient_head=True,orient_hand_l=True,orient_hand_r=True,foot_pose=True,video_batch_updates=self.optimize)
        self.generator=self.run()

    def rollback(self):
        if self.closed:return
        a=addon();a.retarget.restore_pose(self.rig,self.before,update=False)
        for t in self.rig.animation_data.nla_tracks if self.rig.animation_data else ():
            if t.name in self.nla:t.mute=self.nla[t.name]
        a.pose_tools.restore_backup(self.rig,self.backup)
        if self.restore_record is None:
            if 'sp_video_restore' in self.rig:del self.rig['sp_video_restore']
        else:self.rig['sp_video_restore']=self.restore_record
        self.scene.frame_start,self.scene.frame_end=self.range;self.scene.frame_set(self.frame)
        if self.action:
            self.action.use_fake_user=False
            if self.action.users==0:bpy.data.actions.remove(self.action)
        self.closed=True

    def tick(self):
        if bpy.context.scene!=self.scene or self.s.target!=self.rig:raise ValueError('场景或目标角色已改变，已撤回生成')
        if self.scene.frame_current!=self.frame:raise ValueError('生成期间时间轴被改变，已撤回生成')
        try:next(self.generator);return True
        except StopIteration:return False

    @lru_cache(maxsize=24)
    def raw(self,index):return json.loads((self.folder/('%06d.json'%index)).read_text(encoding='utf8'))

    def rotation(self,index,name):return self.corrections.get(index,{}).get(name,self.raw(index)[name]['rotation'])

    def run(self):
        a=addon();rig=self.rig;s=self.s
        if self.original:self.original.use_fake_user=True
        rig.animation_data_create();rig.animation_data.action=None
        for track in rig.animation_data.nla_tracks:track.mute=True
        count=len(self.timed)
        stage=time.perf_counter()
        if self.optimize:
            self.proxy.values['video_sole_calibration']=a.foot_pose.sole_calibration(rig,a.pose_tools.mapping_for(s),s.shoe_mesh.name if s.shoe_mesh else '')
        finger_controls={side:[b.name for finger in a.finger_pose.FINGERS for segment in (1,2,3) if (b:=a.finger_pose.control(rig,finger,segment,side))] for side in ('L','R')}
        self.finger_controls=set(n for names in finger_controls.values() for n in names)
        self.finger_sides={n:side for side,names in finger_controls.items() for n in names}
        finger_dependencies={}
        for side in ('L','R'):
            hand=rig.pose.bones.get(a.pose_tools.mapping_for(s).get('hand.'+side,''))
            for finger in a.finger_pose.FINGERS:
                base=rig.pose.bones.get('c_'+finger+'1_base.'+side.lower())
                for segment in (1,2,3):
                    bone=a.finger_pose.control(rig,finger,segment,side)
                    if bone:finger_dependencies[bone.name]=[b for b in (hand,base) if b]
        if self.hand_continuity and getattr(s,'video_palm_candidates',False):
            observations=[]
            for index,(frame,item) in enumerate(self.timed):
                observations.append(video_hands.compact(a.pose_data.read_pose(video_data.child(self.path.parent,item['pose']))))
                if index%16==0:s.status='规划手掌候选 %d / %d'%(index+1,count);yield
            resets={'scene_cut','time_gap','missing_person','roi_jump'}
            cuts={i for i,d in enumerate(observations) if not d.get('video_tracking') or d.get('video_tracking',{}).get('reset') in resets}
            self.hand_cuts=cuts
            if any(d.get('orientation',{}).get('hands') for d in observations) and any(not d.get('video_tracking') for d in observations):
                self.warnings.add('部分缓存缺少连续跟踪标记，手部衔接在这些采样处中断；可重新识别后再生成')
            self.palm_plan,self.palm_events=video_hands.plan_palms(observations,[item['time'] for frame,item in self.timed],s.orientation_threshold,cuts)
        for index,(frame,item) in enumerate(self.timed):
            a.retarget.restore_pose(rig,self.base)
            data=a.pose_data.read_pose(video_data.child(self.path.parent,item['pose']))
            self.motion_observations.append(video_motion.observation(data))
            if index in self.palm_plan:
                data=copy.deepcopy(data);data['orientation']['hands'].update(self.palm_plan[index])
            tracking=data.get('video_tracking',{})
            if (self.hand_continuity or getattr(s,'video_wrist_repair',True) or getattr(s,'video_local_memory',True)) and not tracking:
                self.hand_cuts.add(index)
                if data.get('orientation',{}).get('hands'):self.warnings.add('部分缓存缺少连续跟踪标记，手部衔接在这些采样处中断；可重新识别后再生成')
            if tracking.get('reset') in ('scene_cut','time_gap','missing_person','roi_jump') and index:self.cut_indices.append(index)
            for event in tracking.get('events',[]):
                if event.endswith('_side_ambiguity_review'):
                    role='hand' if event.startswith('wrists') else 'foot'
                    for side in ('L','R'):
                        name=a.pose_tools.mapping_for(s).get(role+'.'+side)
                        if name:self.recognition_issues.append(dict(index=index,control=name,reason='识别阶段左右对应疑点，请核对原图',repaired=False))
            p=self.proxy;p.values['json_path']=str(video_data.child(self.path.parent,item['pose']))
            points,_=a.pose_data.to_joints(data,p.coordinate_mode,p.depth,p.mirror,p.yaw,p.threshold)
            from mathutils import Vector
            transform=rig.matrix_world.inverted_safe().to_3x3();joints={n:transform@Vector(v) for n,v in points.items()}
            options={}
            if self.stage_observer is not None:
                options['stage_observer']=lambda stage,target,result:self.stage_observer(index,frame,item,stage,target,result)
            result=a._apply_current_once(bpy.context,p,joints_override=joints,data_override=data,**options)
            self.warnings.update(result.get('warnings',[]))
            confidence=video_stability.confidences(data,a.pose_tools.mapping_for(s),finger_controls,result.get('fingers',{}) if self.hand_continuity else None)
            observed_fingers=set(result.get('fingers',{}).get('touched',[]))
            for bone in rig.pose.bones:
                old=self.before['bones'][bone.name]
                if any(abs(bone.matrix_basis[i][j]-old['matrix_basis'][i][j])>1e-7 for i in range(4) for j in range(4)):self.names.add(bone.name)
                for key,value in old['properties'].items():
                    if bone.get(key)!=value:self.properties.add((bone.name,key))
            record={b.name:dict(location=list(b.location),rotation=list(b.matrix_basis.to_quaternion()),scale=list(b.scale),
                confidence=confidence.get(b.name,0.),properties={k:float(b[k]) for k in self.before['bones'][b.name]['properties'] if k in b}) for b in rig.pose.bones}
            record['__object_properties__']={k:rig[k] for k in self.before.get('object_properties',{})}
            record['__wrists__']=video_wrists.observe(rig,a.pose_tools.mapping_for(s),p,data,result.get('orientation',{}))
            if self.hand_continuity:
                for side,names in finger_controls.items():
                    protected=bool(data.get('edited_indices') or a.hand_presets.manual_fingers(data,side) or a.hand_presets.selected(p,data,side)!='NONE')
                    for name in names:
                        record[name]['observed']=name in observed_fingers;record[name]['protected']=protected
                        record[name]['dependency_state']={b.name:a.retarget.snapshot_properties(rig,b) for b in finger_dependencies.get(name,[])}
            (self.folder/('%06d.json'%index)).write_text(json.dumps(record),encoding='utf8')
            self.frames.append(frame);s.status='适配角色 %d / %d 帧'%(index+1,count)
            yield
        self.stage_times['solve_seconds']=time.perf_counter()-stage
        yield from self.finish()

    def finish(self):
        """Review and key solved records; raw records remain unchanged."""
        a=addon();rig=self.rig;s=self.s;count=len(self.timed);stage=time.perf_counter()
        times=[item['time'] for frame,item in self.timed]
        missing_boundaries={i for i in range(1,count) if any(times[i-1]<f['time']<times[i] and f['status']=='missing' for f in self.data['frames'])}
        boundaries=set(self.cut_indices)|missing_boundaries
        if self.review:
            self.corrections,self.issues=video_stability.inspect([item['time'] for frame,item in self.timed],self.raw,self.names)
        self.issues.extend(self.recognition_issues)
        for cut in self.cut_indices:
            for i in range(count):
                if abs(self.timed[i][1]['time']-self.timed[cut][1]['time'])<=max(.3,self.smooth):
                    self.corrections.pop(i,None)
                    for name in self.names:self.issues.append(dict(index=i,control=name,reason='镜头或跟踪重置附近：保留原识别，建议分段',repaired=False))
        manual_locked=video_wrists.locked_controls((self.raw(i) for i in range(count)),getattr(self,'finger_sides',{}))
        for index,name in manual_locked:
            self.corrections.get(index,{}).pop(name,None)
        self.issues=[issue for issue in self.issues if (issue['index'],issue['control']) not in manual_locked]
        self.issues.sort(key=lambda item:(item['index'],item['control']))
        if self.hand_continuity:
            fixes,self.hand_estimates=video_hands.finger_gaps([item['time'] for frame,item in self.timed],self.raw,self.finger_controls,self.hand_gap,set(self.cut_indices)|self.hand_cuts,max(.5,self.s.orientation_threshold))
            if getattr(s,'video_local_memory',True):
                memory,events=video_hands.carry_fingers(times,self.raw,self.finger_controls,getattr(s,'video_local_memory_seconds',.35),boundaries|self.hand_cuts,fixes)
                for index,values in memory.items():fixes.setdefault(index,{}).update(values)
                self.hand_estimates.extend(events)
            keys={(index,name) for index,values in fixes.items() for name in values}
            self.issues=[issue for issue in self.issues if (issue['index'],issue['control']) not in keys]
            for index,values in fixes.items():self.corrections.setdefault(index,{}).update(values)
            self.issues.extend(video_hands.estimate_issues(self.hand_estimates,self.finger_sides,a.pose_tools.mapping_for(s)))
            self.issues.sort(key=lambda item:(item['index'],item['control']))
        wrist_report=dict(enabled=getattr(s,'video_wrist_repair',True) or getattr(s,'video_local_memory',True),repair_enabled=getattr(s,'video_wrist_repair',True),
            memory_enabled=getattr(s,'video_local_memory',True),max_gap=getattr(s,'video_wrist_gap',.35),estimates=[],accepted=[],rejected=[])
        if wrist_report['enabled']:
            records=[{'__wrists__':self.raw(i).get('__wrists__',{})} for i in range(count)]
            targets,events,reviews=video_wrists.plan(times,records,wrist_report['max_gap'] if wrist_report['repair_enabled'] else 0.,boundaries|self.hand_cuts)
            if wrist_report['memory_enabled']:
                mapping=a.pose_tools.mapping_for(s)
                blocked_decay={(index,side) for index,values in self.corrections.items() for side in ('L','R') if mapping.get('hand.'+side) in values}
                memory,carry_events=video_wrists.carry(times,records,getattr(s,'video_local_memory_seconds',.35),boundaries|self.hand_cuts,targets,blocked_decay)
                for index,values in memory.items():targets.setdefault(index,{}).update(values)
                events.extend(carry_events)
            fixes,accepted,rejected=video_wrists.resolve(rig,self.base,self.raw,targets,a.pose_tools.mapping_for(s),s) if targets else ({},[],[])
            keys={(r['index'],r['control']) for r in accepted}
            self.issues=[i for i in self.issues if (i['index'],i['control']) not in keys]
            for index,values in fixes.items():self.corrections.setdefault(index,{}).update(values)
            accepted_sides={(r['index'],r['side']) for r in accepted}
            for event in events:
                if (event['index'],event['side']) not in accepted_sides:continue
                name=a.pose_tools.mapping_for(s)['hand.'+event['side']]
                self.issues.append(dict(index=event['index'],control=name,repaired=True,estimated=True,
                    reason='短遮挡：手腕相对前臂的有限姿势记忆' if event['method']=='forearm_relative_memory_decay' else '手腕局部时序补间：可靠前后观测衔接，仍需回放复核'))
            for review in reviews:
                if (review['index'],review['side']) in accepted_sides or (review['index']-1,review['side']) in accepted_sides:continue
                name=a.pose_tools.mapping_for(s).get('hand.'+review['side'])
                if name:self.issues.append(dict(index=review['index'],control=name,repaired=False,reason=review['reason'],angle_deg=review['degrees'],review_only=True))
            rejected.extend(dict(index=e['index'],side=e['side'],reason='记忆建议会增加恢复帧跳变，保留原结果') for e in events if e.get('quality_guard_rejected'))
            wrist_report.update(estimates=events,accepted=accepted,rejected=rejected,metadata_samples=sum(bool(row['__wrists__']) for row in records))
        for issue in self.issues:issue['frame']=self.frames[issue['index']]
        protected={(issue['index'],issue['control']) for issue in self.issues if not issue['repaired'] and not issue.get('review_only')}|manual_locked
        temporal=video_temporal.Smoother([item['time'] for frame,item in self.timed],self.raw,self.rotation,self.smooth,self.temporal_mode,protected)
        observations=self.motion_observations or [video_motion.observation(a.pose_data.read_pose(video_data.child(self.path.parent,item['pose']))) for frame,item in self.timed]
        a.retarget.restore_pose(rig,self.base)
        calibration=self.proxy.values.get('video_sole_calibration')
        if calibration is None:calibration=a.foot_pose.sole_calibration(rig,a.pose_tools.mapping_for(s),s.shoe_mesh.name if s.shoe_mesh else '')
        motion=video_support.Solver(rig,a.pose_tools.mapping_for(s),s,times,observations,boundaries,calibration)
        self.names.update(motion.touched);self.warnings.update(motion.warnings)
        self.stage_times['review_seconds']=time.perf_counter()-stage;stage=time.perf_counter()
        # Independent Action, with FK/IK properties keyed as well as rotations.
        a.retarget.restore_pose(rig,self.base,update=False)
        self.action=bpy.data.actions.new(Path(self.data.get('source_video','Video')).stem+' · 视频姿势')
        self.action.use_fake_user=True;a.retarget._assign_action(rig,self.action)
        previous={};fps=self.scene.render.fps/self.scene.render.fps_base;origin=self.data.get('start_seconds',times[0])
        output=[(frame,times[i],i) for i,frame in enumerate(self.frames)]
        resampler=None
        if self.interpolation!='LEGACY':
            resampler=video_resample.Resampler(times,self.raw,temporal.sample,self.interpolation,boundaries)
            end=self.frames[-1]
            if 'end_seconds' in self.data:end=max(end,s.video_start_frame+math.ceil((self.data['end_seconds']-origin)*fps-1e-6)-1)
            if end-s.video_start_frame>100000:raise ValueError('重采样动画超过 100000 帧，请分段生成')
            output=[(f,origin+(f-s.video_start_frame)/fps,resampler.bracket(origin+(f-s.video_start_frame)/fps)[0]) for f in range(s.video_start_frame,end+1)]
        self.output_frames=[row[0] for row in output];self.motion_frames=[]
        for baked_index,(frame,stamp,index) in enumerate(output):
            center=self.raw(index)
            for name in self.names:
                bone=rig.pose.bones[name];raw=center[name]
                rotation,location=resampler.at(stamp,name) if resampler else temporal.sample(index,name);q=Quaternion(rotation)
                if name in previous and q.dot(previous[name])<0:q.negate()
                bone.rotation_mode='QUATERNION';bone.rotation_quaternion=q
                bone.location=location
                bone.scale=raw['scale']
            for key,value in center.get('__object_properties__',{}).items():
                rig[key]=value;rig.keyframe_insert(data_path=a.retarget.object_property_path(key),frame=frame,group='MHX FK switches')
            for name,key in self.properties:
                bone=rig.pose.bones[name];bone[key]=center[name]['properties'][key];bone.keyframe_insert(data_path='["'+key+'"]',frame=frame,group=name)
            self.motion_frames.append(dict(frame=frame,time=stamp,**motion.apply(stamp)))
            for name in self.names:
                bone=rig.pose.bones[name];q=bone.matrix_basis.to_quaternion()
                if name in previous and q.dot(previous[name])<0:q.negate()
                bone.rotation_mode='QUATERNION';bone.rotation_quaternion=q;previous[name]=q.copy()
                for channel in ('rotation_quaternion','location','scale'):bone.keyframe_insert(data_path=channel,frame=frame,group=name)
            s.status='写入动画 %d / %d 帧'%(baked_index+1,len(output));yield
        switch_paths=a.retarget.location_switch_paths(rig,a.pose_tools.mapping_for(s))
        for curve in a.animation_preview.curves(self.action):
            interpolation='CONSTANT' if '"]["' in curve.data_path or curve.data_path.startswith('["') or curve.data_path in switch_paths else 'LINEAR'
            for point in curve.keyframe_points:point.interpolation=interpolation
            if interpolation=='LINEAR':
                hold_frames={self.frames[i-1] for i in boundaries} if self.interpolation=='LEGACY' else {s.video_start_frame+math.ceil((times[i]-origin)*fps-1e-8)-1 for i in boundaries}
                for point in curve.keyframe_points:
                    if round(point.co.x) in hold_frames:point.interpolation='CONSTANT'
        in_place=motion.mode=='IN_PLACE' and not motion.plan['segments']
        self.issues.extend(video_motion.review_issues(times,a.pose_tools.mapping_for(s),motion.plan['segments'],motion.records,s.video_start_frame,origin,fps))
        self.issues.sort(key=lambda issue:(issue['frame'],issue['control']))
        self.action['sp_video_source']=str(self.path);self.action['sp_video_in_place']=in_place
        if motion.root:self.action['sp_video_root_origin']=json.dumps(dict(bone=motion.root.name,location=self.raw(0)[motion.root.name]['location']))
        restore=dict(snapshot=self.before,tracks=self.nla)
        rig['sp_video_restore']=json.dumps(restore)
        a.pose_tools.restore_backup(rig,self.backup)
        if s.video_set_range:self.scene.frame_start=self.output_frames[0];self.scene.frame_end=self.output_frames[-1]
        self.scene.frame_set(self.output_frames[0]);a.retarget._update(rig)
        missing=sum(f['status']=='missing' for f in self.data['frames'])
        report=dict(action=self.action.name,source=str(self.path),frames=self.frames,frame_map=[dict(frame=f,time=item['time'],pose=item['pose']) for f,item in self.timed],controls=len(self.names),missing=missing,
            warnings=sorted(self.warnings),smooth_seconds=self.smooth,temporal_mode=self.temporal_mode,in_place=in_place,nla_suspended=self.nla,
            temporal_review=dict(enabled=self.review,repaired=sum(i['repaired'] for i in self.issues),flagged=len(self.issues),issues=self.issues),
            optimized=self.optimize,timing=dict(self.stage_times,write_seconds=time.perf_counter()-stage,total_seconds=time.perf_counter()-self.started))
        report['hand_continuity']=dict(enabled=self.hand_continuity,max_finger_gap_seconds=self.hand_gap,
            experimental_palm_candidates=getattr(s,'video_palm_candidates',False),
            palm_candidate_changes=self.palm_events,estimated_finger_controls=len(self.hand_estimates),
            estimated_finger_samples=len({e['index'] for e in self.hand_estimates}),estimates=self.hand_estimates,
            observations_are_not_estimates=True)
        report['motion']=dict(motion.report(),frames=self.motion_frames)
        report['wrist_continuity']=wrist_report
        report['local_arbitration']=dict(scope=['wrists','fingers'],priority=['manual_and_reliable_observations','bracketed_reliable_anchors','finite_local_memory','raw_body_or_baseline'],
            finite_memory_seconds=getattr(s,'video_local_memory_seconds',.35),memory_enabled=getattr(s,'video_local_memory',True),
            wrist_selected=len(wrist_report['accepted']),wrist_rejected=len(wrist_report['rejected']),
            finger_bracketed=sum(e['method']=='bracketed_finger_interpolation' for e in self.hand_estimates),
            finger_memory=sum(e['method']=='finite_finger_memory_decay' for e in self.hand_estimates),
            manual_protected_controls=len(manual_locked),
            source_cache_modified=False,whole_body_temporal_model=False)
        report['resampling']=dict(mode=self.interpolation,observed_samples=count,baked_frames=len(output),fps=fps,
            start_frame=s.video_start_frame,
            original_timestamps=times,hold_boundaries=sorted(boundaries),synthetic_frames_are_not_detections=True,
            leading_hold_seconds=max(0.,times[0]-origin),trailing_hold_seconds=max(0.,output[-1][1]-times[-1]))
        (self.folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        self.action['sp_video_report']=str(self.folder/'report.json')
        s.video_report=str(self.folder/'report.json');s.status='视频动画已生成：%d 个姿势，%d–%d 帧；原动画已保留'%(count,self.frames[0],self.frames[-1])
        if self.issues:s.status+='；复核 %d 处，自动修补 %d 处'%(len(self.issues),sum(i['repaired'] for i in self.issues))
        self.closed=True;self.committed=True

class SP_OT_video_detect(Operator):
    bl_idname='screenshot_pose.video_detect';bl_label='识别视频并缓存'
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose
        try:
            if a._JOB:raise ValueError('已有任务正在运行')
            source=Path(bpy.path.abspath(s.video_path));model=a.runtime_dir()/'pose_landmarker_full.task'
            if not source.is_file():raise ValueError('请先选择本地视频文件')
            if not a.runtime_python(s.video_backend).is_file() or not model.is_file():raise ValueError('请先准备识别模型与环境')
            out=a.runtime_dir()/'videos'/uuid.uuid4().hex
            command=[a.runtime_python(s.video_backend),a.HERE/'worker_video.py','--video',source,'--output',out,'--model',model,'--backend',s.video_backend,
                '--start',str(s.video_start),'--end',str(s.video_end),'--sample-fps',str(s.video_fps),'--confidence',str(s.confidence)]
            if s.video_backend=='SAM3D':command+=a.sam_root_args()
            if s.hand3d_video_backend!='LOCAL':command+=['--hand3d-backend',s.hand3d_video_backend]
            if not s.video_continuous:command+=['--no-continuity']
            command+=['--sampling',s.video_sampling,'--max-sample-fps',str(s.video_sampling_max)]
            def done(scene):
                sequence=video_data.read(out/'sequence.json');scene.screenshot_pose.video_cache=str(out/'sequence.json')
                scene.screenshot_pose.status='视频识别完成，可生成动画，也可换角色重复使用缓存'
                hands=sequence.get('hand3d_summary')
                if hands:scene.screenshot_pose.status+='；'+hands['backend']+' 手部采用 %d / %d，其余保留原观测'%(hands['adopted'],hands['attempted'])
            a.start_job(context.scene,command,'视频识别',done);a._JOB['progress']=out/'progress.json'
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

class SP_OT_video_bake(Operator):
    bl_idname='screenshot_pose.video_bake';bl_label='生成角色动画'
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose
        try:
            if a._JOB:raise ValueError('已有任务正在运行')
            session=Bake(context,s,bpy.path.abspath(s.video_cache))
            a._JOB=dict(tick=session.tick,cancel=session.rollback,scene=context.scene,purpose='生成视频动画',started=time.monotonic())
            bpy.app.timers.register(a.poll_job,first_interval=.05);s.status='开始适配视频姿势…';a._redraw()
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

class SP_OT_video_restore(Operator):
    bl_idname='screenshot_pose.video_restore';bl_label='恢复生成前动画';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose
        try:
            a.video_review.guard()
            if a._JOB or s.candidate_state:raise ValueError('请先结束当前任务或姿势预览')
            rig=s.target
            if not rig or 'sp_video_restore' not in rig:raise ValueError('当前角色没有视频动画恢复记录')
            a.animation_preview.end(rig);state=json.loads(rig['sp_video_restore']);a.retarget.restore_pose(rig,state['snapshot'],update=False)
            for t in rig.animation_data.nla_tracks if rig.animation_data else ():
                if t.name in state['tracks']:t.mute=state['tracks'][t.name]
            del rig['sp_video_restore'];a.retarget._update(rig);s.status='已恢复生成前动画；视频动画仍保存在动作列表'
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

class SP_OT_video_import(Operator,ImportHelper):
    bl_idname='screenshot_pose.video_import';bl_label='载入已有视频缓存'
    filename_ext='.json';filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
    def execute(self,context):
        try:
            a=addon()
            if a._JOB:raise ValueError('已有任务正在运行')
            video_data.read(self.filepath);context.scene.screenshot_pose.video_cache=self.filepath
            context.scene.screenshot_pose.status='视频缓存已载入，选择角色后生成动画'
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

class SP_OT_video_result(Operator):
    bl_idname='screenshot_pose.video_result';bl_label='查看视频复核结果'
    def execute(self,context):
        try:
            report=json.loads(Path(bpy.path.abspath(context.scene.screenshot_pose.video_report)).read_text(encoding='utf8'))
            review=report.get('temporal_review',{});issues=review.get('issues',[])
            def draw(menu,_context):
                l=menu.layout;l.label(text='动画：'+report['action'])
                l.label(text='自动修补 %d 处；待人工检查 %d 处'%(review.get('repaired',0),sum(not i['repaired'] for i in issues)))
                l.label(text='生成耗时 %.1f 秒'%report.get('timing',{}).get('total_seconds',0.))
                l.label(text='连续性：'+{'ADAPTIVE':'自适应 · 保留快动作','LEGACY':'固定平滑 · 对照','OFF':'关闭平滑'}.get(report.get('temporal_mode'),'旧版处理'))
                motion=report.get('motion',{});sampling=report.get('resampling',{})
                if sampling:l.label(text='识别 %d 个姿势 → 烘焙 %d 帧 · %s'%(sampling['observed_samples'],sampling['baked_frames'],sampling['mode']))
                if motion:
                    l.label(text='骨盆：'+{'HEIGHT':'相对高度','SCREEN':'相对画面位移','IN_PLACE':'原地'}.get(motion['mode'],motion['mode']))
                    l.label(text='自动支撑 %d 段；求解接受 %d 帧次，拒绝 %d 帧次'%(len(motion['segments']),motion['accepted'],motion['rejected']))
                    for warning in motion.get('warnings',[])[:3]:l.label(text=warning,icon='ERROR')
                for item in issues[:16]:l.label(text='第 %d 帧 · %s · %s'%(item['frame'],item['control'],'已修补' if item['repaired'] else '需检查'))
                if len(issues)>16:l.label(text='其余记录见识别环境 videos/bake_*/report.json')
                l.label(text='未标记不代表识别正确；持续错误仍需手工校正')
            context.window_manager.popup_menu(draw,title='视频动作复核',icon='INFO')
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

OPERATORS=(SP_OT_video_detect,SP_OT_video_bake,SP_OT_video_restore,SP_OT_video_import,SP_OT_video_result,SP_OT_video_record)
