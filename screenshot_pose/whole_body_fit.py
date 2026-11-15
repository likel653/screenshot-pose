"""Final, reversible all-body proposal evaluated with fixed explicit contacts."""
import math
import numpy as np
from mathutils import Matrix,Vector
from . import retarget,projection_fit,contact_solver,contact_targets,camera_data,torso_fit,whole_body_math,foot_pose

def effective_weights(data,weights):
    w=np.array(weights,float,copy=True);coarse=data.get('wholebody',[])
    if len(coarse)==133 and not data.get('edited_indices'):
        for i,index in enumerate((5,7,9,11,13,15,6,8,10,12,14,16)):
            score=coarse[index].get('confidence',w[i])
            if math.isfinite(score):w[i]=min(w[i],max(0.,score))
    return w

def refine(rig,mapping,s,data,fit,prepared=None,baseline_contacts=None):
    report=dict(enabled=True,accepted=False,touched=[],warnings=[],camera_fixed=True,reason='no_clear_gain',root_delta=[0.,0.,0.])
    if s.contact_enabled and abs(s.strength-1)>1e-6:report['reason']='contacts_require_full_strength';return report
    snapshot=retarget.snapshot_pose(rig);initial=projection_fit.points(rig,mapping);obs=np.array(fit['observed']);original_weights=np.array(fit['weights']);weights=effective_weights(data,original_weights)
    if (weights>.1).sum()<6:report['reason']='insufficient_reliable_observations';return report
    camera=fit['camera'];width=fit['image']['width'];span=max(float(np.linalg.norm(np.ptp(initial,axis=0))),.01)
    pins={};prepared=prepared or [];frozen_targets={job['object'].name:np.array(job['object'].matrix_world) for job in prepared if job.get('object') is not None}
    for job in prepared:
        end=5 if job['kind']=='FOOT' and job['side']=='L' else 11 if job['kind']=='FOOT' else 2 if job['side']=='L' else 8
        pins[end]=dict(target=list(job['target']),pole=list(job['pole']),explicit_pole=job.get('explicit_pole',False),kind=job['kind'],side=job['side'])
    if s.ground_feet and not s.contact_enabled:
        for side in contact_solver.grounded_sides(s):
            end=5 if side=='L' else 11;pins[end]=dict(target=initial[end].tolist(),kind='FOOT',side=side)
    root=None;root_reason=None;limit=span*s.whole_body_root_limit*s.strength
    if s.contact_enabled and not s.support_move_pelvis:limit=0.
    if s.contact_enabled:limit=min(limit,s.support_pelvis_limit)
    try:root=contact_solver.root_control(rig,mapping,s.root_bone)
    except ValueError as exc:limit=0.;root_reason=str(exc)
    # Verify root translation really moves all mapped FK chains, including rigs
    # whose controls follow through constraints instead of direct parenting.
    if root and limit>0:
        before=root.matrix_basis.copy();delta=Vector((.001,.0015,.0005))
        try:
            contact_solver._translate_world(rig,root,delta);moved=projection_fit.points(rig,mapping)
            if np.max(np.linalg.norm(moved-initial-np.array(delta),axis=1))>2e-5:limit=0.;root_reason='root_does_not_translate_entire_fk_body'
        finally:root.matrix_basis=before;retarget._update(rig)
    lower=np.repeat(-limit,3);upper=np.repeat(limit,3)
    if s.contact_enabled and baseline_contacts:
        dz=baseline_contacts.get('pelvis_delta_z',0.)
        lower[2]=max(lower[2],-s.support_pelvis_limit-dz);upper[2]=min(upper[2],s.support_pelvis_limit-dz)
    controls=torso_fit.controls_for(rig,mapping,s.strength,True,s.whole_body_rotation_limit,respect_locks=True) if s.whole_body_rotation_limit>0 else []
    before_pred=camera_data.project(initial,camera);before=camera_data.error(before_pred,obs,original_weights)
    report.update(before_px=before*width,contacts=list(pins),root_translation_available=bool(limit>0),root_unavailable_reason=root_reason,controls=[c['name'] for c in controls],weight_kind='detector_scores_not_calibrated_visibility')
    try:proposal=whole_body_math.solve(initial,obs,weights,camera,controls,pins,lower,upper,steps=min(s.fit_iterations,55),regularization=max(s.fit_regularization,.02))
    except Exception:
        retarget.restore_pose(rig,snapshot);raise
    report.update(proposal_px=proposal['after']*width,proposal_anatomy=proposal['anatomy'],proposal_valid=proposal['valid'],proposal_joint_guard=proposal['reliable_joint_guard'],proposal_max_length_error=proposal['max_length_error'],proposed_root_delta=proposal['root_delta'].tolist(),reason=proposal['reason'])
    if not proposal['accepted']:
        retarget.restore_pose(rig,snapshot);return report
    ends={mapping[role+'.'+side]:rig.pose.bones[mapping[role+'.'+side]].matrix.to_quaternion().copy() for role in ('hand','foot') for side in ('L','R')}
    if mapping.get('head') in rig.pose.bones:ends[mapping['head']]=rig.pose.bones[mapping['head']].matrix.to_quaternion().copy()
    head=rig.pose.bones.get(mapping.get('head',''));head_before=np.array(rig.matrix_world@head.head) if head else None
    touched=[]
    try:
        if root and limit>0 and np.linalg.norm(proposal['root_delta'])>1e-8:
            contact_solver._translate_world(rig,root,Vector(proposal['root_delta']));touched.append(root.name)
        world=rig.matrix_world.to_quaternion().to_matrix()
        for i,c in enumerate(controls):
            b=rig.pose.bones[c['name']];R=whole_body_math.fit_math.rotation(proposal['torso_parameters'][3*i:3*i+3]);dq=(world.transposed()@Matrix(R.tolist())@world).to_quaternion()
            retarget._set_rotation(b,retarget._to_local_rotation(b,dq@b.matrix.to_quaternion()));retarget._update(rig);touched.append(b.name)
        for side in ('L','R'):
            for roles,ids in [(('upper_arm','forearm','hand'),(0,1,2) if side=='L' else (6,7,8)),(('thigh','shin','foot'),(3,4,5) if side=='L' else (9,10,11))]:
                bones=[rig.pose.bones[mapping[r+'.'+side]] for r in roles];a,b,e=ids;q=proposal['points']
                projection_fit._aim_link(rig,bones[0],bones[1],q[b]-q[a]);projection_fit._aim_link(rig,bones[1],bones[2],q[e]-q[b]);touched.extend(bone.name for bone in bones)
        for name,q in ends.items():
            b=rig.pose.bones[name];retarget._set_rotation(b,retarget._to_local_rotation(b,q));retarget._update(rig);touched.append(name)
        cleanup=contact_targets.apply(rig,mapping,s,data,prepared=prepared,move_root=False) if prepared else None
        if cleanup:touched+=cleanup['touched']
        actual=projection_fit.points(rig,mapping);pred=camera_data.project(actual,camera);error=camera_data.error(pred,obs,original_weights)
        contact_error=max((np.linalg.norm(actual[e]-np.asarray(job['target'])) for e,job in pins.items()),default=0.)
        length_error=max(abs(np.linalg.norm(actual[v]-actual[u])/max(np.linalg.norm(initial[v]-initial[u]),1e-8)-1) for r,m,e in whole_body_math.CHAINS for u,v in ((r,m),(m,e)))
        initial_contact_error=max((np.linalg.norm(initial[e]-np.asarray(job['target'])) for e,job in pins.items()),default=0.)
        physical=whole_body_math.anatomy(actual,initial);old_errors=np.linalg.norm(before_pred-obs,axis=1);new_errors=np.linalg.norm(pred-obs,axis=1);reliable=weights>.75
        joint_guard=bool(np.all(new_errors[reliable]<=np.maximum(old_errors[reliable]*1.2,old_errors[reliable]+(.03 if initial_contact_error>.003 else .006))))
        head_drift=float(np.linalg.norm(camera_data.project(np.array([rig.matrix_world@head.head]),camera)-camera_data.project(head_before[None],camera))) if head else 0.
        contact_gain=initial_contact_error>.003 and error<before*1.2+.015
        gain=before-error>max(.0005,before*.02) or contact_gain
        support_guard=not cleanup or (all(abs(v['sole_clearance'])<.003 and v['rotation_error_deg']<3 for v in cleanup['feet'].values()) and all(v['palm_error']<.003 and v['rotation_error_deg']<3 for v in cleanup['hands'].values()))
        target_guard=all(np.max(abs(np.array(job['object'].matrix_world)-frozen_targets[job['object'].name]))<1e-5 for job in prepared if job.get('object') is not None)
        valid=bool(gain and joint_guard and physical['valid'] and support_guard and target_guard and contact_error<.002 and length_error<.002 and head_drift<(.07 if contact_gain else .03))
        report.update(actual_px=error*width,contact_error=float(contact_error),bone_length_relative_error=float(length_error),anatomy=physical,head_screen_drift_px=head_drift*width,reliable_joint_guard=joint_guard,sole_and_palm_guard=bool(support_guard),fixed_targets_unchanged=bool(target_guard))
        if not valid:
            retarget.restore_pose(rig,snapshot);report['reason']='evaluated_whole_body_proposal_rejected';return report
        report.update(accepted=True,reason='verified_contact_reach_gain' if contact_gain else 'verified_all_body_gain',after_px=error*width,root_delta=proposal['root_delta'].tolist(),touched=sorted(set(touched)))
        if cleanup:
            cleanup['pelvis_delta_z']=float((baseline_contacts or {}).get('pelvis_delta_z',0.)+proposal['root_delta'][2]);report['final_contacts']=cleanup
        fit.update(accepted=True,after_px=error*width,after_projection=pred.tolist(),points=actual.tolist(),touched=sorted(set(fit['touched']+touched)))
        return report
    except Exception:
        retarget.restore_pose(rig,snapshot);raise
