"""Verify hierarchical proposals against actual evaluated Blender controls."""
import math
import numpy as np
from mathutils import Matrix,Vector
from . import retarget,projection_fit,contact_solver,camera_data,torso_math

def controls_for(rig,mapping,strength=1.,allow_shoulders=True,limit_degrees=12.,respect_locks=False):
    safe=lambda b:b and not retarget._active_rotation_constraints(b) and not retarget._driver_controls_rotation(rig,b) and not (respect_locks and any(b.lock_rotation))
    names=[mapping.get(r,'') for r in ('hips','spine','chest')]
    if allow_shoulders:names+=['c_shoulder.l','c_shoulder.r']
    bones=[]
    for name in names:
        b=rig.pose.bones.get(name)
        if safe(b) and b not in bones:bones.append(b)
    bones.sort(key=lambda b:len(b.parent_recursive));joint_bones=[rig.pose.bones[mapping[projection_fit.JOINT_ROLES[n.split('.')[0]]+'.'+n[-1]]] for n in projection_fit.NAMES]
    controls=[];original_points=projection_fit.points(rig,mapping)
    for b in bones:
        ids=[i for i,j in enumerate(joint_bones) if b in j.parent_recursive]
        if not ids:
            basis=b.matrix_basis.copy();v=np.array([.004,.003,.005]);world=rig.matrix_world.to_quaternion().to_matrix()
            try:
                delta=world.transposed()@Matrix(torso_math.fit_math.rotation(v).tolist())@world
                retarget._set_rotation(b,retarget._to_local_rotation(b,delta.to_quaternion()@b.matrix.to_quaternion()));retarget._update(rig)
                moved=projection_fit.points(rig,mapping);pivot=np.array(rig.matrix_world@b.head);expected=(original_points-pivot)@torso_math.fit_math.rotation(v).T+pivot
                ids=[i for i in range(12) if np.linalg.norm(moved[i]-original_points[i])>1e-6 and np.linalg.norm(moved[i]-expected[i])<2e-5]
            finally:b.matrix_basis=basis;retarget._update(rig)
        if ids:controls.append(dict(name=b.name,pivot=list(rig.matrix_world@b.head),points=ids,children=[],limit=math.radians(limit_degrees)*strength))
    for c in controls:
        b=rig.pose.bones[c['name']];c['children']=[j for j,d in enumerate(controls) if b in rig.pose.bones[d['name']].parent_recursive]
    return controls

def refine(rig,mapping,settings,fit,_allow_shoulders=True):
    report=dict(enabled=True,accepted=False,touched=[],warnings=[],camera_fixed=True)
    if getattr(settings,'contact_enabled',False):
        report['reason']='explicit_contacts_have_priority';return report
    safe=lambda b:b and not retarget._active_rotation_constraints(b) and not retarget._driver_controls_rotation(rig,b)
    names=[mapping.get(r,'') for r in ('hips','spine','chest')]
    if _allow_shoulders:names += ['c_shoulder.l','c_shoulder.r']
    bones=[]
    for name in names:
        b=rig.pose.bones.get(name)
        if safe(b) and b not in bones:bones.append(b)
    bones.sort(key=lambda b:len(b.parent_recursive));joint_bones=[rig.pose.bones[mapping[projection_fit.JOINT_ROLES[n.split('.')[0]]+'.'+n[-1]]] for n in projection_fit.NAMES]
    controls=[];original_points=projection_fit.points(rig,mapping)
    for b in bones:
        ids=[i for i,j in enumerate(joint_bones) if b in j.parent_recursive]
        if not ids:
            # FK arm controls can follow the clavicle through constraints,
            # rather than parenting. Verify their rigid influence by a small
            # reversible probe instead of assuming the hierarchy is complete.
            basis=b.matrix_basis.copy();v=np.array([.004,.003,.005]);world=rig.matrix_world.to_quaternion().to_matrix()
            try:
                delta=world.transposed()@Matrix(torso_math.fit_math.rotation(v).tolist())@world
                retarget._set_rotation(b,retarget._to_local_rotation(b,delta.to_quaternion()@b.matrix.to_quaternion()));retarget._update(rig)
                moved=projection_fit.points(rig,mapping);pivot=np.array(rig.matrix_world@b.head)
                expected=(original_points-pivot)@torso_math.fit_math.rotation(v).T+pivot
                ids=[i for i in range(12) if np.linalg.norm(moved[i]-original_points[i])>1e-6 and np.linalg.norm(moved[i]-expected[i])<2e-5]
            finally:b.matrix_basis=basis;retarget._update(rig)
        if not ids:continue
        controls.append(dict(name=b.name,pivot=list(rig.matrix_world@b.head),points=ids,children=[],limit=math.radians(12)*settings.strength))
    for i,c in enumerate(controls):
        b=rig.pose.bones[c['name']];c['children']=[j for j,d in enumerate(controls) if b in rig.pose.bones[d['name']].parent_recursive]
    if not controls:report['reason']='no_writable_torso_hierarchy';return report
    initial=projection_fit.points(rig,mapping);obs=np.array(fit['observed']);weights=np.array(fit['weights']);camera=fit['camera'];width=fit['image']['width']
    locked=tuple(5 if side=='L' else 11 for side in contact_solver.grounded_sides(settings)) if settings.ground_feet else ()
    proposal=torso_math.solve(initial,obs,weights,camera,controls,locked,steps=min(settings.fit_iterations,40))
    report.update(before_px=proposal['before']*width,proposal_px=proposal['after']*width,controls=[c['name'] for c in controls],reason='no_clear_joint_gain')
    if not proposal['accepted']:return report
    snapshot=retarget.snapshot_pose(rig);ends={mapping[r+'.'+side]:rig.pose.bones[mapping[r+'.'+side]].matrix.to_quaternion().copy() for r in ('hand','foot','head') for side in ('L','R') if r+'.'+side in mapping}
    if mapping.get('head') in rig.pose.bones:ends[mapping['head']]=rig.pose.bones[mapping['head']].matrix.to_quaternion().copy()
    try:
        world=rig.matrix_world.to_quaternion().to_matrix();inv=world.transposed()
        for i,c in enumerate(controls):
            b=rig.pose.bones[c['name']];delta=inv@Matrix(torso_math.fit_math.rotation(proposal['parameters'][3*i:3*i+3]).tolist())@world
            q=delta.to_quaternion()@b.matrix.to_quaternion();retarget._set_rotation(b,retarget._to_local_rotation(b,q));retarget._update(rig)
        for root,mid,end in ((3,4,5),(9,10,11)):
            if end in locked:
                side='L' if end==5 else 'R';a,b,c=[rig.pose.bones[mapping[r+'.'+side]] for r in ('thigh','shin','foot')]
                projection_fit._aim_link(rig,a,b,proposal['points'][mid]-proposal['points'][root]);projection_fit._aim_link(rig,b,c,proposal['points'][end]-proposal['points'][mid])
        for name,q in ends.items():
            b=rig.pose.bones[name];retarget._set_rotation(b,retarget._to_local_rotation(b,q));retarget._update(rig)
        actual=projection_fit.points(rig,mapping);pred=camera_data.project(actual,camera);error=camera_data.error(pred,obs,weights)
        drift=max((np.linalg.norm(actual[i]-initial[i]) for i in locked),default=0.)
        lengths=[abs(np.linalg.norm(actual[j]-actual[i])/max(np.linalg.norm(initial[j]-initial[i]),1e-9)-1) for a,b,c in ((0,1,2),(3,4,5),(6,7,8),(9,10,11)) for i,j in ((a,b),(b,c))]
        deviation=float(np.max(np.linalg.norm(actual-proposal['points'],axis=1)))
        old_errors=np.linalg.norm(camera_data.project(initial,camera)-obs,axis=1);new_errors=np.linalg.norm(pred-obs,axis=1)
        reliable=weights>.75;reliable_guard=bool(np.all(new_errors[reliable]<np.maximum(old_errors[reliable]*1.2,old_errors[reliable]+.006)))
        # Constraint-following clavicles are approximate in the fast geometric
        # model. Judge the real evaluated pose, with the same joint guards.
        tolerance=max(.005,float(np.linalg.norm(np.ptp(initial,axis=0)))*.01)
        valid=error<proposal['before']-max(.0005,proposal['before']*.02) and drift<.002 and max(lengths)<.002 and deviation<tolerance and reliable_guard
        if not valid:
            retarget.restore_pose(rig,snapshot);report.update(reason='evaluated_rig_proposal_rejected',actual_px=error*width,ground_drift=float(drift),geometry_deviation=deviation)
            if _allow_shoulders and any(c['name'].startswith('c_shoulder.') for c in controls):
                fallback=refine(rig,mapping,settings,fit,_allow_shoulders=False);fallback['shoulder_trial']=report;return fallback
            return report
        touched=[c['name'] for c in controls]+list(ends)
        if locked:touched += [mapping[r+'.'+side] for side in ('L','R') for r in ('thigh','shin')]
        report.update(accepted=True,reason='verified_joint_gain',after_px=error*width,ground_drift=float(drift),geometry_deviation=deviation,touched=sorted(set(touched)),parameters_deg=np.degrees(proposal['parameters']).tolist())
        fit.update(points=actual.tolist(),after_projection=pred.tolist(),after_px=error*width,accepted=True,touched=sorted(set(fit['touched']+touched)))
        if 'perspective' in fit and not fit['perspective'].get('fallback'):
            fit['perspective'].update(torso_roots_fixed=False,torso_rotation_refined=True)
        return report
    except Exception:
        retarget.restore_pose(rig,snapshot);raise
