"""Convert explicitly named Anny bones to the existing pose interchange."""
import copy
import numpy as np
try:
    from . import pose_data, orientation_math as om
except ImportError:
    import pose_data, orientation_math as om

BODY = {11:'upperarm01.L',12:'upperarm01.R',13:'lowerarm01.L',14:'lowerarm01.R',
        15:'wrist.L',16:'wrist.R',23:'upperleg01.L',24:'upperleg01.R',
        25:'lowerleg01.L',26:'lowerleg01.R',27:'foot.L',28:'foot.R',
        31:'toe1-1.L',32:'toe1-1.R'}

def convert(raw, reference):
    """Keep independent image observations; add a named 3D candidate and frames.

    Model detection scores are not calibrated per-joint confidence. Orientation
    confidence is an explicit heuristic gated by independent 2D agreement.
    """
    data=copy.deepcopy(pose_data.validate(reference))
    labels=raw['bone_labels']; index={n:i for i,n in enumerate(labels)}
    posed=raw['posed']; neutral=raw['neutral']
    p=np.asarray(posed['bone_heads'],float)[0]
    tails=np.asarray(posed['bone_tails'],float)[0]
    bones=np.asarray(posed['bone_poses'],float)[0]
    rest=np.asarray(neutral['bone_poses'],float)[0]
    j=np.asarray(raw['j3d'],float)
    if p.shape!=(len(labels),3) or not all(np.isfinite(v).all() for v in (p,tails,bones,rest,j)):
        raise ValueError('Anny bone data is incomplete or non-finite')
    for name in list(BODY.values())+['head']:
        if name not in index:raise ValueError('Missing Anny bone: '+name)
    w,h=raw['image_size']
    if (w,h)!=(data['image']['width'],data['image']['height']):
        raise ValueError('Reference and Anny image sizes differ')
    K=np.asarray(raw['K'],float)
    # Convert network-square pixels back through the exact contain/pad transform.
    xy=np.asarray(raw['j2d'],float)-np.asarray(raw['pad'])
    xy*=np.array([w,h])/np.asarray(raw['resized_size'])
    worlds=copy.deepcopy(data['world_landmarks'])
    root=(p[index['upperleg01.L']]+p[index['upperleg01.R']])/2
    def set_point(i,point):
        v=point-root
        worlds[i].update(x=float(v[0]),y=float(v[1]),z=float(v[2]))
    for i,name in BODY.items():set_point(i,p[index[name]])
    for side,ids in [('L',(17,19,21)),('R',(18,20,22))]:
        for legacy,finger in zip(ids,(5,2,1)):
            set_point(legacy,tails[index[f'finger{finger}-3.{side}']])
    for i in range(11):set_point(i,p[index['head']])
    for side,i in [('L',29),('R',30)]:
        set_point(i,p[index['foot.'+side]])  # ankle proxy; no invented heel observation
    candidates=copy.deepcopy(data.get('body_candidates',[]))
    if not candidates:candidates=[dict(source=data.get('depth_source','previous'),world_landmarks=copy.deepcopy(data['world_landmarks']))]
    candidates=[c for c in candidates if c['source']!='multihmr_anny_depth']
    candidates.append(dict(source='multihmr_anny_depth',world_landmarks=worlds))
    details=copy.deepcopy(data.get('orientation',dict(schema='screenshot_pose/orientation_v1',camera_axes='X_right_Y_away_Z_up',hands={},feet={},warnings=[])))
    details.setdefault('hands',{});details.setdefault('feet',{});details.setdefault('warnings',[])
    def error(i,name):
        ref=data['landmarks'][i]
        return float(np.linalg.norm(xy[index[name]]-np.array([ref['x']*w,ref['y']*h])))
    errors={str(i):error(i,name) for i,name in BODY.items()}
    original_warnings=list(details['warnings'])
    accepted={}
    for side,wrist in [('L',15),('R',16)]:
        hand=[p[index['wrist.'+side]]]
        for finger in range(1,6):
            hand.extend(p[index[f'finger{finger}-{k}.{side}']] for k in (1,2,3))
            hand.append(tails[index[f'finger{finger}-3.{side}']])
        hand=np.asarray(hand)@om.C_DOWN.T
        palm_pixels=max(np.linalg.norm(xy[index[f'finger2-1.{side}']]-xy[index['wrist.'+side]]),10.)
        err=errors[str(wrist)]
        # A bad prediction must not grant itself a larger tolerance merely by
        # producing an oversized palm. Keep a fixed image-relative upper bound.
        reliable=bool(err<min(.035*max(w,h),max(.015*max(w,h),.75*palm_pixels)))
        accepted[side]=dict(wrist_error_px=err,palm_pixels=float(palm_pixels),accepted=reliable)
        if reliable:
            details['warnings']=[n for n in details['warnings'] if not n.startswith(side+'手')]
            details['hands'][side]=dict(frame=om.palm_frame(hand).tolist(),world=hand.tolist(),curls=om.curls(hand),
                confidence=.65,confidence_method='heuristic_2d_agreement_not_model_probability',source='multihmr_anny_hand',anatomical_side=side)
        else:details['warnings'].append(side+'手 Anny 投影与独立识别不一致，保留原手部结果')
    # Correct for Anny's actual rest bone roll before converting camera axes.
    head=index['head']; anatomical=np.column_stack(([1,0,0],[0,0,1],[0,-1,0]))
    frame=om.C_DOWN@bones[head,:3,:3]@rest[head,:3,:3].T@anatomical
    details['head']=dict(frame=om.proper_rotation(frame).tolist(),confidence=.65,
        confidence_method='uncalibrated_model_candidate',source='multihmr_anny_head')
    for side,ankle in [('L',27),('R',28)]:
        big=p[index['toe1-1.'+side]]; small=p[index['toe5-1.'+side]];heel=p[index['foot.'+side]]
        frame=om.frame(om.C_DOWN@((big-small)*(1 if side=='L' else -1)),om.C_DOWN@((big+small)*.5-heel))
        if errors[str(ankle)]<.06*max(w,h):
            details['feet'][side]=dict(frame=frame.tolist(),confidence=.6,source='multihmr_anny_foot',
                confidence_method='heuristic_2d_agreement_not_model_probability')
    rest_heads=np.asarray(neutral['bone_heads'],float)[0]
    rest_tails=np.asarray(neutral['bone_tails'],float)[0]
    skeleton={'schema':'screenshot_pose/skeleton_v1','pose_axes':'X_right_Y_away_Z_up',
              'rest_axes':'source_native','bones':{name:{'parent':labels[raw['bone_parents'][i]] if raw.get('bone_parents') and raw['bone_parents'][i]>=0 else None,'pose_rotation':(om.C_DOWN@bones[i,:3,:3]).tolist(),
                  'rest_rotation':rest[i,:3,:3].tolist(),'rest_head':rest_heads[i].tolist(),
                  'rest_tail':rest_tails[i].tolist(),'pose_head':(om.C_DOWN@p[i]).tolist(),
                  'pose_tail':(om.C_DOWN@tails[i]).tolist()} for i,name in enumerate(labels)}}
    # Preserve all 163 rigid frames, including finger axial rotation, rather than
    # reducing the model output to the legacy 33 landmarks.
    for side,quality in accepted.items():
        if quality['accepted']:
            details['hands'][side]['bone_frames']={f'{finger}:{segment}':skeleton['bones'][f'finger{fi}-{segment}.{side}']
                for fi,finger in enumerate(('thumb','index','middle','ring','pinky'),1) for segment in (1,2,3)}
    data.update(source='multihmr_anny',body_candidates=candidates,orientation=details,full_skeleton=skeleton,suggested_mode='HYBRID',
        multihmr=dict(runtime=raw['runtime'],body_reprojection_errors_px=errors,hand_quality=accepted,reference_orientation_warnings=original_warnings,
                      license='NAVER Non-Commercial; see official Checkpoint_License_Anny.txt',camera_intrinsics=K.tolist()))
    # Default depth remains the reference; Blender compares candidates using the
    # same observations and the target rig, rather than claiming a larger model wins.
    return pose_data.validate(data)
