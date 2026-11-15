"""Explicit per-hand component choice, preserving palm-local finger geometry."""
import copy
import numpy as np
try:from . import hand_geometry as hg,orientation_math as om
except ImportError:import hand_geometry as hg,orientation_math as om

def compose(current,candidate,component):
    if component not in ('ALL','PALM','FINGERS'):raise ValueError('无效手部选择范围')
    if component=='ALL':return copy.deepcopy(candidate)
    if not current or not candidate:raise ValueError('拆分选择需要当前手部与候选手部均有观测')
    palm=candidate if component=='PALM' else current
    fingers=current if component=='PALM' else candidate
    points=np.asarray(fingers.get('world'),float)
    if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('保留或采用的手指需要完整 21 点')
    a=om.proper_rotation(palm['frame']);b=om.proper_rotation(fingers['frame'])
    result=hg.transport(fingers,a@b.T)
    result['frame']=a.tolist();result['alternatives']=copy.deepcopy(palm.get('alternatives',[]))
    result['source']='component_selected_hand3d'
    result['palm_observation_source']=palm.get('palm_observation_source',palm.get('source'))
    result['finger_observation_source']=fingers.get('finger_observation_source',fingers.get('source'))
    result['confidence']=min(float(palm.get('confidence',0)),float(fingers.get('confidence',0)))
    result['confidence_kind']='minimum_component_gate_not_accuracy'
    result['component_selection']=dict(component=component,palm_source=result['palm_observation_source'],finger_source=result['finger_observation_source'],
        method='manual_selection_with_rigid_palm_local_alignment',accuracy_validated=False)
    result['palm_reference_image']=palm.get('palm_reference_image',palm.get('reference_image'))
    # Pixels belong to the original finger observation, not the composed 3D hand.
    result.pop('reprojection_error',None);result.pop('crop_selection',None)
    return result

def select(current,candidate,sides,component,backend):
    result=copy.deepcopy(current);rows={}
    for side in sides:
        if side not in ('L','R'):raise ValueError('无效左右手')
        old=current.get('orientation',{}).get('hands',{}).get(side,{})
        if current.get('manual_hand_edits',{}).get(side) or old.get('source')=='manual_hand_landmarks' or old.get('manual_fingers'):
            raise ValueError('此手包含人工校正，请先撤回该手校正，再选择模型结果')
        info=candidate.get('hand_refinement',{}).get('hands',{}).get(side,{})
        if not info.get('adopted'):raise ValueError(side+' 手在此模型中未通过检查，不采用回退观测')
        hd=candidate.get('orientation',{}).get('hands',{}).get(side)
        if not hd:raise ValueError('候选缺少所选手部')
        selected=compose(old,hd,component)
        result.setdefault('orientation',{}).setdefault('hands',{})[side]=selected
        rows[side]=dict(adopted=True,backend=backend,source=selected.get('source'),component=component,
            reference=selected.get('reference_image',current['image']['path']),has_21_points='world' in selected,
            confidence=selected.get('confidence',0),gesture='NONE',component_selection=selected.get('component_selection'))
    return result,rows
