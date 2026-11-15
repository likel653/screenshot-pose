"""Evidence-based screenshot review rows. No Blender access or accuracy scores."""
import math
import numpy as np

PARTS=(('BODY','身体',''),('HEAD','头部',''),('HAND_L','左手',''),('HAND_R','右手',''),('ARM_L','左臂',''),('ARM_R','右臂',''),('LEG_L','左腿',''),('LEG_R','右腿',''))
LABELS={k:v for k,v,_ in PARTS}
GROUPS={'ARM_L':(11,13,15),'ARM_R':(12,14,16),'LEG_L':(23,25,27,29,31),'LEG_R':(24,26,28,30,32)}

def issues(data,result=None,threshold=.45,foot_threshold=.45):
    rows=[];result=result or {}
    def add(key,part,message,fix='LOCAL',severity='REVIEW'):
        rows.append(dict(id=key,part=part,title=LABELS.get(part,part),message=message,fix=fix,severity=severity))
    if not data:return rows
    coarse=data.get('wholebody',[]);landmarks=data.get('landmarks',[])
    coarse_map={11:5,13:7,15:9,23:11,25:13,27:15,12:6,14:8,16:10,24:12,26:14,28:16}
    for part,ids in GROUPS.items():
        low=[]
        for i in ids:
            if i>=len(landmarks):continue
            p=landmarks[i];value=min(p.get('visibility',1),p.get('presence',1))
            if len(coarse)==133 and i in coarse_map:value=min(value,coarse[coarse_map[i]].get('confidence',1))
            if not math.isfinite(value) or value<threshold:low.append(i)
        if low:add('low_'+part,part,'部分二维关节点证据不足，先核对原图中的关节位置。','IMAGE')
    # Agreement is evidence availability, not a calibrated correctness score.
    if not data.get('edited_indices'):
        candidates=[c.get('world_landmarks',[]) for c in data.get('body_candidates',[]) if not c.get('manual_only')]
        for part,ids in GROUPS.items():
            normals=[]
            for points in candidates:
                if len(points)!=33:continue
                p=np.asarray([[points[i][k] for k in ('x','y','z')] for i in ids[:3]],float)
                a,b=p[1]-p[0],p[2]-p[1];n=np.cross(a,b);size=np.linalg.norm(n)
                if np.isfinite(n).all() and size>np.linalg.norm(a)*np.linalg.norm(b)*.15:normals.append(n/size)
            disagreement=max((float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for i,a in enumerate(normals) for b in normals[i+1:]),default=0.)
            if disagreement>60:add('models_'+part,part,'不同身体模型对此肢体的弯曲平面分歧较大，需看侧面复核深度。','CANDIDATES')
    orient=data.get('orientation',{});head=orient.get('head',{})
    if not head or head.get('confidence',0)<threshold:add('head_missing','HEAD','没有达到质量下限的独立头向，需复核身体头向。')
    heads=[h for h in [head]+orient.get('head_candidates',[]) if h.get('frame') and h.get('confidence',0)>=threshold]
    if len(heads)>1:
        angles=[]
        for h in heads[1:]:
            a=np.asarray(heads[0]['frame'],float);b=np.asarray(h['frame'],float)
            if a.shape==b.shape==(3,3) and np.isfinite(a).all() and np.isfinite(b).all():angles.append(float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))))
        if angles and max(angles)>45:add('head_disagree','HEAD','头向候选分歧较大，请检查转头与歪头方向。')
    for side in ('L','R'):
        part='HAND_'+side;hd=orient.get('hands',{}).get(side,{})
        if not hd or hd.get('confidence',0)<threshold:add('hand_missing_'+side,part,'缺少达到质量下限的掌向，建议放大原图检查。','HAND')
        if not isinstance(hd.get('world'),list) or len(hd['world'])!=21:add('fingers_missing_'+side,part,'没有完整的三维手指 21 点，可局部精修或手工校正。','HAND')
        for section in ('hand_refinement','hand3d_refinement'):
            info=data.get(section,{}).get('hands',{}).get(side,{})
            if section=='hand3d_refinement' and data.get('automatic_hands') and info.get('adopted') is False:
                add('hand_fallback_'+side,part,('GPU 手模型未就绪，保留已有观测。' if info.get('attempted') is False else 'GPU 手部结果未采用，沿用已有观测：'+str(info.get('reason','质量检查未通过'))[:160]),'HAND','INFO' if info.get('attempted') is False else 'REVIEW')
            elif info.get('fallback'):add('hand_fallback_'+side,part,'局部识别失败，当前沿用已有观测，并未获得新的手指结果。','HAND')
            elif info and info.get('adopted') is False:add('hand_reject_'+side,part,'最近局部结果未采用：'+str(info.get('reason','检查未通过'))[:180],'HAND')
        crops=hd.get('crop_selection',{});choices=crops.get('candidates',[]);chosen=crops.get('chosen',-1)
        if isinstance(chosen,int) and 0<=chosen<len(choices):
            reference=choices[chosen].get('reference') or {}
            if reference.get('error',0)>.25:add('hand_2d_disagree_'+side,part,'手指投影与独立二维指节点分歧较大；这两种观测都需要目视复核。','HAND')
        transfer=result.get('hand_transfer',{}).get(side,{})
        if not transfer.get('intentional_manual_or_partial_strength') and (transfer.get('mean_direction_error_deg') or 0)>25:
            add('finger_transfer_'+side,part,'识别手指传到角色后偏差较大，需检查手指控制链和校准。','CALIBRATION')
        fd=orient.get('feet',{}).get(side,{})
        if not fd or fd.get('confidence',0)<foot_threshold:add('foot_missing_'+side,'LEG_'+side,'脚向证据不足；支撑约束不能恢复被遮挡的脚尖方向。','FEET')
    comparison=result.get('depth_selection',{})
    if comparison.get('ambiguous'):add('depth_tie','BODY','身体深度候选接近，自动结果仍有歧义，请通过侧面预览选择。','CANDIDATES')
    whole=result.get('fit',{}).get('whole_body',{})
    if whole and not whole.get('accepted') and whole.get('reason') not in ('no_clear_gain','verified_no_gain'):
        add('whole_reject','BODY','全身协调未采用，当前保留此前姿势；可查看详细结果。','FIT','INFO')
    for section,part,fix in [('contacts','BODY','FIT'),('feet','BODY','FEET'),('fingers','BODY','HAND'),('orientation','HEAD','LOCAL')]:
        for index,message in enumerate(result.get(section,{}).get('warnings',[])):
            text=str(message);chosen=part
            side='L' if text.startswith('L') or '左' in text else 'R' if text.startswith('R') or '右' in text else None
            if side and ('手' in text or section=='fingers'):chosen='HAND_'+side
            elif side and ('脚' in text or '腿' in text):chosen='LEG_'+side
            add('applied_'+section+str(index),chosen,'上次应用提示：'+text,fix)
    unique={r['id']:r for r in rows}
    return list(unique.values())
