"""Per-video observations: shared person ROI and bounded hand candidate continuity."""
import copy
import numpy as np

class Continuity:
    def __init__(self):
        self.box=None;self.thumb=None;self.time=None;self.hands={};self.events=[];self.reason='first_frame';self.last_points=None
    def reset(self,reason):
        self.box=None;self.hands={};self.last_points=None;self.reason=reason
    def begin(self,image,time):
        thumb=np.asarray(image.resize((32,32)),dtype=float)/255.
        dt=time-self.time if self.time is not None else 0.
        cut=self.thumb is not None and float(np.mean(np.abs(thumb-self.thumb)))>.32
        self.reason='continuous'
        if self.time is None:self.reset('first_frame')
        elif dt<=0 or dt>.5:self.reset('time_gap')
        elif cut:self.reset('scene_cut')
        self.dt=dt;self.thumb=thumb;self.time=time;self.frame_events=[]
    def missing(self):self.reset('missing_person')
    def select(self,boxes,size):
        boxes=[np.asarray(b[:4],float) for b in boxes]
        boxes=[b for b in boxes if np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]]
        if not boxes:self.missing();raise ValueError('未检测到人物')
        # A prior box is not proof of identity when a second person appears.
        if len(boxes)>1:raise ValueError('连续跟踪检测到多个人物，请框选单人区域，避免身份切换')
        raw=boxes[0];box=raw.copy()
        if self.box is not None:
            old=self.box;span=max(old[2]-old[0],old[3]-old[1],1.)
            motion=np.linalg.norm((raw[:2]+raw[2:]-old[:2]-old[2:])*.5)/span
            ratio=np.max((raw[2:]-raw[:2])/(old[2:]-old[:2]))
            if motion>.55 or not .55<ratio<1.8:self.reset('roi_jump')
            else:
                # Only damp small box noise; retain current extent so fast limbs
                # cannot be cropped out by the previous frame's ROI.
                alpha=.85 if motion>.08 else .55
                box=old*(1-alpha)+raw*alpha
                box[:2]=np.minimum(box[:2],raw[:2]);box[2:]=np.maximum(box[2:],raw[2:])
        box[[0,2]]=np.clip(box[[0,2]],0,size[0]);box[[1,3]]=np.clip(box[[1,3]],0,size[1])
        self.box=box.copy();self.raw_box=raw.tolist();return box.tolist()
    def choose_hand(self,side,candidates):
        best=min(candidates,key=lambda c:c['reprojection_error']);old=self.hands.get(side)
        if old is None:return best
        def distance(c):
            r=np.asarray(c['frame']);prior=np.asarray(old['frame']);return float(np.arccos(np.clip((np.trace(prior.T@r)-1)/2,-1,1)))
        # Compare only near-equivalent image fits; do not suppress well-supported
        # turns, or synthesize an orientation absent from current observations.
        eligible=[c for c in candidates if c['reprojection_error']<=best['reprojection_error']+.015]
        selected=min(eligible,key=lambda c:c['reprojection_error']+.012*distance(c)/np.pi)
        if selected is not best:self.frame_events.append(side+'_hand_candidate_continuity')
        return selected
    def commit(self,data):
        points=data.get('landmarks',[])
        if self.last_points and len(points)==len(self.last_points):
            for left,right,label in ((15,16,'wrists'),(27,28,'ankles')):
                ids=(left,right)
                if min(points[i].get('visibility',0) for i in ids)<.5:continue
                a=np.array([[points[i]['x'],points[i]['y']] for i in ids]);b=np.array([[self.last_points[i]['x'],self.last_points[i]['y']] for i in ids])
                direct=np.linalg.norm(a-b,axis=1).sum();cross=np.linalg.norm(a-b[::-1],axis=1).sum()
                if direct>.18 and cross<direct*.35:self.frame_events.append(label+'_side_ambiguity_review')
        self.last_points=copy.deepcopy(points)
        self.hands={s:copy.deepcopy(h) for s,h in data.get('orientation',{}).get('hands',{}).items() if h.get('confidence',0)>=.45}
        report=dict(reset=self.reason,detector_box=getattr(self,'raw_box',None),box=self.box.tolist() if self.box is not None else None,events=list(self.frame_events),identity_policy='single_person_only')
        data['video_tracking']=report;self.events.append(dict(time=self.time,**report));return report
