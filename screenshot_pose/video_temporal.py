"""Offline, centered motion filtering. Never change source pose-cache records."""
import math
from . import video_stability as rotations

def multiply(a,b):
    w,x,y,z=a;v,i,j,k=b
    return [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v]

def tangent(center,q):
    a=rotations.unit(center);b=rotations.unit(q)
    if sum(x*y for x,y in zip(a,b))<0:b=[-v for v in b]
    d=multiply([a[0],-a[1],-a[2],-a[3]],b)
    norm=math.sqrt(sum(v*v for v in d[1:]))
    factor=2*math.atan2(norm,max(0.,d[0]))/norm if norm>1e-10 else 2.
    return [v*factor for v in d[1:]]

def from_tangent(center,v):
    norm=math.sqrt(sum(x*x for x in v))
    factor=math.sin(norm/2)/norm if norm>1e-10 else .5
    return rotations.unit(multiply(rotations.unit(center),[math.cos(norm/2),*[x*factor for x in v]]))

def intercept(samples):
    """Weighted local linear fit evaluated at the current timestamp, not its mean."""
    total=sum(w for t,v,w in samples);sx=sum(w*t for t,v,w in samples);sxx=sum(w*t*t for t,v,w in samples)
    denom=total*sxx-sx*sx
    if denom<1e-12:return [sum(w*v[k] for t,v,w in samples)/total for k in range(3)]
    return [(sxx*sum(w*v[k] for t,v,w in samples)-sx*sum(w*t*v[k] for t,v,w in samples))/denom for k in range(3)]

class Smoother:
    def __init__(self,times,raw,rotation,window=.12,mode='ADAPTIVE',protected=()):
        if mode not in ('ADAPTIVE','LEGACY','OFF'):raise ValueError('视频连续性模式无效')
        if not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('连续性处理时间戳必须递增')
        self.times=times;self.raw=raw;self.rotation=rotation;self.window=window;self.mode=mode;self.protected=set(protected)
        self.neighbors={}
    def neighborhood(self,index):
        if index not in self.neighbors:
            stamp=self.times[index];left=index;right=index
            while left>0 and stamp-self.times[left-1]<=self.window:left-=1
            while right+1<len(self.times) and self.times[right+1]-stamp<=self.window:right+=1
            self.neighbors[index]=[(j,max(.001,1-abs(self.times[j]-stamp)/(self.window+1e-6))) for j in range(left,right+1)]
        return self.neighbors[index]
    def sample(self,index,name):
        center=self.raw(index)[name];q=self.rotation(index,name);location=list(center['location'])
        if self.window<=0 or self.mode=='OFF' or index in (0,len(self.times)-1) or (index,name) in self.protected:return list(q),location
        neighbors=[(j,w) for j,w in self.neighborhood(index) if (j,name) not in self.protected]
        if self.mode=='LEGACY':
            # Original arithmetic quaternion/position average for comparison.
            from .video_data import smooth_quaternion
            result=smooth_quaternion(q,[(self.rotation(j,name),w) for j,w in neighbors])
            total=sum(w for j,w in neighbors)
            return result,[sum(self.raw(j)[name]['location'][k]*w for j,w in neighbors)/total for k in range(3)]
        times=self.times;span=times[index+1]-times[index-1]
        # A whole-frame detection gap is not evidence for a continuous motion.
        if max(times[index]-times[index-1],times[index+1]-times[index])>.3:return list(q),location
        a=self.rotation(index-1,name);b=self.rotation(index+1,name)
        speed=rotations.angle(a,b)/span
        bridge=rotations.slerp(a,b,(times[index]-times[index-1])/span)
        bend=rotations.angle(q,bridge)
        # Fast motion and well-defined turning points receive less filtering.
        amount=1/(1+(speed/60.)**2+(bend/6.)**4)
        backward=tangent(q,a);forward=tangent(q,b)
        velocity=[(r-l)/span for l,r in zip(backward,forward)]
        angular=[];position=[]
        for j,weight in neighbors:
            record=self.raw(j)[name];confidence=max(.05,min(1.,float(record.get('confidence',0.))))
            # Timestamp-cell weights prevent densely sampled one-sided data
            # from shifting the fitted pose toward earlier or later frames.
            cell=(times[min(j+1,len(times)-1)]-times[max(0,j-1)])/2
            w=weight*confidence*confidence*max(cell,1e-6)
            t=(times[j]-times[index])/self.window
            observed=tangent(q,self.rotation(j,name))
            predicted=[v*(times[j]-times[index]) for v in velocity]
            residual=math.degrees(math.sqrt(sum((x-y)**2 for x,y in zip(observed,predicted))))
            # Do not blend a future/previous fast gesture into a locally quiet
            # pose simply because it falls inside the time window.
            compatibility=1/(1+(residual/6.)**4)
            angular.append((t,observed,w*compatibility))
            position.append((t,record['location'],w))
        delta=intercept(angular);fitted=intercept(position)
        # Also protect genuine position-only turning points (e.g. root motion).
        fraction=(times[index]-times[index-1])/span
        prior=self.raw(index-1)[name]['location'];after=self.raw(index+1)[name]['location']
        departure=math.sqrt(sum((location[k]-(prior[k]*(1-fraction)+after[k]*fraction))**2 for k in range(3)))
        distance=math.sqrt(sum((after[k]-prior[k])**2 for k in range(3)))
        position_amount=amount/(1+(departure/max(.001,distance*.25))**4)
        return from_tangent(q,[v*amount for v in delta]),[a+(b-a)*position_amount for a,b in zip(location,fitted)]
