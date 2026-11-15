"""Timestamp-aware bounded Hermite rotation/position resampling; no bpy."""
import bisect,math
from . import video_stability as q,video_temporal as t

def slope(a,b,ha,hb):
    if a*b<=0:return 0.
    return (3*(ha+hb))/((2*hb+ha)/a+(hb+2*ha)/b)

def hermite(a,b,m0,m1,u,h):
    value=(2*u**3-3*u*u+1)*a+(u**3-2*u*u+u)*h*m0+(-2*u**3+3*u*u)*b+(u**3-u*u)*h*m1
    return min(max(a,b),max(min(a,b),value))

class Resampler:
    def __init__(self,times,raw,sample,mode='HERMITE',cuts=(),max_gap=.3):
        if mode not in ('HERMITE','SLERP'):raise ValueError('未知动作重采样方式')
        if len(times)<2 or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('采样时间必须递增')
        self.times=times;self.raw=raw;self.sample=sample;self.mode=mode;self.cuts=set(cuts);self.max_gap=max_gap
        self.cache={}
    def bracket(self,stamp):
        left=max(0,bisect.bisect_right(self.times,stamp+1e-8)-1);right=min(left+1,len(self.times)-1)
        if stamp<=self.times[0]:return 0,0,0.
        if left==right:return left,right,0.
        h=self.times[right]-self.times[left]
        if right in self.cuts or h>self.max_gap:return left,left,0.
        return left,right,min(1.,max(0.,(stamp-self.times[left])/h))
    def pose(self,index,name):
        key=index,name
        if key not in self.cache:self.cache[key]=self.sample(index,name)
        return self.cache[key]
    def at(self,stamp,name):
        a,b,u=self.bracket(stamp);qa,pa=self.pose(a,name)
        if a==b or u<1e-8:return list(qa),list(pa)
        qb,pb=self.pose(b,name);rotation=q.slerp(qa,qb,u);position=[x+(y-x)*u for x,y in zip(pa,pb)]
        if self.mode=='SLERP' or q.angle(qa,qb)>120:return rotation,position
        h=self.times[b]-self.times[a];prev=a-1;after=b+1
        use_prev=prev>=0 and a not in self.cuts and self.times[a]-self.times[prev]<=self.max_gap
        use_after=after<len(self.times) and after not in self.cuts and self.times[after]-self.times[b]<=self.max_gap
        qp,pp=self.pose(prev,name) if use_prev else (qa,pa);qn,pn=self.pose(after,name) if use_after else (qb,pb)
        hp=self.times[a]-self.times[prev] if use_prev else h;hn=self.times[after]-self.times[b] if use_after else h
        for k in range(3):
            sec=(pb[k]-pa[k])/h
            m0=slope((pa[k]-pp[k])/hp,sec,hp,h) if use_prev else sec
            m1=slope(sec,(pn[k]-pb[k])/hn,h,hn) if use_after else sec
            position[k]=hermite(pa[k],pb[k],m0,m1,u,h)
        # Use one local logarithmic chart, with componentwise monotone tangents.
        d=t.tangent(qa,qb);dp=t.tangent(qa,qp);dn=t.tangent(qa,qn);value=[]
        for k in range(3):
            sec=d[k]/h
            m0=slope(-dp[k]/hp,sec,hp,h) if use_prev else sec
            m1=slope(sec,(dn[k]-d[k])/hn,h,hn) if use_after else sec
            value.append(hermite(0.,d[k],m0,m1,u,h))
        return t.from_tangent(qa,value),position
