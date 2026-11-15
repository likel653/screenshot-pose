"""Local, offline source/target skeleton inspection. No remote assets."""
import html,json

VIEWER = r'''
function viewer(canvas, sets){
 let yaw=0,pitch=0,down=null;
 function render(){const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.fillStyle='#111c2d';c.fillRect(0,0,w,h);
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  for(const set of sets){let pts=set.points.map(p=>{let x=p[0]*cy-p[1]*sy,y=p[0]*sy+p[1]*cy,z=p[2]*cp-y*sp;return [w/2+x*w*.34,h*.55-z*h*.34]});
   c.strokeStyle=set.color;c.fillStyle=set.color;c.lineWidth=3;
   for(const [a,b] of set.edges){if(!pts[a]||!pts[b])continue;c.beginPath();c.moveTo(...pts[a]);c.lineTo(...pts[b]);c.stroke()}
   for(const p of pts){c.beginPath();c.arc(...p,3,0,Math.PI*2);c.fill()}
  }
 }
 canvas.onpointerdown=e=>{down=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
 canvas.onpointermove=e=>{if(!down)return;yaw+=(e.clientX-down[0])*.012;pitch+=(e.clientY-down[1])*.012;down=[e.clientX,e.clientY];render()};
 canvas.onpointerup=()=>down=null;canvas.ondblclick=()=>{yaw=0;pitch=0;render()};render();
}
function handSets(hand){let source=hand.source_points.map(p=>p.slice()),target=[[0,0,0]],edges=[];
 for(let i=0;i<5;i++){let f=['thumb','index','middle','ring','pinky'][i],p=hand.fingers[f]?.target_points;
  if(!p)return [];target.push(...p);let start=1+i*4;edges.push([0,start],[start,start+1],[start+1,start+2],[start+2,start+3])}
 function norm(p){let root=p[0],width=Math.hypot(...p[5].map((x,i)=>x-p[17][i]));return p.map(v=>v.map((x,i)=>(x-root[i])/width))}
 return [{points:norm(source),edges,color:'#6bd7ff'},{points:norm(target),edges,color:'#ffb46b'}];
}
function bodySets(body){let keys=Object.keys(body.source),index=Object.fromEntries(keys.map((k,i)=>[k,i]));
 let edges=Object.entries(body.parents).filter(([k,v])=>k in index&&v in index).map(([k,v])=>[index[k],index[v]]);
 function norm(data){let p=keys.map(k=>data[k]),root=data.hips,span=Math.max(...p.map(v=>v[2]))-Math.min(...p.map(v=>v[2]));return p.map(v=>v.map((x,i)=>(x-root[i])/Math.max(span,.001)))}
 return [{points:norm(body.source),edges,color:'#6bd7ff'},{points:norm(body.target),edges,color:'#ffb46b'}];
}
'''

def page(payload, image='source.png'):
    serialized=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>截图识别与角色转移诊断</title><style>body{background:#0c1422;color:#e5edf8;font:16px/1.75 'Microsoft YaHei',sans-serif;margin:auto;max-width:1250px;padding:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#19283c;padding:18px;border-radius:14px}img,canvas{width:100%;height:auto}button{background:#294967;color:white;padding:8px 18px;border:0;border-radius:8px;margin-right:12px;font:inherit}pre{white-space:pre-wrap;word-break:break-word}a{color:#7fddf0}@media(max-width:700px){main{grid-template-columns:1fr}}</style>
<h1>原图 → 源姿势 → 角色</h1><p>蓝色：识别预测；橙色：角色实际骨骼。拖动骨架可旋转，双击复位。数值衡量转移保真，不是原图识别准确率。</p>
<main><div class="card"><h2>输入截图</h2><img src="'''+html.escape(image,quote=True)+'''"></div><div class="card"><h2>身体主要关节</h2><canvas id="body" width="650" height="650"></canvas><p>按各自身体高度归一化，角色转入匹配相机坐标。这里仅显示主要关节；处理数据仍保留完整源骨架。</p><pre id="choice"></pre></div>
<div class="card"><h2>手指：预测与角色</h2><button data-side="L">人物左手</button><button data-side="R">人物右手</button><canvas id="hand" width="650" height="650"></canvas><p>按各自掌宽归一化并移除手腕整体朝向。位置差还包括角色手掌比例差异；重叠好也不能证明预测吻合原图。</p></div>
<div class="card"><h2>转移检查</h2><pre id="metrics"></pre><p>蓝色源手本身与原图不符：优先检查裁剪、遮挡和模型。源手合理而橙色不同：检查控制链、强度、校准或人工预设。没有有效识别时沿用的数据会保留原来源。</p><a href="diagnosis.json">下载完整诊断数据</a></div></main>
<script>const data='''+serialized+''';'''+VIEWER+'''
if(data.body)viewer(document.getElementById('body'),bodySets(data.body));
document.getElementById('choice').textContent=JSON.stringify(data.comparison||{},null,2);
function show(side){let hand=data.hands[side],canvas=document.getElementById('hand');viewer(canvas,hand?handSets(hand):[]);document.getElementById('metrics').textContent=hand?JSON.stringify({side,source:hand.source,mean_direction_error_deg:hand.mean_direction_error_deg,max_direction_error_deg:hand.max_direction_error_deg,intentional_manual_or_partial_strength:hand.intentional_manual_or_partial_strength,fingers:Object.fromEntries(Object.entries(hand.fingers).map(([k,v])=>[k,{source_flex_deg:v.source_flex_deg,target_flex_deg:v.target_flex_deg,direction_errors_deg:v.direction_errors_deg}]))},null,2):'该手没有完整的三维观测或角色映射。'}
document.querySelectorAll('[data-side]').forEach(b=>b.onclick=()=>show(b.dataset.side));show('R');</script>'''
