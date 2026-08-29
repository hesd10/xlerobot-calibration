"""A self-contained, interactive 3D view of the calibrated geometry.

The output is one HTML file with no external references: no CDN, no three.js,
no fonts, no network of any kind. A calibration lab may well be offline, and a
result that only renders with a working internet connection is not a result you
can archive. It opens in any modern browser on any OS, and can be copied or
emailed as a single file.

The 3D maths is deliberately small and hand-rolled -- rotate, project,
depth-sort -- because a full engine would mean either a CDN or vendoring a
large dependency, for a scene of a dozen markers and arrows.
"""
from __future__ import annotations

import json
from html import escape as _escape
from typing import Any

from . import robot_overview
from .i18n import text as _text

# Drawn in the body frame, in millimetres. The chassis outline is nominal: it
# is scene furniture that gives the eye something to judge the calibrated
# positions against, and is not itself a calibration output.
CHASSIS_HALF_WIDTH = 155.0
CHASSIS_DEPTH = 120.0
CHASSIS_TOP = 420.0

AXIS_LENGTH = 200.0
CAMERA_RAY = 260.0


def _payload(overview: dict[str, Any]) -> dict[str, Any]:
    """The geometry the page draws, reduced to plain JSON."""
    arms_by_key = {arm["key"]: arm for arm in overview["arms"]}
    items: list[dict[str, Any]] = []

    for arm in overview["arms"]:
        position = arm.get("position_mm")
        if not position:
            continue
        items.append({
            "kind": "arm",
            "label": arm["label"],
            "at": position,
            "axes": [arm["x_axis"], arm["y_axis"], arm["z_axis"]],
        })

    for camera in overview["cameras"]:
        if camera.get("axis_frame") != "body":
            # Only body-frame directions can be drawn in a body-frame scene.
            continue
        if camera.get("position_frame") == "body":
            at = camera["position_mm"]
            anchored = False
        elif camera.get("body_position_mm"):
            # A gripper-frame camera whose body-frame place at the zero pose
            # could be worked out. Drawn where it is, so the ray leaves the
            # camera rather than the shoulder.
            at = camera["body_position_mm"]
            anchored = False
        else:
            # No model to pose the arm with, so there is nowhere honest to put
            # it. The ray is anchored at the arm root and labelled as such
            # rather than drawn somewhere it is not.
            arm_key = ("left_arm" if camera["key"].startswith("left")
                       else "right_arm")
            at = arms_by_key.get(arm_key, {}).get("position_mm")
            if not at:
                continue
            anchored = True
        items.append({
            "kind": "camera",
            "key": camera["key"],
            "label": camera["label"],
            "at": at,
            # What the table prints. Now that a wrist camera is drawn where it
            # actually is, the body-frame place is the useful reading and the
            # marker agrees with it. Only a camera that could not be placed
            # falls back to its gripper-frame offset, and its note says so.
            "reported": (camera.get("body_position_mm")
                         or camera["position_mm"]),
            "axis": camera["optical_axis"],
            "anchored": anchored,
            # A wrist camera placed at the zero pose is drawn truthfully but
            # only for that posture, which is a different caveat from a head
            # camera that is simply bolted to the body.
            "onGripper": camera.get("position_frame") != "body",
        })

    return {
        "items": items,
        "chassis": {
            "halfWidth": CHASSIS_HALF_WIDTH,
            "depth": CHASSIS_DEPTH,
            "top": CHASSIS_TOP,
        },
        "axisLength": AXIS_LENGTH,
        "cameraRay": CAMERA_RAY,
        "frame": overview.get("frame"),
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__T_TITLE__</title><style>
:root{color-scheme:light dark;--bg:#f4f6f8;--panel:#fff;--text:#17202a;
--muted:#5b6672;--line:#d9dee5;--accent:#1769aa}
@media(prefers-color-scheme:dark){:root{--bg:#111417;--panel:#191d21;
--text:#edf1f5;--muted:#aeb7c2;--line:#363d45;--accent:#60a5dc}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 system-ui,sans-serif}
.wrap{max-width:1100px;margin:auto;padding:20px}
h1{font-size:22px;margin:0 0 4px}
.muted{color:var(--muted);font-size:13px}
.stage{margin-top:14px;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:10px}
canvas{width:100%;height:auto;display:block;cursor:grab;touch-action:none}
canvas.dragging{cursor:grabbing}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
button{padding:8px 13px;border:1px solid var(--line);border-radius:6px;
background:var(--panel);color:inherit;cursor:pointer;font:inherit}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}
th,td{padding:7px 8px;border-top:1px solid var(--line);text-align:left;
white-space:nowrap}
th{color:var(--muted);font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:13px}
.chip{display:flex;align-items:center;gap:6px}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block}
/* The mounting decides what every "left" on this page means, so it is stated
   once, prominently, rather than left to be inferred from the drawing. */
.mounting{margin:10px 0 0;padding:9px 12px;border-radius:7px;font-size:13px;
border-left:3px solid #b4804a;background:#2a2018;color:#e8d9c6}
.mounting.normal{border-left-color:#3f8f4a;background:#1c261e;color:#d3e4d6}
</style></head><body><div class="wrap">
<h1>__T_TITLE__</h1>
<div class="muted">__T_HINT____FRAME__</div>
<div class="mounting __MOUNTING_CLASS__">__T_MOUNTING__</div>
<div class="stage"><canvas id="view" width="1040" height="620"></canvas>
<div class="controls">
<button id="spin" class="on">__T_SPIN__</button>
<button data-view="iso">__T_ISO__</button>
<button data-view="front">__T_FRONT__</button>
<button data-view="side">__T_SIDE__</button>
<button data-view="top">__T_TOP__</button>
<button id="reset">__T_RESET__</button>
</div>
<div class="legend" id="legend"></div></div>
<table id="table"><thead><tr><th>__T_OBJECT__</th><th>__T_POSITION__</th>
<th>__T_HEADING__</th><th>__T_NOTE__</th></tr></thead><tbody></tbody></table>
<p class="muted">__T_FOOTER__</p>
</div><script>
const DATA=__DATA__;
const T=__TEXT__;
const COLOURS={head:'#e0a500',left_wrist:'#d94f8a',right_wrist:'#1aa3a3',
arm:'#1769aa'};
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let yaw=-0.9,pitch=0.5,zoom=1,spinning=true,dragging=false,lastX=0,lastY=0;

function rotate(p){
  // Yaw about the vertical body axis, then pitch toward the viewer. Returns
  // camera-space coordinates with +z pointing away from the eye.
  const cy=Math.cos(yaw),sy=Math.sin(yaw);
  const x=p[0]*cy-p[1]*sy, y=p[0]*sy+p[1]*cy, z=p[2];
  const cp=Math.cos(pitch),sp=Math.sin(pitch);
  return [x, y*cp-z*sp, y*sp+z*cp];
}
function project(p){
  const r=rotate([p[0],p[1],p[2]-view.centreZ]);
  const d=2600, f=d/(d+r[1]*0.6);
  return [canvas.width/2+view.offsetX+r[0]*view.scale*f,
          canvas.height/2+view.offsetY-r[2]*view.scale*f, r[1]];
}
// The scale is fitted to the scene every frame instead of being a constant.
// A fixed scale that frames one viewing angle well will overflow at another,
// because the ground plane's projected height changes as the view tilts; this
// is what let the grid clip at the bottom on nearly half of all angles.
let view={scale:0.4,offsetX:0,offsetY:0,centreZ:330};
function fitView(){
  const pts=[];
  const c=DATA.chassis,g=400;
  for(const x of [-g,g])for(const y of [-g,g])pts.push([x,y,0]);
  for(const x of [-c.depth,c.depth])for(const y of [-c.halfWidth,c.halfWidth])
    {pts.push([x,y,0]);pts.push([x,y,c.top]);}
  const L=DATA.axisLength;
  pts.push([L,0,0],[0,L,0],[0,0,L]);
  for(const item of DATA.items){
    pts.push(item.at);
    if(item.kind==='camera'){
      const r=DATA.cameraRay;
      pts.push([item.at[0]+item.axis[0]*r,item.at[1]+item.axis[1]*r,
                item.at[2]+item.axis[2]*r]);
    }else{
      for(const a of item.axes)
        pts.push([item.at[0]+a[0]*120,item.at[1]+a[1]*120,item.at[2]+a[2]*120]);
    }
  }
  // Project with a neutral scale, measure, then solve for the scale that
  // makes the result fit inside the margin.
  view.scale=1;view.offsetX=0;view.offsetY=0;
  let minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;
  for(const p of pts){
    const q=project(p);
    if(q[0]<minX)minX=q[0];if(q[0]>maxX)maxX=q[0];
    if(q[1]<minY)minY=q[1];if(q[1]>maxY)maxY=q[1];
  }
  const cx=canvas.width/2,cy=canvas.height/2;
  // Room for the text that hangs off the markers, which is not in `pts`.
  const padX=110,padY=52;
  const spanX=Math.max(maxX-cx,cx-minX)*2,spanY=Math.max(maxY-cy,cy-minY)*2;
  const fit=Math.min((canvas.width-padX*2)/Math.max(spanX,1),
                     (canvas.height-padY*2)/Math.max(spanY,1));
  view.scale=fit*zoom;
  // Re-measure at the chosen scale and recentre on what is actually drawn.
  view.offsetX=0;view.offsetY=0;
  minX=1e9;maxX=-1e9;minY=1e9;maxY=-1e9;
  for(const p of pts){
    const q=project(p);
    if(q[0]<minX)minX=q[0];if(q[0]>maxX)maxX=q[0];
    if(q[1]<minY)minY=q[1];if(q[1]>maxY)maxY=q[1];
  }
  view.offsetX=cx-(minX+maxX)/2;
  view.offsetY=cy-(minY+maxY)/2;
}
function line(a,b,colour,width){
  const p=project(a),q=project(b);
  ctx.strokeStyle=colour;ctx.lineWidth=width||1.5;
  ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();
}
function arrow(at,dir,length,colour,label){
  const tip=[at[0]+dir[0]*length,at[1]+dir[1]*length,at[2]+dir[2]*length];
  line(at,tip,colour,2.4);
  const p=project(tip),q=project(at);
  const dx=p[0]-q[0],dy=p[1]-q[1],len=Math.hypot(dx,dy)||1;
  const ux=dx/len,uy=dy/len,s=9;
  ctx.fillStyle=colour;ctx.beginPath();
  ctx.moveTo(p[0],p[1]);
  ctx.lineTo(p[0]-ux*s-uy*s*0.5,p[1]-uy*s+ux*s*0.5);
  ctx.lineTo(p[0]-ux*s+uy*s*0.5,p[1]-uy*s-ux*s*0.5);
  ctx.closePath();ctx.fill();
  if(label){ctx.fillStyle=colour;ctx.font='12px system-ui,sans-serif';
    placeLabel(p[0]+6,p[1]-6,label,colour,p[0],p[1]);}
}
function marker(at,colour,label){
  const p=project(at);
  ctx.fillStyle=colour;ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fill();
  if(label){placeLabel(p[0]+8,p[1]+14,label,colour,p[0],p[1]);}
}
// Labels are placed against the boxes already drawn this frame and nudged
// until they no longer collide. Two markers can share a position exactly --
// a wrist camera sits on its arm root -- and at some viewing angles any fixed
// offset would put one label straight on top of another, so the offset has to
// be chosen per frame rather than baked in.
let placed=[];
function placeLabel(x,y,text,colour,anchorX,anchorY){
  ctx.font='bold 12px system-ui,sans-serif';
  const w=ctx.measureText(text).width,h=14;
  const candidates=[[x,y],[x,y+16],[x,y+32],[x,y-20],[x,y-36],
                    [anchorX-w-12,anchorY+14],[anchorX-w-12,anchorY-6],
                    [anchorX-w-12,anchorY+30],[anchorX-w-12,anchorY-22],
                    [x,y+48],[x,y-52],[anchorX-w-12,anchorY+46],
                    [anchorX-w-12,anchorY-38],[x+40,y+64],[x-w/2,y-68]];
  let spot=null,bestCost=Infinity;
  for(const c of candidates){
    const box={x:c[0],y:c[1]-h,w:w,h:h+3};
    let cost=0;
    for(const o of placed){
      const ox=Math.min(box.x+box.w,o.x+o.w)-Math.max(box.x,o.x);
      const oy=Math.min(box.y+box.h,o.y+o.h)-Math.max(box.y,o.y);
      if(ox>0&&oy>0){cost+=ox*oy;}
    }
    // Falling off the canvas is worse than a little crowding.
    if(box.x<2||box.x+box.w>canvas.width-2||
       box.y<2||box.y+box.h>canvas.height-2){cost+=100000;}
    if(cost===0){spot=c;break;}
    if(cost<bestCost){bestCost=cost;spot=c;}
  }
  placed.push({x:spot[0],y:spot[1]-h,w:w,h:h+3});
  if(spot[0]!==x||spot[1]!==y){
    // Tie the displaced text back to the marker it belongs to.
    ctx.strokeStyle=colour;ctx.globalAlpha=0.45;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(anchorX,anchorY);
    ctx.lineTo(spot[0]+(spot[0]<anchorX?w:0),spot[1]-4);ctx.stroke();
    ctx.globalAlpha=1;
  }
  ctx.fillStyle=colour;ctx.fillText(text,spot[0],spot[1]);
}
function grid(){
  // A ground grid at z=0 gives the rotation something to read against.
  ctx.strokeStyle='rgba(130,140,150,0.30)';ctx.lineWidth=1;
  for(let i=-400;i<=400;i+=100){
    const a=project([i,-400,0]),b=project([i,400,0]);
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    const c=project([-400,i,0]),d=project([400,i,0]);
    ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(d[0],d[1]);ctx.stroke();
  }
}
function chassis(){
  const c=DATA.chassis,w=c.halfWidth,d=c.depth,t=c.top;
  const corners=[[-d,-w,0],[d,-w,0],[d,w,0],[-d,w,0]];
  const colour='rgba(130,140,150,0.75)';
  for(let i=0;i<4;i++){
    line(corners[i],corners[(i+1)%4],colour,1.4);
    const up=corners[i].slice();up[2]=t;
    const nextUp=corners[(i+1)%4].slice();nextUp[2]=t;
    line(corners[i],up,colour,1.4);
    line(up,nextUp,colour,1.4);
  }
}
function bodyAxes(){
  const L=DATA.axisLength;
  arrow([0,0,0],[1,0,0],L,'#b4404a','');
  arrow([0,0,0],[0,1,0],L,'#3f8f4a','');
  arrow([0,0,0],[0,0,1],L,'#3a6ea8','');
}
function bodyAxisLabels(){
  // Placed after the data labels so that when space is tight it is the fixed
  // scene furniture that gives way, not a calibrated value.
  const L=DATA.axisLength;
  const ends=[[[1,0,0],'#b4404a',T.axisX],[[0,1,0],'#3f8f4a',T.axisY],
              [[0,0,1],'#3a6ea8',T.axisZ]];
  for(const [dir,colour,text] of ends){
    const p=project([dir[0]*L,dir[1]*L,dir[2]*L]);
    ctx.font='12px system-ui,sans-serif';
    placeLabel(p[0]+6,p[1]-6,text,colour,p[0],p[1]);
  }
}
function draw(){
  const dpr=window.devicePixelRatio||1;
  const cssWidth=canvas.clientWidth||1040;
  if(canvas.width!==Math.round(cssWidth*dpr)){
    canvas.width=Math.round(cssWidth*dpr);
    canvas.height=Math.round(cssWidth*dpr*620/1040);
  }
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  placed=[];
  fitView();
  grid();chassis();bodyAxes();
  // Painter's algorithm: far things first, so near markers sit on top.
  const ordered=DATA.items.slice().sort((a,b)=>project(b.at)[2]-project(a.at)[2]);
  for(const item of ordered){
    if(item.kind==='arm'){
      marker(item.at,COLOURS.arm,item.label);
      // The arm's own axis triad, drawn unlabelled: the body axes above
      // already establish which colour means what, and six more labels here
      // would crowd out the ones that carry information.
      item.axes.forEach((axis,i)=>arrow(item.at,axis,120,
        ['#1769aa','#6aa0c8','#9dc4dd'][i],''));
    }else{
      const colour=COLOURS[item.key]||'#5b6672';
      // No "(direction)" suffix: the marker now sits at the camera itself, and
      // the table's note already says the offset is a gripper-frame one.
      marker(item.at,colour,item.label);
      arrow(item.at,item.axis,DATA.cameraRay,colour,'');
    }
  }
  bodyAxisLabels();
}
function tick(){
  if(spinning&&!dragging){yaw+=0.004;}
  draw();requestAnimationFrame(tick);
}
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;
  lastY=e.clientY;canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;
  yaw+=(e.clientX-lastX)*0.008;
  pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-lastY)*0.006));
  lastX=e.clientX;lastY=e.clientY;});
canvas.addEventListener('pointerup',e=>{dragging=false;
  canvas.classList.remove('dragging');});
canvas.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.max(0.25,Math.min(6,zoom*(e.deltaY<0?1.1:0.9)));},{passive:false});
const spin=document.getElementById('spin');
spin.onclick=()=>{spinning=!spinning;spin.classList.toggle('on',spinning);};
document.getElementById('reset').onclick=()=>{yaw=-0.9;pitch=0.5;zoom=1;};
const views={iso:[-0.9,0.5],front:[-Math.PI/2,0.06],side:[0,0.06],
top:[-Math.PI/2,1.42]};
for(const button of document.querySelectorAll('[data-view]')){
  button.onclick=()=>{const v=views[button.dataset.view];
    yaw=v[0];pitch=v[1];spinning=false;spin.classList.remove('on');};
}
const legend=document.getElementById('legend');
const tbody=document.querySelector('#table tbody');
const seen=new Set();
for(const item of DATA.items){
  const colour=item.kind==='arm'?COLOURS.arm:(COLOURS[item.key]||'#5b6672');
  if(!seen.has(item.label)){
    seen.add(item.label);
    const chip=document.createElement('span');chip.className='chip';
    chip.innerHTML='<span class="dot" style="background:'+colour+
      '"></span>'+item.label;
    legend.appendChild(chip);
  }
  const fixed=v=>v.map(n=>(n>=0?'+':'')+n.toFixed(2)).join('  ');
  const dir=item.kind==='arm'?item.axes[0]:item.axis;
  const row=document.createElement('tr');
  const note=item.kind==='arm'?T.armNote:
    (item.anchored?T.wristNoteAnchored:(item.onGripper?T.wristNote:T.bodyNote));
  // item.at is where the marker is drawn; item.reported is what the position
  // means. They agree everywhere except a camera that could not be placed,
  // which borrows the arm root to draw from but reports its gripper offset.
  const at=item.reported||item.at;
  row.innerHTML='<td>'+item.label+'</td><td>'+fixed(at)+
    '</td><td>'+fixed(dir)+'</td><td>'+note+'</td>';
  tbody.appendChild(row);
}
tick();
</script></body></html>
"""


def page(overview: dict[str, Any]) -> str:
    """Render the interactive 3D page for an already-collected overview."""
    data = _payload(overview)
    frame = data.get("frame")
    note = (_text("ov.note.frameId", value=frame) + " ") if frame else ""
    # json.dumps escapes nothing that can close a script tag except "/" in
    # "</script>", which cannot appear here because the payload is numbers and
    # known labels; guard anyway so a future label change cannot break out.
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    strings = {name: _text(f"view3d.{name}") for name in (
        "axisX", "axisY", "axisZ", "directionOnly",
        "armNote", "wristNote", "wristNoteAnchored", "bodyNote")}
    mounting = overview.get("mounting") or robot_overview.mounting_mod.NORMAL
    is_normal = mounting != robot_overview.mounting_mod.FLIPPED
    page_text = (PAGE_TEMPLATE
                 .replace("__DATA__", encoded)
                 .replace("__TEXT__",
                          json.dumps(strings, ensure_ascii=False).replace("</", "<\\/"))
                 .replace("__MOUNTING_CLASS__", "normal" if is_normal else "flipped")
                 .replace("__T_MOUNTING__", _escape(_text(
                     "view3d.mounting.normal" if is_normal
                     else "view3d.mounting.flipped")))
                 .replace("__FRAME__", _escape(note)))
    for name, key in (("__T_TITLE__", "title"), ("__T_HINT__", "hint"),
                      ("__T_SPIN__", "spin"), ("__T_ISO__", "iso"),
                      ("__T_FRONT__", "front"), ("__T_SIDE__", "side"),
                      ("__T_TOP__", "top"), ("__T_RESET__", "reset"),
                      ("__T_OBJECT__", "object"), ("__T_POSITION__", "position"),
                      ("__T_HEADING__", "heading"), ("__T_NOTE__", "note"),
                      ("__T_FOOTER__", "footer")):
        page_text = page_text.replace(
            name, _escape(_text(f"view3d.{key}")))
    return page_text


def build(workspace) -> str:
    """Collect the geometry and render the standalone page."""
    return page(robot_overview.collect(workspace))
