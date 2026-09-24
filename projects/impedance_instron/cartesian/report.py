# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the single-leg experiment using recorded motion and force comparisons."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _spring_payload(springs: dict) -> dict:
    """Pack view snapshots as little-endian float32 blocks for offline playback."""
    if not springs.get("available"):
        return springs
    result = dict(springs)
    for key in ("bottom_m", "top_m", "compression_m"):
        array = np.asarray(result[key], dtype="<f4")
        result[key] = {"shape": array.shape, "f32le": base64.b64encode(array.tobytes()).decode("ascii")}
    return result


def write_report(directory, reference, trace, summary, *, profile=None, include_springs: bool = True):
    """Write a saved one-leg replay without inferred-output fitting panels.

    Args:
        directory: Report destination.
        reference: Saved recorded inputs.
        trace: Saved simulated states and contact outputs.
        summary: Saved numerical results and provenance.
        profile: Fixed body and controller settings.
        include_springs: Load or reconstruct audited spring history. False renders
            meshes only and never advances contact, including for GPU results.
    """
    from .rendering import load_geometry  # noqa: PLC0415
    from .springs import load_springs  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Match the saved JSON identity even when failed diagnostics contain infinity.
    summary = _plain(summary)
    run = summary.get("run", summary)
    complete = run.get("status") == "completed" and not run.get("failure")
    title = "Single-leg rollout completed" if complete else "Single-leg rollout stopped"
    geometry = load_geometry(summary.get("shoe", {}))
    springs = (
        load_springs(directory, reference, trace, summary, profile or {})
        if include_springs
        else {"available": False, "reason": "Mesh-only report requested; no contact replay or leg simulation was run."}
    )
    payload = _plain(
        {
            "reference": reference,
            "trace": trace,
            "summary": summary,
            "profile": profile or {},
            "geometry": geometry,
            "springs": _spring_payload(springs),
            "rendering_info": {
                "foot_representation": "actual rigid fullfoot_last triangles, no ankle-to-marker foot stick",
                "physics_changed": False,
                "midsole_representation": "selectable undeformed CAD mesh, solved spring endpoints, or both",
                "spring_representation": "verified contact-history replay of saved simulated states, not a new leg rollout",
            },
        }
    )
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    metrics = summary.get("metrics", {})
    cards = []
    for key, value in metrics.items():
        display_key, display_value = key, value
        if key.endswith("_rad"):
            display_key = key.removesuffix("_rad") + "_deg"
            display_value = np.rad2deg(np.asarray(value)).tolist()
        cards.append(f"{display_key}: {display_value}")
    if "foot_ground_target_rad" in reference:
        cards.insert(0, "Angle metric channels: knee relative angle, shoe pitch to ground [deg]")
    note = "No trunk, second leg, hip torque, or upper-body weight. Hip Cartesian spring + knee/ankle springs; one shoe supplies GRF."
    metadata = json.loads(str(reference.get("metadata_json", "{}")))
    side = metadata.get("side", "unspecified")
    note = f"{side.capitalize()} leg. " + note
    angle_conv = metadata.get("angle_convention", {})
    if angle_conv.get("virtual_foot_reference") in {"ground", "reconstructed_ground"}:
        note += " Foot ground pitch is reconstructed from Visual3D 3D Cardan relative angle (RVirtualFootAngle relative to RSK) and reconstructed shank frame RSK, transporting the calibrated forward axis onto the simulation sagittal plane. Positive pitch is toe-up. Shank inclination, relative virtual-foot angle, and shoe pitch to ground are reported separately in degrees."
    elif "foot_ground_target_rad" in reference:
        note += " Virtual-foot X is interpreted as ground pitch by explicit declaration; positive is toe-up. The supplied FullBuild script still defines a shank-relative angle, so the ground interpretation remains provisional."
    document = _PAGE.replace("__DATA__", encoded).replace("__TITLE__", html.escape(title))
    document = document.replace("__NOTE__", html.escape(note)).replace("__METRICS__", html.escape("\n".join(cards)))
    document = document.replace("__SUMMARY__", html.escape(json.dumps(_plain(summary), indent=2, allow_nan=False)))
    path = directory / "report.html"
    path.write_text(document)
    return path


_PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cartesian hip — single-leg stance</title><style>
body{font:15px system-ui,sans-serif;background:#f5f7f8;color:#223641;margin:0}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:18px}.note,.card{background:white;border:1px solid #d7e0e4;border-radius:8px;padding:16px;margin:12px 0}.note{border-left:5px solid #bd8630}canvas{width:100%;height:430px;background:#fff;border:1px solid #d7e0e4}.replay-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(0,1fr);gap:16px}.foot-card{min-width:0;background:white;padding:12px;border:1px solid #d7e0e4}.foot-card h2{margin:0 0 8px}#foot_view{height:310px}@media(max-width:800px){.replay-grid{grid-template-columns:1fr}}input[type=range]{width:65%}#spring_slice{width:110px;vertical-align:middle}select{padding:6px;margin:4px 12px 4px 3px}#spring_controls label{white-space:nowrap;font-size:13px}.color-ramp{height:12px;max-width:360px;background:linear-gradient(to right,#0000ff 0%,#00ffff 33.333%,#ffff00 66.667%,#ff0000 100%);border:1px solid #a2b1b9;margin-top:6px}.ramp-ticks{display:flex;justify-content:space-between;max-width:360px;font-size:11px}#compression_title{font-size:12px}#spring_map{height:245px;border:0}button{padding:7px 15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}.card svg{width:100%;height:auto}svg text{font:11px system-ui;fill:#536774}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px ui-monospace,monospace}.legend{font-size:12px;color:#526975}details{margin-top:18px}</style>
<main><h1>Cartesian hip + knee/ankle impedance</h1><div class="note"><b>__TITLE__ — experimental, not a validated running fit.</b><p>__NOTE__</p></div>
<div class="card"><label>Shoe view <select id="shoe_view"><option value="mesh">Mesh</option><option value="springs">Springs + deformation heat map</option><option value="both" selected>Both</option></select></label>
<span id="spring_controls"> <label>Color <select id="spring_metric"><option value="mm">Compression [mm]</option><option value="strain">Compression / rest length [%]</option></select></label> <label>Side-view width slice <input id="spring_slice" type="range" min="0" max="1" value="0"> <span id="slice_label"></span></label> <label><input id="all_springs" type="checkbox"> All columns (side-view overlap)</label></span>
<p class="legend" id="spring_status"></p><div id="compression_legend"><span id="compression_title"></span><div class="color-ramp"></div><div class="ramp-ticks" id="compression_ticks"></div></div></div>
<div class="replay-grid"><div><canvas id="view"></canvas></div><div class="foot-card"><h2>Ankle-last attachment</h2><canvas id="foot_view"></canvas><p class="legend">The shank connects at the ankle hinge. The short connector shows the existing rigid offset to the last; it is not a new joint or force path.</p></div></div>
<p><button id="play">Play</button> <input id="slider" type="range" min="0" max="1000" value="0"> <span id="clock"></span></p>
<p><label><input id="show_reference" type="checkbox" checked> Recorded-reference leg</label> <label><input id="show_markers" type="checkbox"> Recorded foot markers</label></p>
<p class="legend">Blue: shank/thigh · gold: rigid Instron last (foot geometry) · green: undeformed midsole · gray: reference · purple cross/arrow: hip equilibrium/applied force · orange: resultant GRF. No marker-endpoint stick is drawn as the physical foot. The last and its attachment stay unchanged. Springs use the existing carried-column endpoint rule and simulated compression. In Both mode, the translucent midsole is undeformed context, not a deformed mesh. No new last-surface contact is enabled.</p>
<div class="card" id="spring_map_card"><h2>Whole-shoe deformation map</h2><canvas id="spring_map"></canvas><p class="legend" id="spring_hover">Hover over a column for its compression and rest length.</p><p class="legend">All columns, in the fixed shoe-local layout. Squares: driven footprint. Circles: passive surround. The outlined band is the side-view slice; click the map to move it. Color is the solved foundation compression, not force or a fit target.</p></div>
<div class="grid" id="plots"></div><details open><summary>Recorded-output metrics</summary><pre>__METRICS__</pre></details><details><summary>Full experiment settings and numerical diagnostics</summary><p class="legend">Plots and recorded-output angle metrics use degrees. The machine-readable diagnostics below retain their original radian-valued fields.</p><pre>__SUMMARY__</pre></details></main>
<script>
const data=__DATA__,r=data.reference,tr=data.trace,s=data.summary,rt=r.time_s,tt=tr.time_s||[],duration=rt.at(-1),svgNS="http://www.w3.org/2000/svg";
function idx(t,arr){let lo=0,hi=arr.length-1;while(lo<hi){const mid=(lo+hi+1)>>1;if(arr[mid]<=t)lo=mid;else hi=mid-1;}return lo;}
function joints(q){const a=q[2],b=a+q[3],L=r.lengths_m,p=[[q[0],q[1]]];p.push([p[0][0]+L[0]*Math.cos(a),p[0][1]+L[0]*Math.sin(a)]);p.push([p[1][0]+L[1]*Math.cos(b),p[1][1]+L[1]*Math.sin(b)]);return p;}
const canvas=document.getElementById("view"),ctx=canvas.getContext("2d"),footCanvas=document.getElementById("foot_view"),footCtx=footCanvas.getContext("2d"),geometry=data.geometry||{},meshCache={};
function makeMeshCache(mesh,color){const vertices=mesh.vertices,pp=6000,pad=3;let xmin=Infinity,xmax=-Infinity,zmin=Infinity,zmax=-Infinity;for(const v of vertices){xmin=Math.min(xmin,v[0]);xmax=Math.max(xmax,v[0]);zmin=Math.min(zmin,v[2]);zmax=Math.max(zmax,v[2]);}const image=document.createElement("canvas");image.width=Math.ceil((xmax-xmin)*pp)+2*pad;image.height=Math.ceil((zmax-zmin)*pp)+2*pad;const c=image.getContext("2d");for(let f=0;f<mesh.triangles.length;f++){const face=mesh.triangles[f],shade=mesh.shade[f];c.fillStyle=`rgb(${color.map(v=>Math.round(v*shade)).join(",")})`;c.beginPath();for(let k=0;k<3;k++){const v=vertices[face[k]],x=pad+(v[0]-xmin)*pp,y=pad+(zmax-v[2])*pp;k?c.lineTo(x,y):c.moveTo(x,y);}c.closePath();c.fill();}return {image,xmin,xmax,zmin,zmax,pp,pad};}
if(geometry.available){meshCache.last=makeMeshCache(geometry.last,[231,178,112]);meshCache.midsole=makeMeshCache(geometry.midsole,[92,159,147]);}

const springs=data.springs||{},viewSelect=document.getElementById("shoe_view"),metricSelect=document.getElementById("spring_metric"),sliceSelect=document.getElementById("spring_slice"),allSprings=document.getElementById("all_springs"),mapCanvas=document.getElementById("spring_map");
let springBottom,springTop,springCompression,springRows=[],sliceColumns=[],depthColumns=[],mapProjection=null,hoveredColumn=null;
function unpackFloat(block){const bytes=Uint8Array.from(atob(block.f32le),c=>c.charCodeAt(0)),view=new DataView(bytes.buffer),values=new Float32Array(bytes.length/4);for(let i=0;i<values.length;i++)values[i]=view.getFloat32(i*4,true);return values;}
if(springs.available){springBottom=unpackFloat(springs.bottom_m);springTop=unpackFloat(springs.top_m);springCompression=unpackFloat(springs.compression_m);springRows=[...new Set(springs.anchor_local_m.map(p=>p[1].toFixed(8)))].map(Number).sort((a,b)=>a-b);sliceSelect.max=springRows.length-1;sliceSelect.value=springRows.reduce((best,y,i)=>Math.abs(y)<Math.abs(springRows[best])?i:best,0);depthColumns=springs.anchor_local_m.map((_,i)=>i).sort((a,b)=>springs.anchor_local_m[a][1]-springs.anchor_local_m[b][1]);}else{viewSelect.value="mesh";for(const option of viewSelect.options)if(option.value!=="mesh")option.disabled=true;document.getElementById("spring_controls").hidden=true;}
function springColor(value){const u=Math.max(0,Math.min(1,value));let rgb;if(u<1/3)rgb=[0,u*3,1];else if(u<2/3)rgb=[u*3-1,1,2-u*3];else rgb=[1,3-u*3,0];return `rgb(${rgb.map(v=>Math.round(v*255)).join(",")})`;}
function colorValue(frame,column){const compression=springCompression[frame*springs.rest_length_m.length+column];return metricSelect.value==="strain"?compression/springs.rest_length_m[column]:compression*1000/springs.color_max_mm;}
function refreshSpringControls(){const active=springs.available&&viewSelect.value!=="mesh";document.getElementById("compression_legend").hidden=!active;document.getElementById("spring_map_card").hidden=!active;if(!springs.available){document.getElementById("spring_status").textContent=springs.reason||"Per-column history is unavailable; mesh view only.";return;}const y=springRows[Number(sliceSelect.value)];sliceColumns=depthColumns.filter(i=>Math.abs(springs.anchor_local_m[i][1]-y)<springs.spacing_m*.45);document.getElementById("slice_label").textContent=(y*1000).toFixed(1)+" mm";document.getElementById("compression_title").textContent=metricSelect.value==="strain"?"Foundation compression / original rest length [%] — fixed 0-100%":"Foundation compression [mm] — one fixed scale for the whole rollout";const max=metricSelect.value==="strain"?100:springs.color_max_mm;document.getElementById("compression_ticks").replaceChildren(...[0,1/3,2/3,1].map(v=>{const e=document.createElement("span");e.textContent=(max*v).toFixed(1);return e;}));document.getElementById("spring_status").textContent=`${springs.rest_length_m.length} columns; ${allSprings.checked?springs.rest_length_m.length:sliceColumns.length} drawn in the side view. Springs and pose use the same saved step. Contact replay is checked against saved forces and moment; no leg dynamics or fitting rerun.`;}
function springFrame(t){return springs.available?idx(t,springs.time_s):0;}
function springPoints(frame){if(!springs.available||viewSelect.value==="mesh")return [];const result=[],count=springs.rest_length_m.length;for(const i of (allSprings.checked?depthColumns:sliceColumns)){const k=(frame*count+i)*3;result.push([springBottom[k],springBottom[k+2]],[springTop[k],springTop[k+2]]);}return result;}
function drawSprings(context,v,frame){if(!springs.available||viewSelect.value==="mesh")return;const count=springs.rest_length_m.length;context.save();context.lineCap="round";context.lineJoin="round";context.lineWidth=Math.max(.8,Math.min(1.6,v.scale*springs.spacing_m*.15));for(const i of (allSprings.checked?depthColumns:sliceColumns)){const k=(frame*count+i)*3,a=v.xy([springBottom[k],springBottom[k+2]]),b=v.xy([springTop[k],springTop[k+2]]),dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy),amp=Math.min(springs.spacing_m*v.scale*.26,length*.13,3);context.strokeStyle=springColor(colorValue(frame,i));context.beginPath();context.moveTo(...a);if(length>2){context.lineTo(a[0]+dx*.13,a[1]+dy*.13);for(let j=0;j<6;j++){const u=.2+j*.12,offset=(j%2?1:-1)*amp;context.lineTo(a[0]+dx*u-dy/length*offset,a[1]+dy*u+dx/length*offset);}context.lineTo(a[0]+dx*.87,a[1]+dy*.87);}context.lineTo(...b);context.stroke();}context.restore();}
function drawSpringMap(frame){if(!springs.available||viewSelect.value==="mesh")return;const [c,w,h]=setup(mapCanvas,245),anchors=springs.anchor_local_m.map(p=>p.map((v,i)=>v+geometry.mount_m[i])),xs=anchors.map(p=>p[0]),ys=anchors.map(p=>p[1]),xmin=Math.min(...xs)-springs.spacing_m,xmax=Math.max(...xs)+springs.spacing_m,ymin=Math.min(...ys)-springs.spacing_m,ymax=Math.max(...ys)+springs.spacing_m,scale=Math.min((w-110)/(xmax-xmin),(h-55)/(ymax-ymin)),ox=w/2-(xmin+xmax)*scale/2,oy=h/2+(ymin+ymax)*scale/2,xy=p=>[ox+p[0]*scale,oy-p[1]*scale],size=springs.spacing_m*scale*.82;mapProjection={scale,oy,ox,frame,offsetX:geometry.mount_m[0],offsetY:geometry.mount_m[1]};c.font="11px system-ui";for(let i=0;i<anchors.length;i++){const p=xy(anchors[i]);c.fillStyle=springColor(colorValue(frame,i));if(springs.driven[i])c.fillRect(p[0]-size/2,p[1]-size/2,size,size);else{c.beginPath();c.arc(...p,size*.43,0,2*Math.PI);c.fill();}}const y=springRows[Number(sliceSelect.value)]+geometry.mount_m[1],bandTop=xy([xmin,y+springs.spacing_m/2]),bandBottom=xy([xmax,y-springs.spacing_m/2]);c.strokeStyle="#233b4a";c.lineWidth=1.5;c.setLineDash([5,3]);c.strokeRect(bandTop[0],bandTop[1],bandBottom[0]-bandTop[0],bandBottom[1]-bandTop[1]);c.setLineDash([]);c.fillStyle="#536774";for(let j=0;j<5;j++){const x=xmin+(xmax-xmin)*j/4;c.textAlign="center";c.fillText((x*1000).toFixed(0),xy([x,ymin])[0],xy([x,ymin])[1]+15);}for(let j=0;j<3;j++){const y=ymin+(ymax-ymin)*j/2;c.textAlign="right";c.fillText((y*1000).toFixed(0),xy([xmin,y])[0]-7,xy([xmin,y])[1]+4);}c.textAlign="center";c.fillText("Intrinsic shoe forward x [mm]",w/2,h-2);c.textAlign="left";c.fillText("Width y [mm]",10,15);c.fillText("Saved spring frame: "+springs.time_s[frame].toFixed(3)+" s",10,30);}
function updateColumnLabel(frame){if(hoveredColumn===null)return;const i=hoveredColumn,compression=springCompression[frame*springs.rest_length_m.length+i],rest=springs.rest_length_m[i];document.getElementById("spring_hover").textContent=`Column ${i} (${springs.driven[i]?"driven":"passive"}): compression ${(compression*1000).toFixed(2)} mm / ${(compression/rest*100).toFixed(1)}%; original rest length ${(rest*1000).toFixed(2)} mm (saved frame ${springs.time_s[frame].toFixed(3)} s).`;}
mapCanvas.onmouseleave=()=>{hoveredColumn=null;document.getElementById("spring_hover").textContent="Hover over a column for its compression and rest length.";};
mapCanvas.onmousemove=e=>{if(!mapProjection)return;const rect=mapCanvas.getBoundingClientRect(),x=(e.clientX-rect.left-mapProjection.ox)/mapProjection.scale-mapProjection.offsetX,y=(mapProjection.oy-e.clientY+rect.top)/mapProjection.scale-mapProjection.offsetY;let best=0,distance=Infinity;for(let i=0;i<springs.anchor_local_m.length;i++){const a=springs.anchor_local_m[i],d=Math.hypot(a[0]-x,a[1]-y);if(d<distance){distance=d;best=i;}}const label=document.getElementById("spring_hover");if(distance>springs.spacing_m*.75){hoveredColumn=null;label.textContent="Hover over a column for its compression and rest length.";return;}hoveredColumn=best;updateColumnLabel(mapProjection.frame);};
mapCanvas.onclick=e=>{if(!mapProjection)return;const y=(mapProjection.oy-(e.clientY-mapCanvas.getBoundingClientRect().top))/mapProjection.scale-mapProjection.offsetY;sliceSelect.value=springRows.reduce((best,v,i)=>Math.abs(v-y)<Math.abs(springRows[best]-y)?i:best,0);refreshSpringControls();draw(time);};for(const control of [viewSelect,metricSelect,sliceSelect,allSprings])control.oninput=()=>{refreshSpringControls();draw(time);};refreshSpringControls();

function place(v,ankle,a){const c=Math.cos(a),sn=Math.sin(a);return [ankle[0]+c*v[0]-sn*v[2],ankle[1]+sn*v[0]+c*v[2]];}
function angle(q){return q[2]+q[3]+Math.PI/2+q[4]-(geometry.static_pitch_rad||0);}
function boundsPoints(p,q){if(!geometry.available)return [];const result=[],a=angle(q);for(const key of ["midsole","last"]){const g=meshCache[key];for(const x of [g.xmin,g.xmax])for(const z of [g.zmin,g.zmax])result.push(place([x,0,z],p[2],a));}return result;}
function setup(c,height){const ratio=devicePixelRatio||1,w=c.clientWidth;c.width=w*ratio;c.height=height*ratio;const context=c.getContext("2d");context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,w,height);return [context,w,height];}
function projection(points,w,h,pad){const xmin=Math.min(...points.map(p=>p[0]))-pad,xmax=Math.max(...points.map(p=>p[0]))+pad,ymin=Math.min(0,...points.map(p=>p[1]))-pad,ymax=Math.max(...points.map(p=>p[1]))+pad,scale=Math.min((w-40)/(xmax-xmin),(h-45)/(ymax-ymin)),ox=w/2-(xmin+xmax)*scale/2,oy=h-20+ymin*scale;return {scale,oy,xy:p=>[ox+scale*p[0],oy-scale*p[1]]};}
function ground(context,w,v){context.strokeStyle="#7b9d86";context.lineWidth=1;context.beginPath();context.moveTo(0,v.oy);context.lineTo(w,v.oy);context.stroke();}
function chain(context,p,v,color,dashed=false){context.strokeStyle=color;context.fillStyle=color;context.lineWidth=dashed?2:4;context.setLineDash(dashed?[6,5]:[]);context.beginPath();p.slice(0,3).forEach((point,i)=>{const z=v.xy(point);i?context.lineTo(...z):context.moveTo(...z);});context.stroke();context.setLineDash([]);for(const point of p.slice(0,2)){context.beginPath();context.arc(...v.xy(point),4,0,Math.PI*2);context.fill();}}
function meshes(context,p,q,v,alpha=1,mode=viewSelect.value){if(!geometry.available)return;const a=angle(q),ankle=v.xy(p[2]);context.save();context.translate(...ankle);context.rotate(-a);for(const key of ["last","midsole"]){if(key==="midsole"&&mode!=="mesh"&&mode!=="both")continue;context.globalAlpha=alpha*(key==="midsole"&&mode==="both"?.18:1);const g=meshCache[key],unit=v.scale/g.pp;context.drawImage(g.image,g.xmin*v.scale-g.pad*unit,-g.zmax*v.scale-g.pad*unit,g.image.width*unit,g.image.height*unit);}context.restore();}
function mounting(context,p,q,v,label=false){if(!geometry.available)return;const ankle=v.xy(p[2]),mount=v.xy(place(geometry.mount_connector_local_m,p[2],angle(q)));context.strokeStyle="#506373";context.lineWidth=5;context.beginPath();context.moveTo(...ankle);context.lineTo(...mount);context.stroke();context.strokeStyle="#233b4a";context.fillStyle="white";context.lineWidth=2;context.beginPath();context.arc(...ankle,label?7:5,0,Math.PI*2);context.fill();context.stroke();context.fillStyle="#176b89";context.beginPath();context.arc(...ankle,2,0,Math.PI*2);context.fill();if(label){context.fillStyle="#233b4a";context.font="12px system-ui";context.fillText("Ankle hinge",Math.max(8,ankle[0]-45),ankle[1]-16);}}
function markers(context,ri,v){if(!document.getElementById("show_markers").checked||!r.foot_marker_target_m)return;context.fillStyle="#b3387d";for(const p of r.foot_marker_target_m[ri]){const z=v.xy(p);context.fillRect(z[0]-2,z[1]-2,4,4);}}
function arrow(context,p,f,v,color){if(!f)return;const a=v.xy(p),b=v.xy([p[0]+f[0]*.0001,p[1]+f[1]*.0001]),ang=Math.atan2(b[1]-a[1],b[0]-a[0]);context.strokeStyle=color;context.fillStyle=color;context.lineWidth=2;context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke();context.beginPath();context.moveTo(...b);context.lineTo(b[0]-8*Math.cos(ang-.5),b[1]-8*Math.sin(ang-.5));context.lineTo(b[0]-8*Math.cos(ang+.5),b[1]-8*Math.sin(ang+.5));context.closePath();context.fill();}
function draw(t){const [context,w,h]=setup(canvas,430),sf=springFrame(Math.min(t,tt.at(-1)||0)),ti=springs.available?springs.trace_index[sf]:(tt.length?idx(Math.min(t,tt.at(-1)),tt):0),ri=idx(springs.available?springs.time_s[sf]:t,rt),q=tt.length?tr.state[ti]:null,ref=joints(r.state[ri]),actual=q?joints(q):null,eq=tr.equilibrium&&tr.equilibrium[ti],showRef=document.getElementById("show_reference").checked;let points=(actual||ref).concat(boundsPoints(actual||ref,q||r.state[ri]),springPoints(sf));if(showRef)points=points.concat(ref,boundsPoints(ref,r.state[ri]));if(eq)points.push(eq.slice(0,2));const v=projection(points,w,h,.1);ground(context,w,v);if(showRef){meshes(context,ref,r.state[ri],v,.18,"last");chain(context,ref,v,"#a1adb3",true);}if(actual){meshes(context,actual,q,v);drawSprings(context,v,sf);chain(context,actual,v,"#176b89");mounting(context,actual,q,v);}else if(!showRef){meshes(context,ref,r.state[ri],v);chain(context,ref,v,"#a1adb3",true);}markers(context,ri,v);
if(eq&&actual){const e=v.xy(eq.slice(0,2)),p=v.xy(actual[0]);context.strokeStyle="#874daf";context.lineWidth=2;context.beginPath();context.moveTo(e[0]-6,e[1]);context.lineTo(e[0]+6,e[1]);context.moveTo(e[0],e[1]-6);context.lineTo(e[0],e[1]+6);context.stroke();context.setLineDash([3,3]);context.beginPath();context.moveTo(...p);context.lineTo(...e);context.stroke();context.setLineDash([]);}
if(actual){arrow(context,actual[0],tr.hip_force_n&&tr.hip_force_n[ti],v,"#874daf");const f=tr.grf_n&&tr.grf_n[ti],m=tr.ankle_contact_moment_nm&&tr.ankle_contact_moment_nm[ti];let cop=actual[2][0];if(f&&f[1]>5&&Number.isFinite(m))cop+=(m-actual[2][1]*f[0])/f[1];arrow(context,[cop,0],f,v,"#c36b2d");}context.fillStyle="#536774";context.font="12px system-ui";context.fillText("y: up | x: forward",15,22);if(tt.length&&t>tt.at(-1)+.001)context.fillText("Simulation held at last saved state",15,40);if(!geometry.available)context.fillText(geometry.reason||"Last mesh unavailable",15,60);
const [detail,dw,dh]=setup(footCanvas,310),dp=actual||ref,dq=q||r.state[ri],ankle=dp[2],knee=dp[1],length=Math.hypot(knee[0]-ankle[0],knee[1]-ankle[1]),shortShank=[ankle[0]+.09*(knee[0]-ankle[0])/length,ankle[1]+.09*(knee[1]-ankle[1])/length],detailPoints=[ankle,shortShank,...boundsPoints(dp,dq),...springPoints(sf)],dv=projection(detailPoints,dw,dh,.025);ground(detail,dw,dv);meshes(detail,dp,dq,dv);drawSprings(detail,dv,sf);detail.strokeStyle="#176b89";detail.lineWidth=6;detail.beginPath();detail.moveTo(...dv.xy(shortShank));detail.lineTo(...dv.xy(ankle));detail.stroke();mounting(detail,dp,dq,dv,true);const origin=dv.xy(ankle),axis=dv.xy(place([.10,0,0],ankle,angle(dq))),horizontal=dv.xy([ankle[0]+.10,ankle[1]]);detail.strokeStyle="#8a949b";detail.lineWidth=1;detail.setLineDash([3,3]);detail.beginPath();detail.moveTo(...origin);detail.lineTo(...horizontal);detail.stroke();detail.setLineDash([]);detail.strokeStyle="#ad4b38";detail.lineWidth=2;detail.beginPath();detail.moveTo(...origin);detail.lineTo(...axis);detail.stroke();markers(detail,ri,dv);detail.fillStyle="#223641";detail.font="12px system-ui";detail.fillText("Shoe pitch: "+(angle(dq)*180/Math.PI).toFixed(2)+" deg (+ toe-up)",10,18);const shankTilt=Math.atan2(knee[0]-ankle[0],knee[1]-ankle[1])*180/Math.PI,vrel=r.raw_virtual_foot_angle_deg?r.raw_virtual_foot_angle_deg[ri][0]:(dq[4]*180/Math.PI);detail.fillText("Shank tilt: "+shankTilt.toFixed(2)+" deg | Relative virtual foot: "+vrel.toFixed(2)+" deg",10,32);detail.fillStyle="#506373";detail.fillText("Red: intrinsic shoe axis | Gray: horizontal | Blue: shank",10,dh-22);detail.fillStyle="#467d73";detail.fillText(viewSelect.value==="mesh"?"Green: undeformed midsole mesh":"Spring colors: solved foundation compression",10,dh-7);drawSpringMap(sf);updateColumnLabel(sf);const meta=JSON.parse(r.metadata_json||"{}"),srcStart=(meta.selected_source_time_s||[0])[0],loaded_indices=[];for(let i=0;i<r.grf_time_s.length;i++){if(r.grf_target_n[i][1]>50)loaded_indices.push(i);}let cText="";if(loaded_indices.length>0){const t_td=r.grf_time_s[loaded_indices[0]],t_to=r.grf_time_s[loaded_indices[loaded_indices.length-1]];if(t>=t_td&&t<=t_to)cText=" ("+((t-t_td)/(t_to-t_td)*100).toFixed(0)+"% stance)";else if(t<t_td)cText=" (pre-contact)";else cText=" (post-contact)";}document.getElementById("clock").textContent="Source: "+(srcStart+t).toFixed(3)+" s | Window: "+t.toFixed(3)+" / "+duration.toFixed(3)+" s"+cText+(springs.available?" | saved frame "+springs.time_s[sf].toFixed(3)+" s":"");}
document.getElementById("show_reference").onchange=()=>draw(time);document.getElementById("show_markers").onchange=()=>draw(time);
function el(name,attrs,text){const e=document.createElementNS(svgNS,name);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function plot(title,unit,curves){const card=document.createElement("div");card.className="card";const h=document.createElement("h2");h.textContent=title;card.append(h);const svg=el("svg",{viewBox:"0 0 560 245"}),values=curves.flatMap(c=>c.v).filter(Number.isFinite);let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1,pad=Math.max((hi-lo)*.1,.001);lo-=pad;hi+=pad;const X=t=>60+480*t/duration,Y=v=>205-175*(v-lo)/(hi-lo);for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,t=duration*i/4;svg.append(el("path",{d:`M60 ${Y(v)}H540`,stroke:"#e0e6e8"}),el("text",{x:54,y:Y(v)+4,"text-anchor":"end"},v.toPrecision(3)),el("text",{x:X(t),y:225,"text-anchor":"middle"},t.toFixed(2)));}svg.append(el("text",{x:6,y:15},unit));for(const c of curves){let d="";for(let i=0;i<c.v.length;i++)if(Number.isFinite(c.v[i]))d+=(d?"L":"M")+X(c.t[i])+" "+Y(c.v[i])+" ";svg.append(el("path",{d,fill:"none",stroke:c.color,"stroke-width":2,"stroke-dasharray":c.dash?"6 4":""}));}card.append(svg);const legend=document.createElement("div");legend.className="legend";legend.textContent=curves.map(c=>c.name).join(" · ");card.append(legend);document.getElementById("plots").append(card);}
const curve=(name,t,v,color,dash=false)=>({name,t,v,color,dash}),state=tr.state||[],eq=tr.equilibrium||[];
for(let j=0;j<2;j++)plot("Hip "+(j?"up":"forward")+" position","m",[curve("Recorded",rt,r.hip_target_m.map(v=>v[j]),"#909ea5",true),curve("Simulated",tt,state.map(v=>v[j]),"#176b89"),curve("Equilibrium",tt,eq.map(v=>v[j]),"#874daf")]);
for(let j=0;j<2;j++)plot((j?"Ankle relative to shank (solver)":"Knee relative angle"),"deg",[curve(j&&r.foot_ground_target_rad?"Derived relative target":"Recorded",rt,r.joint_target_rad.map(v=>v[j]*180/Math.PI),"#909ea5",true),curve("Simulated",tt,state.map(v=>v[j+3]*180/Math.PI),"#176b89"),curve("Equilibrium",tt,eq.map(v=>v[j+2]*180/Math.PI),"#874daf")]);
plot("Shoe pitch to ground (+ toe-up)","deg",[curve(r.foot_ground_target_rad?"Reconstructed ground target":"Reference reconstructed",rt,(r.foot_ground_target_rad||r.state.map(q=>angle(q))).map(a=>a*180/Math.PI),"#909ea5",true),curve("Simulated / rendered",tt,state.map(q=>angle(q)*180/Math.PI),"#176b89")]);
if(r.shank_inclination_rad){plot("Shank inclination (+ forward tilt)","deg",[curve("Reference shank",rt,r.shank_inclination_rad.map(a=>a*180/Math.PI),"#909ea5",true),curve("Simulated shank",tt,state.map(q=>(q[2]+q[3]+Math.PI/2)*180/Math.PI),"#176b89")]);}
if(r.raw_virtual_foot_angle_deg){plot("Virtual-foot relative to shank (Visual3D)","deg",[curve("Exported RVirtualFootAngle alpha",rt,r.raw_virtual_foot_angle_deg.map(v=>v[0]),"#909ea5",true)]);}
for(let j=0;j<2;j++)plot((j?"Vertical":"Fore-aft")+" ground force","N",[curve("Measured",r.grf_time_s,r.grf_target_n.map(v=>v[j]),"#9b744f",true),curve("Simulated",tt,(tr.grf_n||[]).map(v=>v[j]),"#c36b2d")]);
let playing=false,last=null,time=0;const metaInfo=JSON.parse(r.metadata_json||"{}"),sStart=(metaInfo.selected_source_time_s||[0])[0],lIds=[];for(let i=0;i<r.grf_time_s.length;i++){if(r.grf_target_n[i][1]>50)lIds.push(i);}
if(lIds.length>0){const t_td=r.grf_time_s[lIds[0]],t_to=r.grf_time_s[lIds[lIds.length-1]],t_mid=t_td+0.5*(t_to-t_td),t_late=Math.min(t_to-0.005,Math.max(t_td,0.370-sStart));const navs=[["Toe-off",t_to],["Frame 0.370 s",t_late],["Midstance (50%)",t_mid],["Measured touchdown",t_td]];navs.forEach(([name,tv])=>{const b=document.createElement("button");b.textContent=name;b.style.marginLeft="4px";b.onclick=()=>{playing=false;time=tv;document.getElementById("play").textContent="Play";document.getElementById("slider").value=time/duration*1000;draw(time);};document.getElementById("play").after(b);});}
document.getElementById("slider").oninput=e=>{time=duration*Number(e.target.value)/1000;draw(time);};document.getElementById("play").onclick=()=>{playing=!playing;last=null;document.getElementById("play").textContent=playing?"Pause":"Play";};function tick(now){if(playing){if(last!==null)time=(time+(now-last)/1000*.25)%duration;last=now;document.getElementById("slider").value=time/duration*1000;draw(time);}requestAnimationFrame(tick);}addEventListener("resize",()=>draw(time));draw(0);requestAnimationFrame(tick);
</script></html>"""
