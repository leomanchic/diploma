"""Create a dependency-free interactive HTML viewer for one Gazebo run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from simulation.gazebo_offline import load_gazebo_recording, shared_stations
from validation.three_station_audio_tracking_study import ESTIMATOR_VARIANTS


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def create_viewer(directory: Path) -> Path:
    directory = Path(directory)
    recording = load_gazebo_recording(directory)
    summary = json.loads((directory / "summary.json").read_text())
    methods = {}
    for method in ESTIMATOR_VARIANTS:
        with (directory / f"tracking_{method}.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        methods[method] = [{
            "t": float(row["processing_time_s"]),
            "valid": row["valid"] == "True",
            "truth": [_number(row[f"truth_position_{axis}_m"]) for axis in "xyz"],
            "estimate": [_number(row[f"estimate_position_{axis}_m"]) for axis in "xyz"],
            "error": _number(row["position_error_m"]),
            "reason": row["failure_reason"] or row["status"],
            "covered": row["valid_and_covered"] == "True",
        } for row in rows]
    data = {
        "kind": summary["recording_kind"],
        "stations": [{"id": station.station_id, "p": station.position_world_m.tolist()}
                     for station in shared_stations()],
        "truth": [[float(t), *map(float, p)] for t, p in zip(
            recording.trajectory.knot_times_s,
            recording.trajectory.knot_positions_m, strict=True)],
        "methods": methods,
        "metrics": summary["methods"],
        "export_rate": summary["gazebo_export_rate_hz"],
        "audio_rate": summary["audio_sampling_rate_hz"],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = HTML.replace("__DATA__", payload)
    path = directory / "viewer.html"
    path.write_text(html, encoding="utf-8")
    return path


HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gazebo offline · траектория и оценка</title>
<style>
body{margin:0;background:#101725;color:#e7edf6;font:15px system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:20px}h1{font-size:1.5rem;margin:0 0 8px}p{color:#b9c6d8;margin:4px 0 16px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}.controls input{flex:1;min-width:220px}select,button{background:#27374c;color:white;border:1px solid #59708c;padding:7px;border-radius:5px}canvas{width:100%;display:block;background:#172238;border:1px solid #34465f;border-radius:8px;touch-action:none}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:16px}.panel{background:#19263a;padding:14px;border-radius:8px}.metric{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #30405a;padding:7px 0}.metric b{text-align:right}.legend{display:flex;gap:18px;margin:10px 0;color:#ced8e8}.dot{font-size:23px;line-height:12px}.hint{font-size:13px;color:#9fb3cc}#status{min-height:45px}.error{margin-top:16px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>Gazebo → аудио → 3D сопровождение</h1>
<p id="intro"></p><div class="controls"><label>Метод <select id="method"></select></label><button id="play">▶ Воспроизвести</button><input id="time" type="range" min="0" value="0"><span id="stamp"></span></div>
<div class="grid"><div><canvas id="scene" width="900" height="560"></canvas><div class="legend"><span><span class="dot" style="color:#69d8ef">●</span> истина и текущий объект</span><span><span class="dot" style="color:#f5ad54">●</span> оценка</span><span><span class="dot" style="color:#9d8cff">▲</span> станции</span></div><div class="hint">Перетащите сцену для вращения; колесо мыши меняет масштаб. Линия оценки прерывается на каждом отсутствии публикации.</div></div>
<div class="panel"><h2 style="margin-top:0;font-size:1.1rem">Результат</h2><div id="metrics"></div><p id="status"></p><div class="hint" id="reasons"></div></div></div>
<div class="panel error"><h2 style="margin:0 0 8px;font-size:1.1rem">Ошибка координат во времени, м</h2><canvas id="error" width="1100" height="250"></canvas><div class="hint">Тёмные участки: оценки нет; график ошибки разорван. Наведите курсор на участок для причины.</div></div></main>
<script>const D=__DATA__;let method=Object.keys(D.methods)[0],i=0,yaw=-0.6,pitch=0.45,zoom=3.6,timer=null;
const scene=document.getElementById('scene'),ctx=scene.getContext('2d'),plot=document.getElementById('error'),px=plot.getContext('2d'),range=document.getElementById('time');
const methods=document.getElementById('method');for(const m of Object.keys(D.methods)){let o=document.createElement('option');o.value=m;o.textContent=m;methods.appendChild(o)}
const center=[55,42,24];function P(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];let a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);return [scene.width/2+a*zoom,scene.height/2-(b*Math.cos(pitch)-z*Math.sin(pitch))*zoom]}
function line(points,color,width=2,dash=[]){if(points.length<2)return;ctx.beginPath();ctx.setLineDash(dash);points.forEach((p,j)=>{let s=P(p);j?ctx.lineTo(...s):ctx.moveTo(...s)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
function dot(p,color,r,label){let s=P(p);ctx.beginPath();ctx.arc(s[0],s[1],r,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();if(label){ctx.font='15px system-ui';ctx.fillStyle='#e7edf6';ctx.fillText(label,s[0]+8,s[1]-8)}}
function number(v,n=3){return v===null||v===undefined?'—':Number(v).toFixed(n)}
function metric(label,value){return '<div class="metric"><span>'+label+'</span><b>'+value+'</b></div>'}
function draw(){let rows=D.methods[method],row=rows[i],t=row.t;range.max=rows.length-1;range.value=i;document.getElementById('stamp').textContent=t.toFixed(3)+' s';ctx.clearRect(0,0,scene.width,scene.height);
const O=[0,0,0];line([O,[25,0,0]],'#d77979',2);line([O,[0,25,0]],'#87c798',2);line([O,[0,0,25]],'#82a6ef',2);dot([25,0,0],'#d77979',2,'E');dot([0,25,0],'#87c798',2,'N');dot([0,0,25],'#82a6ef',2,'U');
let truth=D.truth.filter(r=>r[0]>=rows[0].t&&r[0]<=rows.at(-1).t).map(r=>r.slice(1));line(truth,'#46798d',2,[5,5]);line(D.truth.filter(r=>r[0]>=rows[0].t&&r[0]<=t).map(r=>r.slice(1)),'#69d8ef',3);
let segment=[];for(let j=0;j<=i;j++){const r=rows[j];if(r.valid&&r.estimate.every(Number.isFinite)){segment.push(r.estimate)}else{line(segment,'#f5ad54',3);segment=[]}}line(segment,'#f5ad54',3);
D.stations.forEach(s=>dot(s.p,'#9d8cff',6,s.id));dot(row.truth,'#69d8ef',8,'объект');if(row.valid)dot(row.estimate,'#f5ad54',6,'оценка');
let m=D.metrics[method];document.getElementById('metrics').innerHTML=metric('Подтверждение',number(m.first_confirmation_time_s)+' с')+metric('Наличие оценки',m.valid_publication_count+'/'+m.publication_count+' ('+(100*m.availability_fraction).toFixed(1)+'%)')+metric('RMSE при оценке',number(m.position_rmse_m_conditional)+' м')+metric('Покрытие при оценке',m.coverage_fraction_conditional===null?'—':(100*m.coverage_fraction_conditional).toFixed(1)+'%')+metric('Принятые обновления',m.accepted_update_count);
document.getElementById('status').textContent=row.valid?'Оценка есть · ошибка '+number(row.error)+' м · 95% область '+(row.covered?'покрывает':'не покрывает')+' истину':'Оценки нет · '+row.reason;
document.getElementById('reasons').textContent='Причины отсутствия оценки: '+JSON.stringify(m.failure_reasons)+' · Отказы обновлений: '+JSON.stringify(m.update_rejection_reasons);drawError(rows)}
function drawError(rows){px.clearRect(0,0,plot.width,plot.height);let left=58,right=plot.width-20,top=15,bottom=plot.height-38,t0=rows[0].t,t1=rows.at(-1).t,max=Math.max(0.5,...rows.map(r=>r.error||0))*1.1;let X=t=>left+(t-t0)/(t1-t0)*(right-left),Y=e=>bottom-e/max*(bottom-top);px.fillStyle='#29374c';rows.forEach((r,j)=>{if(!r.valid){let a=j?0.5*(rows[j-1].t+r.t):t0,b=j+1<rows.length?0.5*(r.t+rows[j+1].t):t1;px.fillRect(X(a),top,Math.max(1,X(b)-X(a)),bottom-top)}});px.strokeStyle='#7890ae';px.lineWidth=1;px.beginPath();px.moveTo(left,top);px.lineTo(left,bottom);px.lineTo(right,bottom);px.stroke();px.fillStyle='#bccade';px.font='13px system-ui';px.fillText(max.toFixed(2),5,top+5);px.fillText('0',34,bottom+4);px.fillText(t0.toFixed(1)+' s',left,bottom+23);px.fillText(t1.toFixed(1)+' s',right-42,bottom+23);
px.strokeStyle='#f5ad54';px.lineWidth=2;px.beginPath();let open=false;rows.forEach(r=>{if(r.valid&&r.error!==null){open?px.lineTo(X(r.t),Y(r.error)):px.moveTo(X(r.t),Y(r.error));open=true}else{if(open)px.stroke();px.beginPath();open=false}});px.stroke();px.strokeStyle='#69d8ef';px.beginPath();px.moveTo(X(rows[i].t),top);px.lineTo(X(rows[i].t),bottom);px.stroke()}
methods.onchange=()=>{method=methods.value;i=Math.min(i,D.methods[method].length-1);draw()};range.oninput=()=>{i=+range.value;draw()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Воспроизвести'}else{timer=setInterval(()=>{i=(i+1)%D.methods[method].length;draw()},160);document.getElementById('play').textContent='⏸ Пауза'}};
let dragging=false,last=null;scene.onpointerdown=e=>{dragging=true;last=[e.clientX,e.clientY];scene.setPointerCapture(e.pointerId)};scene.onpointerup=()=>dragging=false;scene.onpointermove=e=>{if(!dragging)return;yaw+=(e.clientX-last[0])*0.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-last[1])*0.008));last=[e.clientX,e.clientY];draw()};scene.onwheel=e=>{e.preventDefault();zoom=Math.max(1.2,Math.min(12,zoom*Math.exp(-e.deltaY*0.001)));draw()};plot.onmousemove=e=>{const rect=plot.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*plot.width,rows=D.methods[method];let t=rows[0].t+(x-58)/(plot.width-78)*(rows.at(-1).t-rows[0].t);let j=rows.reduce((best,r,k)=>Math.abs(r.t-t)<Math.abs(rows[best].t-t)?k:best,0);plot.title=rows[j].valid?'t='+rows[j].t.toFixed(3)+' s; ошибка '+number(rows[j].error)+' м':'t='+rows[j].t.toFixed(3)+' s; оценки нет: '+rows[j].reason};
document.getElementById('intro').textContent=D.kind+' · Gazebo '+D.export_rate+' Гц · аудио '+D.audio_rate+' Гц · ENU, метры, секунды';draw();</script></body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(create_viewer(args.directory))
