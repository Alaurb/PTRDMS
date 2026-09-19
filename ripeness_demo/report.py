from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .detectors import CLASS_COLORS, class_order_for
from .mapping import Pose, SpatialDetection


def _plant_columns(
    poses: list[Pose],
    detections: list[SpatialDetection],
    map_size: tuple[int, int] | None = None,
    map_width_m: float = 35.0,
    row_offset_m: float = 1.20,
    taxonomy: str | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[SpatialDetection]] = {}
    for item in detections:
        grouped.setdefault((item.frame, item.side), []).append(item)

    class_order = class_order_for([item.class_name for item in detections], taxonomy)
    plants: list[dict[str, object]] = []
    for pose in poses:
        match = re.search(r"panoramic(\d+)", pose.frame, flags=re.IGNORECASE)
        frame_label = match.group(1) if match else Path(pose.frame).stem
        for side, side_label, offset in (("left", "L", row_offset_m), ("right", "R", -row_offset_m)):
            items = grouped.get((pose.frame, side), [])
            counts = Counter(item.class_name for item in items)
            cubes = [
                {"class_name": item.class_name, "confidence": round(item.confidence, 6),
                 "x": item.x, "y": item.y, "z": item.z, "track_id": item.track_id,
                 "position_source": item.position_source}
                for class_name in class_order
                for item in sorted(
                    (candidate for candidate in items if candidate.class_name == class_name),
                    key=lambda candidate: candidate.confidence,
                    reverse=True,
                )
            ]
            plants.append(
                {
                    "plant_id": f"P{frame_label}-{side_label}",
                    "frame": pose.frame,
                    "frame_label": frame_label,
                    "side": side,
                    "x": round(pose.x - math.sin(pose.yaw) * offset, 6),
                    "y": round(pose.y + math.cos(pose.yaw) * offset, 6),
                    "total": len(items),
                    "counts": {name: counts.get(name, 0) for name in class_order},
                    "cubes": cubes,
                    "map_px": round(
                        pose.map_px - math.sin(pose.yaw) * offset * map_size[0] / map_width_m
                        if map_size else pose.map_px,
                        3,
                    ),
                    "map_py": round(
                        pose.map_py - math.cos(pose.yaw) * offset * map_size[0] / map_width_m
                        if map_size else pose.map_py,
                        3,
                    ),
                    "assignment_source": "frame_side_proxy",
                }
            )
    return plants


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def write_csv_files(output_dir: Path, poses: list[Pose], detections: list[SpatialDetection], row_offset_m: float = 1.20, taxonomy: str | None = None) -> None:
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(asdict(poses[0]).keys()) if poses else ["frame", "x", "y", "z", "yaw", "map_px", "map_py", "source", "route"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in poses)

    with (output_dir / "detections.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [field.name for field in dataclass_fields(SpatialDetection)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in detections)

    class_order = class_order_for([item.class_name for item in detections], taxonomy)
    plant_rows = []
    for plant in _plant_columns(poses, detections, row_offset_m=row_offset_m, taxonomy=taxonomy):
        counts = plant["counts"]
        plant_rows.append(
            {
                "plant_id": plant["plant_id"],
                "frame": plant["frame"],
                "side": plant["side"],
                "x": plant["x"],
                "y": plant["y"],
                "total": plant["total"],
                **{name: counts[name] for name in class_order},
                "assignment_source": plant["assignment_source"],
            }
        )
    with (output_dir / "plants.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "plant_id", "frame", "side", "x", "y", "total",
            *class_order, "assignment_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plant_rows)


def draw_map_overlay(map_image: Image.Image, poses: list[Pose], detections: list[SpatialDetection], output_path: Path, taxonomy: str | None = None) -> None:
    canvas = map_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default(size=13)
    route_points: dict[int, list[tuple[float, float]]] = {}
    for pose in poses:
        route_points.setdefault(pose.route, []).append((pose.map_px, pose.map_py))
    for points in route_points.values():
        if len(points) > 1:
            draw.line(points, fill=(14, 116, 144, 220), width=3)
    for pose in poses:
        draw.ellipse((pose.map_px - 2, pose.map_py - 2, pose.map_px + 2, pose.map_py + 2), fill=(8, 145, 178, 240))
    for detection in detections:
        color = (*_rgb(CLASS_COLORS[detection.class_name]), 220)
        radius = 2 + int(detection.confidence)
        draw.ellipse(
            (detection.map_px - radius, detection.map_py - radius, detection.map_px + radius, detection.map_py + radius),
            fill=color,
            outline=(255, 255, 255, 230),
            width=1,
        )
    legend_x, legend_y = 12, 12
    class_order = class_order_for([item.class_name for item in detections], taxonomy)
    draw.rounded_rectangle((6, 6, 230, 28 + len(class_order) * 20), radius=8, fill=(255, 255, 255, 220), outline=(15, 23, 42, 150))
    draw.text((legend_x, legend_y), "Tomato maturity", fill=(15, 23, 42), font=font)
    for index, name in enumerate(class_order, start=1):
        color = CLASS_COLORS[name]
        y = legend_y + index * 20
        draw.ellipse((legend_x, y, legend_x + 10, y + 10), fill=(*_rgb(color), 255))
        draw.text((legend_x + 17, y - 2), name, fill=(15, 23, 42), font=font)
    canvas.save(output_path, quality=95)


def write_summary(
    output_dir: Path,
    poses: list[Pose],
    detections: list[SpatialDetection],
    detector_name: str,
    pose_source: str,
    processed_frames: int,
    taxonomy: str | None = None,
) -> dict[str, object]:
    counts = Counter(item.class_name for item in detections)
    plants = _plant_columns(poses, detections, taxonomy=taxonomy)
    limitations = [
        "detections are model predictions rather than independently verified ground truth",
        "each plant column is a frame-side proxy because individual plant IDs were not supplied",
    ]
    if pose_source == "synthetic_map_path":
        limitations.append("synthetic_map_path is generated from filename order; not surveyed positions")
    if any(d.position_source == "assumed_row_plane" for d in detections):
        limitations.append("row distance is assumed when depth is unavailable; background rows cannot be verified")
    limitations.append("track IDs are within-pass spatial association candidates, not verified unique fruit identities")
    if detector_name.startswith("color_shape_fallback"):
        limitations.insert(
            0,
            "color_shape_fallback is an engineering demonstration used when trained YOLO weights are unavailable",
        )
    class_order = class_order_for([item.class_name for item in detections], taxonomy)
    summary: dict[str, object] = {
        "taxonomy": taxonomy,
        "processed_frames": processed_frames,
        "trajectory_points": len(poses),
        "detections": len(detections),
        "plant_columns": len(plants),
        "plants_with_detections": sum(int(plant["total"] > 0) for plant in plants),
        "class_counts": {name: counts.get(name, 0) for name in class_order},
        "detector": detector_name,
        "pose_source": pose_source,
        "plant_assignment": "frame_side_proxy",
        "limitations": limitations,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


LEGACY_COLUMN_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PTRDMS | Panoramic Tomato Ripeness Detection and Mapping</title>
<style>
:root{--ink:#15231d;--muted:#64736b;--panel:#fff;--line:#d8e3dc;--bg:#edf3ee;--accent:#08785a;--empty:#9ca3af}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#e7f0e8,#f7f3eb);color:var(--ink);font-family:Inter,system-ui,sans-serif}
header{padding:26px clamp(16px,4vw,52px) 14px}h1{font-size:clamp(25px,4vw,43px);margin:0;letter-spacing:-.035em}header p{margin:7px 0 0;color:var(--muted)}
.wrap{padding:0 clamp(16px,4vw,52px) 46px}.panel,.card{background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:17px;box-shadow:0 12px 34px rgba(24,52,39,.07)}
.panel{padding:14px}.panel h2{font-size:17px;margin:2px 0 12px}.subtle{color:var(--muted);font-size:12px;font-weight:400;margin-left:6px}
.scene{height:590px;width:100%;border-radius:12px;background:#f7f8f6;border:1px solid var(--line);cursor:grab}.scene:active{cursor:grabbing}
.filters,.legend{display:flex;flex-wrap:wrap;gap:10px;margin:9px 0}.filters label,.legend span{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:5px}.dot{width:10px;height:10px;border-radius:2px;display:inline-block;border:1px solid rgba(20,33,29,.35)}
.plant-info{margin-top:8px;padding:8px 10px;border-radius:9px;background:#f2f6f3;color:var(--muted);font-size:12px}.plant-info:empty{display:none}
.location-grid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(300px,.7fr);gap:16px;margin-top:16px}.map-stage{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f4f6f4}.map-stage>img{width:100%;display:block}.map-point{position:absolute;width:7px;height:7px;padding:0;transform:translate(-50%,-50%);border:1px solid #fff;border-radius:50%;box-shadow:0 1px 2px rgba(0,0,0,.35);cursor:pointer}.map-point.side-left{border-radius:1px;transform:translate(-50%,-50%) rotate(45deg)}.map-point.is-selected{outline:2px solid #12251d;outline-offset:2px}.map-point:focus-visible{outline:2px solid #08785a;outline-offset:2px}
.photo-placeholder,.photo-detail{min-height:315px;border:1px dashed var(--line);border-radius:12px;background:#f5f7f5}.photo-placeholder{display:grid;place-items:center;padding:30px;color:var(--muted);text-align:center}.photo-detail{margin:0;overflow:hidden}.photo-detail[hidden]{display:none}.photo-detail img{width:100%;aspect-ratio:1/1;object-fit:contain;display:block;background:#eef1ef}.photo-detail figcaption{padding:10px 12px;color:var(--muted);font-size:12px;line-height:1.55}.count-line{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}.count-line span{display:inline-flex;align-items:center;gap:5px}
.stats-title{font-size:17px;margin:22px 2px 10px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:12px}.card{padding:14px 16px}.card b{display:block;font-size:25px}.card span{font-size:12px;color:var(--muted)}
@media(max-width:980px){.location-grid{grid-template-columns:1fr}.scene{height:480px}.photo-placeholder,.photo-detail{min-height:0}.photo-detail img{aspect-ratio:16/10}}
@media(max-width:600px){header{padding-top:20px}.scene{height:390px}.cards{grid-template-columns:repeat(2,1fr)}.subtle{display:block;margin:4px 0 0}}
</style>
</head>
<body>
<header><h1>PTRDMS</h1><p>Panoramic Tomato Ripeness Detection and Mapping</p></header>
<main class="wrap">
  <section class="panel" aria-labelledby="scene-title">
    <h2 id="scene-title">3D Tomato Ripeness Distribution</h2>
    <label>Display <select id="display-mode"><option value="height">Estimated spatial height</option><option value="count">Count by maturity</option></select></label>
    <div class="filters" id="filters" aria-label="Ripeness filters"></div>
    <canvas class="scene" id="scene" role="img" aria-label="3D columns of tomato ripeness cubes arranged by observation position"></canvas>
    <div class="plant-info" id="plant-info" aria-live="polite"></div>
  </section>

  <section class="location-grid">
    <article class="panel" aria-labelledby="map-title">
      <h2 id="map-title">2D Spatial Map</h2>
      <div class="map-stage" id="map-stage">
        <img src="map_base.png" alt="2D farm map">
      </div>
      <div class="legend" id="legend"></div>
    </article>
    <article class="panel" aria-labelledby="photo-title">
      <h2 id="photo-title">Position Image</h2>
      <div class="photo-placeholder" id="photo-placeholder" aria-hidden="true"></div>
      <figure class="photo-detail" id="photo-detail" hidden>
        <img id="photo-image" alt="Tomato detections at the selected position">
        <figcaption><strong id="photo-location"></strong><div id="photo-meta"></div><div class="count-line" id="photo-counts"></div></figcaption>
      </figure>
    </article>
  </section>

  <h2 class="stats-title">Summary Statistics</h2>
  <section class="cards" id="cards" aria-label="Detection summary statistics"></section>
</main>
<script>
const DATA=__DATA__;
const colors=DATA.colors,summary=DATA.summary,plants=DATA.plants;
let spatialMode=true;
document.querySelector('#display-mode').addEventListener('change',event=>{spatialMode=event.target.value==='height';draw()});
const emptyColor='#9ca3af';
const labels={processed_frames:'Processed frames',plant_columns:'Observation columns',detections:'Detected tomatoes',immature:'Immature',green_mature:'Green mature',discoloration:'Discoloration',mature:'Mature',empty:'No detection'};
const metrics=[['processed_frames',summary.processed_frames],['plant_columns',summary.plant_columns],['detections',summary.detections],...Object.entries(summary.class_counts)];
document.querySelector('#cards').innerHTML=metrics.map(([k,v])=>`<div class="card"><b style="color:${colors[k]||'#173d30'}">${v}</b><span>${labels[k]}</span></div>`).join('');
document.querySelector('#legend').innerHTML=[...Object.entries(colors),['empty',emptyColor]].map(([k,v])=>`<span><i class="dot" style="background:${v}"></i>${labels[k]}</span>`).join('');
const enabled={};
document.querySelector('#filters').innerHTML=Object.entries(colors).map(([k,v])=>{enabled[k]=true;return `<label><input type="checkbox" data-k="${k}" checked><i class="dot" style="background:${v}"></i>${labels[k]}</label>`}).join('');
document.querySelectorAll('#filters input').forEach(input=>input.addEventListener('change',()=>{enabled[input.dataset.k]=input.checked;draw()}));

const galleryByKey=new Map(DATA.gallery.map(item=>[`${item.frame}|${item.side}`,item]));
const mapStage=document.querySelector('#map-stage');
const mapWidth=DATA.map_size.width,mapHeight=DATA.map_size.height;
function plantColor(plant){return plant.total===0?emptyColor:colors[plant.cubes[0].class_name]}
for(const plant of plants){
  const point=document.createElement('button');point.type='button';point.className=`map-point side-${plant.side}`;point.dataset.plantId=plant.plant_id;
  point.style.left=`${Math.max(0,Math.min(100,plant.map_px/mapWidth*100))}%`;point.style.top=`${Math.max(0,Math.min(100,plant.map_py/mapHeight*100))}%`;point.style.background=plantColor(plant);
  point.setAttribute('aria-label',`${plant.plant_id}, ${plant.total?plant.total+' detected tomatoes':'no detected tomato'}`);
  point.addEventListener('click',()=>selectPlant(plant));mapStage.appendChild(point);
}

let selectedPlantId=null;
function countsMarkup(plant){return [...Object.keys(colors).map(k=>[k,plant.counts[k]])].filter(([,v])=>v>0).map(([k,v])=>`<span><i class="dot" style="background:${colors[k]}"></i>${labels[k]} ${v}</span>`).join('')||`<span><i class="dot" style="background:${emptyColor}"></i>No detected tomato</span>`}
function selectPlant(plant){
  selectedPlantId=plant.plant_id;document.querySelectorAll('.map-point').forEach(point=>point.classList.toggle('is-selected',point.dataset.plantId===selectedPlantId));
  const gallery=galleryByKey.get(`${plant.frame}|${plant.side}`),placeholder=document.querySelector('#photo-placeholder'),detail=document.querySelector('#photo-detail');
  placeholder.hidden=true;detail.hidden=false;document.querySelector('#photo-image').src=gallery.path;
  document.querySelector('#photo-image').alt=`Detection image for ${plant.plant_id}`;document.querySelector('#photo-location').textContent=plant.plant_id;
  document.querySelector('#photo-meta').textContent=`${plant.frame} · ${plant.side==='left'?'Left':'Right'} · ${plant.total} detected tomatoes`;
  document.querySelector('#photo-counts').innerHTML=countsMarkup(plant);document.querySelector('#plant-info').textContent=`Selected ${plant.plant_id}: ${plant.total?plant.total+' detected tomatoes':'no detected tomato'}`;draw();
}

const canvas=document.querySelector('#scene'),ctx=canvas.getContext('2d'),info=document.querySelector('#plant-info');
let yaw=-.2,zoom=1,drag=false,moved=false,last=[0,0],start=[0,0],hitboxes=[];
const xs=plants.map(p=>p.x),ys=plants.map(p=>p.y),xMin=Math.min(...xs),xMax=Math.max(...xs),yMin=Math.min(...ys),yMax=Math.max(...ys),maxCount=Math.max(1,...plants.map(p=>Math.max(1,p.total)));
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function shade(hex,amount){const n=parseInt(hex.slice(1),16),r=Math.max(0,Math.min(255,(n>>16)+amount)),g=Math.max(0,Math.min(255,((n>>8)&255)+amount)),b=Math.max(0,Math.min(255,(n&255)+amount));return `rgb(${r},${g},${b})`}
function geometry(w,h){const ground=Math.min(w/Math.max(18,xMax-xMin+8),h/Math.max(20,yMax-yMin+maxCount*.8+9))*.94*zoom;return {ground,vertical:Math.min(24,h/(maxCount+9))*zoom,cx:(xMin+xMax)/2,cy:(yMin+yMax)/2}}
function project(p,w,h,g){const x=p.x-g.cx,y=p.y-g.cy,rx=x*Math.cos(yaw)-y*Math.sin(yaw),depth=x*Math.sin(yaw)+y*Math.cos(yaw);return {x:w/2+rx*g.ground,y:h*.76+depth*g.ground*.42-(p.z||0)*g.vertical,depth}}
function path(vertices,fill,stroke='#26332e',width=.7){ctx.beginPath();ctx.moveTo(vertices[0][0],vertices[0][1]);vertices.slice(1).forEach(v=>ctx.lineTo(v[0],v[1]));ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}
function block(q,color,cw,ch,meta){const x=q.x,y=q.y,dx=cw*.24,dy=-ch*.2,A=[x-cw/2,y-ch],B=[x+cw/2,y-ch],C=[x+cw/2,y],D=[x-cw/2,y],At=[A[0]+dx,A[1]+dy],Bt=[B[0]+dx,B[1]+dy],Ct=[C[0]+dx,C[1]+dy],selected=meta.plant.plant_id===selectedPlantId;path([A,B,C,D],color,selected?'#07150f':'#26332e',selected?2:.7);path([B,Bt,Ct,C],shade(color,-34),selected?'#07150f':'#26332e',selected?2:.7);path([At,Bt,B,A],shade(color,30),selected?'#07150f':'#26332e',selected?2:.7);hitboxes.push({x1:A[0],y1:At[1],x2:Ct[0],y2:C[1],...meta})}
function line(a,b,color='#a7b0ab',width=1){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight,g=geometry(w,h);hitboxes=[];ctx.clearRect(0,0,w,h);ctx.fillStyle='#f7f8f6';ctx.fillRect(0,0,w,h);
  const topHeight=Math.max(1,...DATA.points.map(p=>p.z)),heightScale=maxCount/topHeight;
  const gx0=xMin-1.2,gx1=xMax+1.2,gy0=yMin-1.1,gy1=yMax+1.1;
  for(const p of plants.filter(p=>p.side==='left'))line(project({x:p.x,y:gy0,z:0},w,h,g),project({x:p.x,y:gy1,z:0},w,h,g),'#d3d9d5');
  for(let i=0;i<=4;i++){const y=gy0+(gy1-gy0)*i/4;line(project({x:gx0,y,z:0},w,h,g),project({x:gx1,y,z:0},w,h,g),'#d3d9d5')}
  const ordered=[...plants].sort((a,b)=>project(a,w,h,g).depth-project(b,w,h,g).depth);
  for(const plant of ordered){
    const base=project({x:plant.x,y:plant.y,z:0},w,h,g),cw=Math.max(9,Math.min(22,g.ground*.72)),ch=g.vertical*.88,selected=plant.plant_id===selectedPlantId;
    path([[base.x-cw*.62,base.y],[base.x-cw*.15,base.y-ch*.18],[base.x+cw*.62,base.y],[base.x+cw*.15,base.y+ch*.18]],'#e2e7e3',selected?'#07150f':'#89968f',selected?2:.7);
    if(plant.total===0){block(project({x:plant.x,y:plant.y,z:0},w,h,g),emptyColor,cw,ch,{plant,cube:null,index:0})}
    else plant.cubes.forEach((cube,index)=>{if(enabled[cube.class_name])block(project(spatialMode?{x:cube.x,y:cube.y,z:cube.z*heightScale}:{x:plant.x,y:plant.y,z:index},w,h,g),colors[cube.class_name],spatialMode?Math.min(cw,9):cw,spatialMode?8:ch,{plant,cube,index})});
    ctx.fillStyle='#4c5a53';ctx.font='10px Inter, sans-serif';ctx.textAlign='center';ctx.fillText(plant.plant_id.replace('P',''),base.x,base.y+16);
  }
  const axisBase=project({x:gx0,y:gy0,z:0},w,h,g);line(axisBase,project({x:gx1,y:gy0,z:0},w,h,g),'#4a5650',1.4);line(axisBase,project({x:gx0,y:gy1,z:0},w,h,g),'#4a5650',1.4);line(axisBase,project({x:gx0,y:gy0,z:maxCount+1},w,h,g),'#4a5650',1.4);
  ctx.fillStyle='#26332e';ctx.font='12px Inter, sans-serif';ctx.textAlign='left';const zTop=project({x:gx0,y:gy0,z:maxCount+1},w,h,g);ctx.fillText(spatialMode?'Estimated height (m)':'Tomato count',zTop.x-4,zTop.y-8);ctx.fillText('Frame order / X (m)',w-128,h-16);ctx.fillText('Left / right side / Y',18,h-16);
  for(let tick=0;tick<=4;tick++){const value=(spatialMode?topHeight:maxCount)*tick/4,q=project({x:gx0,y:gy0,z:maxCount*tick/4},w,h,g);ctx.fillText(value.toFixed(spatialMode?2:0),q.x-32,q.y)}
}
function hitAt(clientX,clientY){const r=canvas.getBoundingClientRect(),x=clientX-r.left,y=clientY-r.top;return [...hitboxes].reverse().find(b=>x>=b.x1&&x<=b.x2&&y>=b.y1&&y<=b.y2)}
canvas.addEventListener('mousedown',event=>{drag=true;moved=false;last=[event.clientX,event.clientY];start=[event.clientX,event.clientY]});
window.addEventListener('mouseup',event=>{if(!drag)return;drag=false;if(!moved){const hit=hitAt(event.clientX,event.clientY);if(hit)selectPlant(hit.plant)}});
canvas.addEventListener('mousemove',event=>{if(drag){if(Math.hypot(event.clientX-start[0],event.clientY-start[1])>4)moved=true;if(moved){yaw+=(event.clientX-last[0])*.008;last=[event.clientX,event.clientY];draw()}return}const hit=hitAt(event.clientX,event.clientY);if(hit)info.textContent=hit.cube?`${hit.plant.plant_id} · ${labels[hit.cube.class_name]} · confidence ${hit.cube.confidence.toFixed(2)} · ${hit.plant.total} total`:`${hit.plant.plant_id} · no detected tomato`;else if(!selectedPlantId)info.textContent=''});
canvas.addEventListener('mouseleave',()=>{if(!drag&&!selectedPlantId)info.textContent=''});
canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.max(.55,Math.min(2.4,zoom*(event.deltaY>0?.9:1.1)));draw()},{passive:false});
window.addEventListener('resize',resize);resize();
</script>
</body></html>'''


# The current view deliberately avoids implying that a frame-side observation
# column is a reconstructed tomato plant.  The dashed bundle is an image-space
# grouping aid, not a detected stem or a verified fruit truss.
OBSERVATION_MAP_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PTRDMS | Observation Map</title>
<style>
:root{--ink:#173d30;--muted:#61756b;--line:#d7e3db;--green:#2f6f4f;--bg:#edf4ef}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#edf5ed,#faf8ef);color:var(--ink);font:14px Inter,system-ui,sans-serif}header,main{max-width:1280px;margin:auto}header{padding:28px 24px 14px}h1{margin:0;font-size:clamp(26px,4vw,42px);letter-spacing:-.04em}h2{font-size:17px;margin:0 0 10px}header p,.note,.subtle{color:var(--muted);line-height:1.55}.wrap{padding:0 24px 44px}.panel,.card{background:#ffffffed;border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 28px #31503d12}.panel{padding:15px}.toolbar,.legend,.cards,.count-line{display:flex;flex-wrap:wrap;gap:10px}.toolbar{align-items:center;margin:8px 0 10px}.legend span,.count-line span{display:inline-flex;align-items:center;gap:5px;color:var(--muted);font-size:12px}.dot{width:10px;height:10px;border-radius:50%;border:1px solid #173d3055;display:inline-block}.field{width:100%;min-height:570px;display:block;border:1px solid var(--line);border-radius:12px;background:#f3f7f3}.row-label{fill:#173d30;font-size:12px;font-weight:650}.trajectory{fill:none;stroke:#167a91;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;opacity:.75}.anchor{fill:#2f6f4f;stroke:#fff;stroke-width:2}.anchor.empty{fill:#a7b4ac}.bundle{fill:none;stroke:#47875f;stroke-width:2.4;stroke-dasharray:5 5;stroke-linecap:round;opacity:.72}.fruit{stroke:#fff;stroke-width:1.5;cursor:pointer}.fruit:hover,.selected .fruit{stroke:#122a20;stroke-width:3}.selected .anchor{stroke:#122a20;stroke-width:3}.hint{fill:#fff;stroke:#a9bbb0;stroke-width:1}.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(285px,.6fr);gap:16px;margin-top:16px}.photo-placeholder,.photo-detail{min-height:300px;border:1px dashed var(--line);border-radius:12px;background:#f4f7f4}.photo-placeholder{display:grid;place-items:center;padding:25px;text-align:center;color:var(--muted)}.photo-detail[hidden]{display:none}.photo-detail{margin:0;overflow:hidden;border-style:solid}.photo-detail img{display:block;width:100%;aspect-ratio:1/1;object-fit:contain}.photo-detail figcaption{padding:11px 13px;color:var(--muted);line-height:1.55}.cards{margin-top:14px}.card{padding:13px 16px;min-width:125px}.card b{font-size:25px;display:block}.card span{color:var(--muted);font-size:12px}.warning{margin:12px 0 0;padding:10px 12px;border-left:4px solid #c79523;background:#fff8df;color:#5d4a13;border-radius:6px;font-size:12px;line-height:1.55}@media(max-width:850px){.layout{grid-template-columns:1fr}.field{min-height:430px}.wrap{padding:0 14px 28px}header{padding:22px 14px 12px}}
</style></head><body><header><h1>PTRDMS Observation Map</h1><p>Panoramic tomato monitoring · spatially indexed observations</p></header><main class="wrap">
<section class="panel"><h2>Row-level observation map <span class="subtle">fruit circles and bundle candidates</span></h2><div class="toolbar" id="filters"></div><svg id="field" class="field" role="img" aria-label="Greenhouse row map with tomato observation bundles"></svg><div class="legend" id="legend"></div><div class="warning">Dashed green lines connect detections from the same frame and crop-row side. They are visual observation bundles only—not reconstructed stems, confirmed trusses, or unique plant identities.</div></section>
<section class="layout"><article class="panel"><h2>Selected observation</h2><div class="photo-placeholder" id="placeholder">Select a fruit circle or green anchor to inspect its original annotated panorama view.</div><figure class="photo-detail" id="detail" hidden><img id="photo" alt="Annotated tomato panorama view"><figcaption><strong id="where"></strong><div id="meta"></div><div class="count-line" id="counts"></div></figcaption></figure></article><article class="panel"><h2>How to read this view</h2><p class="note">A green anchor marks one frame-side observation. Fruit circles use the mapped detection estimates. Circle size represents confidence, colour represents the model label, and the route is shown in blue.</p><p class="note">Use the original image at left to review suspected missed fruit. This page preserves the difference between an observation bundle and a biologically verified vine.</p></article></section>
<h2 style="margin:22px 2px 10px">Summary statistics</h2><section class="cards" id="cards"></section></main>
<script>
const DATA=__DATA__, colors=DATA.colors, plants=DATA.plants, summary=DATA.summary, svg=document.querySelector('#field');
const labels={processed_frames:'Processed frames',plant_columns:'Observation positions',detections:'Detected fruit',immature:'Immature',mature_green:'Mature green',harvest_ready:'Harvest ready',overripe_or_defective:'Overripe / defective',green_mature:'Green maturity',discoloration:'Discoloration',mature:'Maturity',empty:'No detection'};
const enabled=Object.fromEntries(Object.keys(colors).map(key=>[key,true]));
document.querySelector('#filters').innerHTML=Object.entries(colors).map(([key,color])=>`<label><input type="checkbox" data-k="${key}" checked><i class="dot" style="background:${color}"></i>${labels[key]}</label>`).join('');
document.querySelectorAll('#filters input').forEach(node=>node.addEventListener('change',()=>{enabled[node.dataset.k]=node.checked;render()}));
document.querySelector('#legend').innerHTML=Object.entries(colors).map(([key,color])=>`<span><i class="dot" style="background:${color}"></i>${labels[key]}</span>`).join('')+`<span><i class="dot" style="background:#2f6f4f"></i>Frame-side anchor</span><span><i class="dot" style="background:#167a91"></i>Camera route</span>`;
const metric=[['processed_frames',summary.processed_frames],['plant_columns',summary.plant_columns],['detections',summary.detections],...Object.entries(summary.class_counts)];document.querySelector('#cards').innerHTML=metric.map(([k,v])=>`<div class="card"><b style="color:${colors[k]||'#173d30'}">${v}</b><span>${labels[k]||k}</span></div>`).join('');
const gallery=new Map(DATA.gallery.map(item=>[`${item.frame}|${item.side}`,item]));const W=DATA.map_size.width,H=DATA.map_size.height;let selected=null;
function position(item){return [item.map_px,item.map_py]};function fruitPosition(plant,cube,index){const x=cube.x/DATA.extent.x*W,y=cube.y/DATA.extent.x*W;return [Number.isFinite(x)?x:plant.map_px,Number.isFinite(y)?y:plant.map_py]}
function esc(text){return String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function select(plant){selected=plant.plant_id;const item=gallery.get(`${plant.frame}|${plant.side}`), detail=document.querySelector('#detail'), holder=document.querySelector('#placeholder');holder.hidden=true;detail.hidden=false;document.querySelector('#photo').src=item?.path||'';document.querySelector('#where').textContent=plant.plant_id;document.querySelector('#meta').textContent=`${plant.frame} · ${plant.side} side · ${plant.total} detected fruit`;document.querySelector('#counts').innerHTML=Object.entries(plant.counts).filter(([,n])=>n).map(([k,n])=>`<span><i class="dot" style="background:${colors[k]}"></i>${labels[k]} ${n}</span>`).join('')||'<span>No detected fruit</span>';render()}
function render(){svg.setAttribute('viewBox',`0 0 ${W} ${H}`);let markup=`<image href="map_base.png" x="0" y="0" width="${W}" height="${H}" opacity=".32"/>`;const route=DATA.trajectory.map(p=>`${p.map_px},${p.map_py}`).join(' ');markup+=`<polyline class="trajectory" points="${route}"/>`;
for(const plant of plants){const isSelected=plant.plant_id===selected, visible=plant.cubes.filter(c=>enabled[c.class_name]), [ax,ay]=position(plant);const dots=visible.map((c,i)=>fruitPosition(plant,c,i));let bundle='';if(dots.length){const ordered=[...dots].sort((a,b)=>a[1]-b[1]);bundle=`<polyline class="bundle" points="${ax},${ay} ${ordered.map(p=>p.join(',')).join(' ')}"/>`;}const circles=visible.map((cube,i)=>{const [x,y]=dots[i],r=4+Math.round(cube.confidence*5);return `<circle class="fruit" data-id="${esc(plant.plant_id)}" cx="${x}" cy="${y}" r="${r}" fill="${colors[cube.class_name]}"/>`}).join('');markup+=`<g class="${isSelected?'selected':''}" data-id="${esc(plant.plant_id)}">${bundle}<circle class="anchor ${plant.total?'':'empty'}" cx="${ax}" cy="${ay}" r="6"/><text class="row-label" x="${ax+8}" y="${ay-8}">${esc(plant.plant_id)}</text>${circles}</g>`;}svg.innerHTML=markup;svg.querySelectorAll('[data-id]').forEach(node=>node.addEventListener('click',()=>{const plant=plants.find(p=>p.plant_id===node.dataset.id);if(plant)select(plant)}));}
render();
</script></body></html>'''


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PTRDMS | 3D Tomato Bunch View</title>
<style>
:root{--ink:#173d30;--muted:#61756b;--line:#d7e3db}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#e8f2e9,#faf8ef);color:var(--ink);font:14px Inter,system-ui,sans-serif}header,main{max-width:1320px;margin:auto}header{padding:14px 20px 8px}h1{font-size:clamp(22px,3vw,32px);letter-spacing:-.04em;margin:0}h2{font-size:15px;margin:0 0 7px}header p,.note,.subtle{color:var(--muted);line-height:1.4}.wrap{padding:0 20px 18px}.panel,.card{background:#ffffffed;border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 22px #31503d12}.panel{padding:12px}.controls,.legend,.count-line{display:flex;gap:8px;flex-wrap:wrap}.controls{align-items:center;margin:5px 0 7px}.legend span,.count-line span{display:inline-flex;align-items:center;gap:4px;color:var(--muted);font-size:11px}.dot{width:9px;height:9px;border-radius:50%;border:1px solid #173d3055;display:inline-block}.scene{height:min(610px,calc(100vh - 170px));min-height:410px;width:100%;display:block;border-radius:11px;border:1px solid var(--line);background:linear-gradient(#f6faf6,#edf3ed);cursor:grab}.scene:active{cursor:grabbing}.warning{margin-top:7px;color:var(--muted);font-size:11px;line-height:1.3}.detail-grid{display:grid;grid-template-columns:1fr;gap:10px;margin:0}.photo-placeholder,.photo-detail{min-height:150px;border:1px dashed var(--line);border-radius:10px;background:#f4f7f4}.photo-placeholder{display:grid;place-items:center;padding:14px;text-align:center;color:var(--muted);font-size:12px}.photo-detail[hidden]{display:none}.photo-detail{margin:0;overflow:hidden;border-style:solid}.photo-detail img{width:100%;height:150px;object-fit:contain;display:block}.photo-detail figcaption{padding:7px 9px;color:var(--muted);font-size:11px;line-height:1.35}.mini-map{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:10px;background:#eff4ef}.mini-map img{display:block;width:100%;max-height:130px;object-fit:contain;opacity:.55}.mini-point{position:absolute;width:7px;height:7px;border-radius:50%;background:#2f6f4f;border:1px solid #fff;transform:translate(-50%,-50%);cursor:pointer}.mini-point.selected{outline:2px solid #122a20;outline-offset:2px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:8px}.card{padding:7px 8px;min-width:0}.card b{font-size:18px;display:block}.card span{color:var(--muted);font-size:10px}@media(min-width:851px){.wrap{height:calc(100vh - 70px);display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:12px;overflow:hidden}.main-panel{grid-column:1;min-height:0}.detail-grid{grid-column:2;min-height:0;align-content:start}.detail-grid>.panel{padding:10px}.detail-grid .note{display:none}}@media(max-width:850px){.wrap{padding:0 14px 28px}.scene{height:460px}header{padding:14px 14px 8px}}
</style></head><body><header><h1>PTRDMS | Tomato observations</h1><p id="evidence-banner">Drag to rotate | scroll to zoom | select a fruit or observation</p></header><main class="wrap">
<section class="panel main-panel"><h2>3D tomato-bunch view</h2><div class="controls" id="filters"></div><canvas class="scene" id="scene" role="img" aria-label="Three-dimensional tomato observation bunches connected by visual vines"></canvas><div class="legend" id="legend"></div><div class="warning">Note: dashed vines are visual frame-side grouping guides, not reconstructed plants.</div></section>
<section class="detail-grid"><article class="panel"><h2>Selected source view</h2><div class="photo-placeholder" id="placeholder">Select a fruit or vine to review the source image.</div><figure class="photo-detail" id="detail" hidden><img id="photo" alt="Annotated tomato panorama view"><figcaption><strong id="where"></strong><div id="meta"></div><div class="count-line" id="counts"></div></figcaption></figure></article><article class="panel"><h2>Row location</h2><div class="mini-map" id="mini-map"><img src="map_base.png" alt="Farm map"></div><section class="cards" id="cards"></section></article></section></main>
<script>
const DATA=__DATA__, colors=DATA.colors, plants=DATA.plants, summary=DATA.summary, evidence=DATA.evidence||{};
document.querySelector('h1').textContent=summary.taxonomy==='green_gem'?'PTRDMS | Green Gem observations':'PTRDMS | Four-stage observations';
// Show individual estimated coordinates by default only when a registered
// measured range is available.  Otherwise, a relative observation layout is
// clearer and does not visually overstate assumed-row-plane positions.
let coordinateMode=summary.range_source==='registered_measured_range';
const modeButton=document.createElement('button');const updateModeButton=()=>{modeButton.textContent=coordinateMode?'Mode: estimated coordinates':'Mode: relative observation layout'};modeButton.onclick=()=>{coordinateMode=!coordinateMode;updateModeButton();draw()};updateModeButton();document.querySelector('.main-panel h2').after(modeButton);
document.querySelector('#evidence-banner').textContent=(evidence.headline||'Observation display')+' Drag to rotate | scroll to zoom.';
const labels={processed_frames:'Processed frames',plant_columns:'Observation positions',detections:'Detected fruit',immature:'Immature',mature_green:'Mature green',harvest_ready:'Harvest ready',overripe_or_defective:'Overripe / defective',green_mature:'Green maturity',discoloration:'Discoloration',mature:'Maturity'};
const enabled=Object.fromEntries(Object.keys(colors).map(key=>[key,true]));document.querySelector('#filters').innerHTML=Object.entries(colors).map(([key,color])=>`<label><input type="checkbox" data-k="${key}" checked><i class="dot" style="background:${color}"></i>${labels[key]}</label>`).join('');document.querySelectorAll('#filters input').forEach(node=>node.addEventListener('change',()=>{enabled[node.dataset.k]=node.checked;draw()}));
document.querySelector('#legend').innerHTML=Object.entries(colors).map(([key,color])=>`<span><i class="dot" style="background:${color}"></i>${labels[key]}</span>`).join('')+`<span><i class="dot" style="background:#2f6f4f"></i>Visual vine</span><span><i class="dot" style="background:#9ca3af"></i>Relative layout</span>`;
const metrics=[['processed_frames',summary.processed_frames],['plant_columns',summary.plant_columns],['detections',summary.detections],...Object.entries(summary.class_counts)];document.querySelector('#cards').innerHTML=metrics.map(([k,v])=>`<div class="card"><b style="color:${colors[k]||'#173d30'}">${v}</b><span>${labels[k]||k}</span></div>`).join('');
const gallery=new Map(DATA.gallery.map(item=>[`${item.frame}|${item.side}`,item]));const map=document.querySelector('#mini-map'),W=DATA.map_size.width,H=DATA.map_size.height;let selected=null;
for(const plant of plants){const p=document.createElement('button');p.className='mini-point';p.style.left=`${Math.max(0,Math.min(100,plant.map_px/W*100))}%`;p.style.top=`${Math.max(0,Math.min(100,plant.map_py/H*100))}%`;p.dataset.id=plant.plant_id;p.title=plant.plant_id;p.addEventListener('click',()=>select(plant));map.appendChild(p)}
function select(plant){selected=plant.plant_id;const item=gallery.get(`${plant.frame}|${plant.side}`),detail=document.querySelector('#detail'),holder=document.querySelector('#placeholder');holder.hidden=true;detail.hidden=false;document.querySelector('#photo').src=item?.path||'';document.querySelector('#where').textContent=plant.plant_id;document.querySelector('#meta').textContent=`${plant.frame} | ${plant.side} side | ${plant.total} detected fruit`;document.querySelector('#counts').innerHTML=Object.entries(plant.counts).filter(([,n])=>n).map(([k,n])=>`<span><i class="dot" style="background:${colors[k]}"></i>${labels[k]} ${n}</span>`).join('')||'<span>No detected fruit</span>';document.querySelectorAll('.mini-point').forEach(p=>p.classList.toggle('selected',p.dataset.id===selected));draw()}
const canvas=document.querySelector('#scene'),ctx=canvas.getContext('2d');let yaw=-.55,zoom=1,drag=false,last=[0,0],hit=[];const minX=Math.min(...plants.map(p=>p.x)),maxX=Math.max(...plants.map(p=>p.x)),minY=Math.min(...plants.map(p=>p.y)),maxY=Math.max(...plants.map(p=>p.y));
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}function project(p,w,h,scale){const x=p.x-(minX+maxX)/2,y=p.y-(minY+maxY)/2,rx=x*Math.cos(yaw)-y*Math.sin(yaw),depth=x*Math.sin(yaw)+y*Math.cos(yaw);return{x:w/2+rx*scale,y:h*.78+depth*scale*.36-(p.z||0)*scale*.44,depth}}
function vineHeight(plant){return Math.max(1.9,.82+plant.cubes.length*.38)}function fruitPosition(plant,cube,index){if(coordinateMode)return {x:cube.x,y:cube.y,z:cube.z};const base=.52+index*.36+(Number.isFinite(cube.z)?Math.min(1.5,Math.max(0,cube.z))*.12:0),side=plant.side==='left'?1:-1,angle=index*2.17+(plant.side==='left'?.5:1.1);return{x:plant.x+side*(.25+.10*(index%3))+Math.cos(angle)*.25,y:plant.y+Math.sin(angle)*.28,z:Math.min(vineHeight(plant)-.16,base)}}
function line(a,b,color,width,dash=[]){ctx.save();ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.restore()}function sphere(q,r,color){const g=ctx.createRadialGradient(q.x-r*.35,q.y-r*.4,1,q.x,q.y,r);g.addColorStop(0,'#fff');g.addColorStop(.14,color);g.addColorStop(1,'#173d30');ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();ctx.strokeStyle='#ffffff';ctx.lineWidth=1.2;ctx.stroke()}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,scale=Math.min(w/Math.max(13,maxX-minX+5),h/10)*zoom;ctx.clearRect(0,0,w,h);ctx.fillStyle='#f5f9f5';ctx.fillRect(0,0,w,h);hit=[];const rowBase=project({x:minX-1,y:minY-1,z:0},w,h,scale);for(let i=0;i<5;i++){const a=project({x:minX-1,y:minY-1+i*(maxY-minY+2)/4,z:0},w,h,scale),b=project({x:maxX+1,y:minY-1+i*(maxY-minY+2)/4,z:0},w,h,scale);line(a,b,'#d4dfd5',1)}const ordered=[...plants].sort((a,b)=>project(a,w,h,scale).depth-project(b,w,h,scale));for(const plant of ordered){const base=project({...plant,z:0},w,h,scale),top=project({...plant,z:vineHeight(plant)},w,h,scale),isSelected=plant.plant_id===selected;if(!coordinateMode)line(base,top,isSelected?'#173d30':'#2f6f4f',isSelected?4:2.2,[4,3]);ctx.beginPath();ctx.arc(base.x,base.y,5,0,Math.PI*2);ctx.fillStyle=plant.total?'#2f6f4f':'#a7b4ac';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.4;ctx.stroke();for(let i=0;i<plant.cubes.length;i++){const cube=plant.cubes[i];if(!enabled[cube.class_name])continue;const pos=fruitPosition(plant,cube,i),fruit=project(pos,w,h,scale),branch=project({x:plant.x,y:plant.y,z:pos.z},w,h,scale);if(!coordinateMode)line(branch,fruit,isSelected?'#173d30':'#4f8d63',1.4);const radius=5+cube.confidence*4;sphere(fruit,radius,colors[cube.class_name]);hit.push({x:fruit.x,y:fruit.y,r:radius+4,plant})}ctx.fillStyle='#4b6054';ctx.font='10px Inter';ctx.fillText(plant.plant_id.replace('P',''),base.x+7,base.y+12)}ctx.fillStyle='#61756b';ctx.font='12px Inter';ctx.fillText(coordinateMode?'Estimated coordinates | '+(summary.range_source||'unverified source'):'Relative observation layout | green dashed vines are not reconstructed stems',16,24)}
function find(x,y){return [...hit].reverse().find(h=>Math.hypot(x-h.x,y-h.y)<=h.r)}canvas.addEventListener('mousedown',e=>{drag=true;last=[e.clientX,e.clientY]});window.addEventListener('mouseup',e=>{if(!drag)return;drag=false;const r=canvas.getBoundingClientRect(),item=find(e.clientX-r.left,e.clientY-r.top);if(item)select(item.plant)});canvas.addEventListener('mousemove',e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.008;last=[e.clientX,e.clientY];draw()});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.55,Math.min(2.2,zoom*(e.deltaY>0?.9:1.1)));draw()},{passive:false});window.addEventListener('resize',resize);resize();
</script></body></html>'''


def write_html_report(
    output_dir: Path,
    summary: dict[str, object],
    poses: list[Pose],
    detections: list[SpatialDetection],
    gallery: list[dict[str, object]],
    map_width_m: float,
    map_height_m: float,
    map_size: tuple[int, int],
    row_offset_m: float = 1.20,
    evidence: dict[str, object] | None = None,
) -> None:
    data = {
        "summary": summary,
        "colors": {name: CLASS_COLORS[name] for name in class_order_for([item.class_name for item in detections], summary.get("taxonomy"))},
        "points": [item.to_dict() for item in detections],
        "plants": _plant_columns(poses, detections, map_size, map_width_m, row_offset_m, summary.get("taxonomy")),
        "trajectory": [asdict(item) for item in poses],
        "gallery": gallery,
        "extent": {"x": map_width_m, "y": map_height_m},
        "map_size": {"width": map_size[0], "height": map_size[1]},
        "evidence": evidence or {},
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (output_dir / "index.html").write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
