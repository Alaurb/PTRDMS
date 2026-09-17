#!/usr/bin/env python3
"""Serve a small browser UI for reviewing YOLO detection labels.

The server operates only on a review package created under ``data/review``.
It never edits a frozen train/validation split.  Reviewers can alter boxes,
save their edits as standard YOLO labels, and mark an image as approved.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLOv8 Tomato Box Review</title>
<style>
:root { color-scheme: dark; font-family: Inter, "Segoe UI", sans-serif; background:#111827; color:#f8fafc }
* { box-sizing:border-box } body { margin:0; min-height:100vh; display:grid; grid-template-rows:auto 1fr; overflow:hidden }
header { display:flex; gap:12px; align-items:center; padding:11px 16px; background:#172033; border-bottom:1px solid #334155; flex-wrap:wrap }
h1 { font-size:16px; margin:0 12px 0 0; white-space:nowrap } .stat { font-size:13px; color:#cbd5e1; margin-right:auto }
.save-state { font-size:12px; color:#86efac; white-space:nowrap }.save-state.dirty { color:#facc15 }
button, select { border:1px solid #475569; border-radius:6px; background:#263348; color:#f8fafc; padding:7px 10px; font:inherit; cursor:pointer }
button:hover { background:#334155 } button.primary { background:#167c52; border-color:#1d9a66 } button.primary:hover { background:#0f9560 }
button.warn { color:#fecaca; border-color:#9f1239 } .layout { display:grid; grid-template-columns:minmax(0,1fr) 305px; min-height:0 }
.stage { position:relative; background:#030712; min-height:0; overflow:hidden; display:grid; place-items:center }
canvas { display:block; max-width:100%; max-height:100%; cursor:crosshair }.side { background:#172033; border-left:1px solid #334155; padding:14px; overflow:auto }
.side h2 { font-size:14px; margin:4px 0 12px }.file { font-size:12px; line-height:1.5; color:#cbd5e1; overflow-wrap:anywhere }.hint { font-size:12px; line-height:1.55; color:#94a3b8 }
.actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:14px 0 }.actions .wide { grid-column:1/-1 }.notice { min-height:22px; color:#86efac; font-size:12px }
.box-list { margin:8px 0; padding:0; list-style:none; max-height:230px; overflow:auto }.box-list li { padding:6px 7px; border-bottom:1px solid #334155; font-size:12px; cursor:pointer; color:#cbd5e1 }
.box-list li.selected { background:#374151; color:#fef08a }.legend { display:flex; gap:10px; font-size:11px; color:#cbd5e1; margin:10px 0 }.dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px }.green{background:#22c55e}.yellow{background:#facc15}
@media(max-width:850px){ .layout{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}.side{border-left:0;border-top:1px solid #334155;max-height:40vh} }
</style></head><body>
<header><h1>YOLOv8 tomato-box review</h1><span class="stat" id="stat">Loading…</span><span class="save-state" id="saveState">Auto-save ready</span><select id="filter"><option value="all">All images</option><option value="pending">Pending only</option><option value="approved">Approved only</option></select><button id="prev">← Previous</button><button id="next">Next →</button></header>
<main class="layout"><section class="stage" id="stage"><canvas id="canvas"></canvas></section><aside class="side">
<h2>Review this image</h2><div class="file" id="filename"></div><div class="notice" id="notice"></div>
<div class="actions"><button class="wide primary" id="approve">Save, approve & next</button><button class="wide" id="addMode">Add-box mode: off</button><button id="save">Save edits</button><button id="pending">Mark pending</button><button class="warn wide" id="delete">Delete selected box</button><button class="warn wide" id="clear">Clear all boxes</button></div>
<div class="legend"><span><i class="dot green"></i>box</span><span><i class="dot yellow"></i>selected</span></div><strong id="boxCount"></strong><ul class="box-list" id="boxList"></ul>
<p class="hint">Changes are saved automatically after each edit and before switching images. If the connection fails, the browser keeps a local draft and warns before refresh/close. Drag inside a box to move it. Drag a corner to resize. Turn on <em>Add-box mode</em> to draw a new box anywhere, even over an existing box; holding <kbd>Shift</kbd> also adds one box temporarily. Click a list item to select it. <kbd>Delete</kbd> removes the selected box.</p>
</aside></main>
<script>
const api = '/api/'; let allItems=[], visibleItems=[], pos=0, current=null, boxes=[], selected=-1, image=new Image(), draw=null, interaction=null, addMode=false, dirty=false, saveTimer=null;
const canvas=document.querySelector('#canvas'), ctx=canvas.getContext('2d'), stage=document.querySelector('#stage');
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function message(text, error=false){const e=document.querySelector('#notice');e.textContent=text;e.style.color=error?'#fca5a5':'#86efac';setTimeout(()=>{if(e.textContent===text)e.textContent=''},2600)}
function draftKey(name){return 'tomato-review-draft:'+name}
function setSaveState(text, pending=false){let e=document.querySelector('#saveState');e.textContent=text;e.classList.toggle('dirty',pending)}
function markDirty(){if(!current)return;dirty=true;localStorage.setItem(draftKey(current.image),JSON.stringify(boxes));setSaveState('Unsaved — auto-saving…',true);clearTimeout(saveTimer);saveTimer=setTimeout(()=>persist(current.audit_status,false),550)}
async function request(path, options={}){let r=await fetch(api+path,options);let d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
async function loadItems(){const d=await request('items');allItems=d.items;applyFilter();document.querySelector('#stat').textContent=`${d.progress.approved}/${d.progress.total} approved · ${d.progress.pending} pending`;}
function applyFilter(){let wanted=document.querySelector('#filter').value;visibleItems=allItems.filter(x=>wanted==='all'||x.audit_status===wanted);pos=clamp(pos,0,Math.max(0,visibleItems.length-1));loadCurrent();}
function loadCurrent(){current=visibleItems[pos]||null;boxes=[];selected=-1;dirty=false;clearTimeout(saveTimer);if(!current){document.querySelector('#filename').textContent='No images in this filter.';setSaveState('Auto-save ready');render();return}document.querySelector('#filename').textContent=`${pos+1}/${visibleItems.length} · ${current.image} · ${current.audit_status}`;image.onload=()=>render();image.src=api+'image/'+encodeURIComponent(current.image)+'?t='+Date.now();boxes=current.boxes.map(b=>({...b}));try{let saved=localStorage.getItem(draftKey(current.image));if(saved){let recovered=JSON.parse(saved);if(Array.isArray(recovered)){boxes=recovered;dirty=true;setSaveState('Recovered local draft — auto-saving…',true);saveTimer=setTimeout(()=>persist(current.audit_status,false),250)}}else setSaveState('Saved')}catch(e){localStorage.removeItem(draftKey(current.image));setSaveState('Saved')}renderList();}
function fit(){if(!image.naturalWidth)return {x:0,y:0,w:1,h:1};let W=stage.clientWidth,H=stage.clientHeight, s=Math.min(W/image.naturalWidth,H/image.naturalHeight),w=image.naturalWidth*s,h=image.naturalHeight*s;return{x:(W-w)/2,y:(H-h)/2,w,h}}
function setCanvas(){let r=stage.getBoundingClientRect(), ratio=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(r.width*ratio));canvas.height=Math.max(1,Math.round(r.height*ratio));canvas.style.width=r.width+'px';canvas.style.height=r.height+'px';ctx.setTransform(ratio,0,0,ratio,0,0)}
function screenBox(b,f){return{x:f.x+(b.x-b.w/2)*f.w,y:f.y+(b.y-b.h/2)*f.h,w:b.w*f.w,h:b.h*f.h}}
function render(){setCanvas();let f=fit();ctx.clearRect(0,0,stage.clientWidth,stage.clientHeight);if(!image.naturalWidth)return;ctx.drawImage(image,f.x,f.y,f.w,f.h);boxes.forEach((b,i)=>{let q=screenBox(b,f), on=i===selected;ctx.strokeStyle=on?'#facc15':'#22c55e';ctx.lineWidth=on?3:2;ctx.strokeRect(q.x,q.y,q.w,q.h);ctx.fillStyle=on?'#facc15':'#22c55e';ctx.font='bold 13px sans-serif';ctx.fillText(String(i+1),q.x+3,Math.max(14,q.y-4));if(on){ctx.fillRect(q.x-4,q.y-4,8,8);ctx.fillRect(q.x+q.w-4,q.y-4,8,8);ctx.fillRect(q.x-4,q.y+q.h-4,8,8);ctx.fillRect(q.x+q.w-4,q.y+q.h-4,8,8)}});if(draw){let x=Math.min(draw.x1,draw.x2),y=Math.min(draw.y1,draw.y2),w=Math.abs(draw.x2-draw.x1),h=Math.abs(draw.y2-draw.y1);ctx.strokeStyle='#60a5fa';ctx.setLineDash([5,4]);ctx.strokeRect(f.x+x*f.w,f.y+y*f.h,w*f.w,h*f.h);ctx.setLineDash([])}}
function renderList(){let ul=document.querySelector('#boxList');ul.innerHTML='';boxes.forEach((b,i)=>{let li=document.createElement('li');li.textContent=`#${i+1}  x=${b.x.toFixed(3)}  y=${b.y.toFixed(3)}  w=${b.w.toFixed(3)}  h=${b.h.toFixed(3)}`;if(i===selected)li.className='selected';li.onclick=()=>{selected=i;renderList();render()};ul.append(li)});document.querySelector('#boxCount').textContent=`${boxes.length} tomato boxes`;}
function pointer(e){let f=fit(),r=canvas.getBoundingClientRect();return{x:clamp((e.clientX-r.left-f.x)/f.w,0,1),y:clamp((e.clientY-r.top-f.y)/f.h,0,1)}}
function hit(p){let f=fit();for(let i=boxes.length-1;i>=0;i--){let q=screenBox(boxes[i],f), sx=f.x+p.x*f.w,sy=f.y+p.y*f.h;let corner=(Math.abs(sx-q.x)<10?'w':Math.abs(sx-(q.x+q.w))<10?'e':'')+(Math.abs(sy-q.y)<10?'n':Math.abs(sy-(q.y+q.h))<10?'s':'');if(corner)return{i,kind:corner};if(sx>=q.x&&sx<=q.x+q.w&&sy>=q.y&&sy<=q.y+q.h)return{i,kind:'move'}}return null}
canvas.addEventListener('pointerdown',e=>{if(!current)return;canvas.setPointerCapture(e.pointerId);let p=pointer(e),h=(addMode||e.shiftKey)?null:hit(p);if(h){selected=h.i;interaction={...h,start:p,original:{...boxes[h.i]}}}else draw={x1:p.x,y1:p.y,x2:p.x,y2:p.y};renderList();render()});
canvas.addEventListener('pointermove',e=>{if(!interaction&&!draw)return;let p=pointer(e);if(draw){draw.x2=p.x;draw.y2=p.y}else{let b=interaction.original,dx=p.x-interaction.start.x,dy=p.y-interaction.start.y;if(interaction.kind==='move'){boxes[interaction.i]={...b,x:clamp(b.x+dx,b.w/2,1-b.w/2),y:clamp(b.y+dy,b.h/2,1-b.h/2)}}else{let x1=b.x-b.w/2,y1=b.y-b.h/2,x2=b.x+b.w/2,y2=b.y+b.h/2;if(interaction.kind.includes('w'))x1=p.x;if(interaction.kind.includes('e'))x2=p.x;if(interaction.kind.includes('n'))y1=p.y;if(interaction.kind.includes('s'))y2=p.y;x1=clamp(x1,0,.995);y1=clamp(y1,0,.995);x2=clamp(x2,x1+.005,1);y2=clamp(y2,y1+.005,1);boxes[interaction.i]={x:(x1+x2)/2,y:(y1+y2)/2,w:x2-x1,h:y2-y1}}}render();});
canvas.addEventListener('pointerup',e=>{let changed=false;if(draw){let w=Math.abs(draw.x2-draw.x1),h=Math.abs(draw.y2-draw.y1);if(w>.005&&h>.005){boxes.push({x:(draw.x1+draw.x2)/2,y:(draw.y1+draw.y2)/2,w,h});selected=boxes.length-1;changed=true}draw=null}else if(interaction){interaction=null;changed=true}renderList();render();if(changed)markDirty()});
async function persist(status, announce=true){if(!current)return false;clearTimeout(saveTimer);let subject=current.image,draft=boxes.map(b=>({...b}));if(!dirty&&status===current.audit_status){if(announce)message('Already saved.');return true}try{setSaveState('Saving…',true);let d=await request('item/'+encodeURIComponent(subject),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({boxes:draft,audit_status:status})});let found=allItems.find(x=>x.image===subject);if(found)Object.assign(found,d.item);if(current&&current.image===subject){current=found||d.item;dirty=false;localStorage.removeItem(draftKey(subject));setSaveState('Saved')}if(announce)message(status==='approved'?'Saved and approved.':'Saved.');return true}catch(e){if(current&&current.image===subject){dirty=true;localStorage.setItem(draftKey(subject),JSON.stringify(draft));setSaveState('Save failed — local draft retained',true)}message(e.message,true);return false}}
async function save(status){if(!await persist(status,true))return;await loadItems();if(status==='approved'&&document.querySelector('#filter').value!=='pending'&&pos<visibleItems.length-1)pos++;loadCurrent()}
async function navigate(delta){if(!await persist(current&&current.audit_status,false))return;pos=clamp(pos+delta,0,Math.max(0,visibleItems.length-1));loadCurrent()}
document.querySelector('#prev').onclick=()=>navigate(-1);document.querySelector('#next').onclick=()=>navigate(1);document.querySelector('#addMode').onclick=()=>{addMode=!addMode;let b=document.querySelector('#addMode');b.textContent=addMode?'Add-box mode: on — drag anywhere':'Add-box mode: off';b.classList.toggle('primary',addMode);canvas.style.cursor=addMode?'crosshair':'default'};document.querySelector('#save').onclick=()=>save(current.audit_status);document.querySelector('#pending').onclick=()=>save('pending');document.querySelector('#approve').onclick=()=>save('approved');document.querySelector('#delete').onclick=()=>{if(selected>=0){boxes.splice(selected,1);selected=-1;renderList();render();markDirty()}};document.querySelector('#clear').onclick=()=>{if(confirm('Remove all boxes from this image?')){boxes=[];selected=-1;renderList();render();markDirty()}};document.querySelector('#filter').onchange=async()=>{if(!await persist(current&&current.audit_status,false))return;pos=0;applyFilter()};window.addEventListener('resize',render);window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='Your current label is still being saved.'}});window.addEventListener('keydown',e=>{if((e.key==='Delete'||e.key==='Backspace')&&selected>=0&&document.activeElement===document.body){boxes.splice(selected,1);selected=-1;renderList();render();markDirty()}if(e.key==='a'&&document.activeElement===document.body)document.querySelector('#addMode').click()});loadItems().catch(e=>message(e.message,true));
</script></body></html>"""


class ReviewStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = self.root / "audit_manifest.csv"
        self.images = self.root / "images"
        self.labels = self.root / "labels_to_review"
        if not self.manifest.is_file() or not self.images.is_dir() or not self.labels.is_dir():
            raise ValueError("review root must contain audit_manifest.csv, images/, and labels_to_review/")

    def rows(self) -> list[dict[str, str]]:
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def boxes(label: Path) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        if not label.exists():
            return output
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            # Initial predictions may contain a sixth trailing confidence value
            # because they were exported with Ultralytics ``save_conf=True``.
            # The reviewer edits geometry only; saving normalizes to 5-column
            # training labels.
            if len(fields) not in {5, 6} or fields[0] != "0":
                raise ValueError(f"invalid YOLO label at {label.name}:{line_number}")
            x, y, width, height = map(float, fields[1:5])
            output.append({"x": x, "y": y, "w": width, "h": height})
        return output

    def items(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for row in self.rows():
            name = row["image"]
            image = self.images / name
            if not image.is_file() or Path(name).name != name:
                continue
            result.append({**row, "boxes": self.boxes(self.labels / f"{image.stem}.txt")})
        return result

    def assert_name(self, name: str) -> tuple[Path, Path]:
        if Path(name).name != name:
            raise ValueError("invalid image name")
        image = self.images / name
        if not image.is_file():
            raise FileNotFoundError(name)
        return image, self.labels / f"{image.stem}.txt"

    def save(self, name: str, boxes: object, status: object) -> dict[str, object]:
        image, label = self.assert_name(name)
        if status not in {"pending", "approved"}:
            raise ValueError("audit status must be pending or approved")
        if not isinstance(boxes, list) or len(boxes) > 300:
            raise ValueError("boxes must be a list of at most 300 entries")
        checked: list[dict[str, float]] = []
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError("each box must be an object")
            try:
                checked_box = {key: float(box[key]) for key in ("x", "y", "w", "h")}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("each box needs numeric x, y, w, h") from error
            x, y, width, height = (checked_box[key] for key in ("x", "y", "w", "h"))
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0.002 <= width <= 1 and 0.002 <= height <= 1 and x - width / 2 >= 0 and y - height / 2 >= 0 and x + width / 2 <= 1 and y + height / 2 <= 1):
                raise ValueError("box is outside the image or too small")
            checked.append(checked_box)
        self.atomic_write(label, "".join(f"0 {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}\n" for box in checked))
        rows = self.rows()
        item: dict[str, object] | None = None
        for row in rows:
            if row["image"] == image.name:
                row["initial_box_count"] = str(len(checked))
                row["audit_status"] = str(status)
                item = {**row, "boxes": checked}
                break
        if item is None:
            raise ValueError("image is absent from audit manifest")
        fieldnames = list(rows[0])
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=self.manifest.parent, delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temp_name = handle.name
        os.replace(temp_name, self.manifest)
        return item

    @staticmethod
    def atomic_write(target: Path, content: str) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            handle.write(content)
            temp_name = handle.name
        os.replace(temp_name, target)


def handler_factory(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                if path == "/" or path == "/index.html":
                    body = PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/items":
                    items = store.items()
                    approved = sum(item["audit_status"] == "approved" for item in items)
                    self.json_response(HTTPStatus.OK, {"items": items, "progress": {"total": len(items), "approved": approved, "pending": len(items) - approved}})
                    return
                if path.startswith("/api/image/"):
                    image, _ = store.assert_name(unquote(path[len("/api/image/") :]))
                    body = image.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mimetypes.guess_type(image.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (FileNotFoundError, ValueError) as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                if not path.startswith("/api/item/"):
                    self.json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                item = store.save(unquote(path[len("/api/item/") :]), payload.get("boxes"), payload.get("audit_status"))
                self.json_response(HTTPStatus.OK, {"item": item})
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, fmt: str, *args: object) -> None:
            print("[review] " + fmt % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1", help="default: localhost only")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    store = ReviewStore(args.review_root)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(store))
    print(f"Review UI: http://{args.host}:{args.port}")
    print(f"Review root: {store.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
