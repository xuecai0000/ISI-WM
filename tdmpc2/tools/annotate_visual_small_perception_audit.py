"""Build a self-contained point-annotation page for perception audits.

The generated HTML embeds only RGB audit frames. It never reads simulator
state, rewards, test videos, or model predictions. The annotator labels the
three moving/task-relevant roles; the fixed base is set to the image centre.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FORMAT = "visual_small_perception_audit_v1"
ROLES = ("base", "elbow", "control_tip", "goal")
ANNOTATED_TIMES = (0, 5, 10, 15)
FORBIDDEN_SPLITS = {"test"}
ALLOWED_SPLITS = {"train", "validation"}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Create an offline HTML point annotator for a perception audit."
	)
	parser.add_argument("--audit", type=Path, required=True)
	parser.add_argument("--output", type=Path, default=None)
	return parser.parse_args()


def _read_payload(path):
	with path.open("r", encoding="utf-8") as file:
		payload = json.load(file)
	if payload.get("format") != FORMAT:
		raise ValueError(f"Expected format={FORMAT!r}, got {payload.get('format')!r}.")
	if tuple(payload.get("roles", ())) != ROLES:
		raise ValueError(f"Audit roles must be ordered as {ROLES}.")
	sequences = payload.get("sequences")
	if not isinstance(sequences, list) or not sequences:
		raise ValueError("Audit must contain a non-empty sequences list.")
	return payload


def _validated_items(payload, audit_path):
	items = []
	seen_images = set()
	for sequence_index, sequence in enumerate(payload["sequences"]):
		split = str(sequence.get("split", ""))
		if split in FORBIDDEN_SPLITS or split not in ALLOWED_SPLITS:
			raise ValueError(
				f"Perception annotation permits train/validation only, got {split!r}."
			)
		frames = sequence.get("frames")
		if not isinstance(frames, list) or len(frames) != 16:
			raise ValueError(
				f"Sequence {sequence_index} must contain exactly 16 ordered frames."
			)
		for expected_t, frame in enumerate(frames):
			if int(frame.get("t", -1)) != expected_t:
				raise ValueError(
					f"Sequence {sequence_index} frame order is not t=0..15."
				)
			if expected_t not in ANNOTATED_TIMES:
				continue
			image_rel = frame.get("image")
			if not isinstance(image_rel, str) or not image_rel:
				raise ValueError("Every annotated frame must have a relative image path.")
			image_path = (audit_path.parent / image_rel).resolve()
			try:
				image_path.relative_to(audit_path.parent.resolve())
			except ValueError as error:
				raise ValueError(f"Image escapes the audit directory: {image_rel}") from error
			if not image_path.is_file():
				raise FileNotFoundError(image_path)
			image_bytes = image_path.read_bytes()
			decoded_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
			if decoded_rgb.shape != (64, 64, 3):
				raise ValueError(
					f"Annotated frame must be decoded 64x64 RGB, got "
					f"{decoded_rgb.shape}: {image_rel}"
				)
			actual_rgb_sha = hashlib.sha256(
				np.ascontiguousarray(decoded_rgb).tobytes()
			).hexdigest()
			recorded_rgb_sha = frame.get("image_sha256")
			if not isinstance(recorded_rgb_sha, str) or recorded_rgb_sha != actual_rgb_sha:
				raise ValueError(f"Decoded-RGB SHA mismatch: {image_rel}")
			if image_rel in seen_images:
				raise ValueError(f"Duplicate annotated image path: {image_rel}")
			seen_images.add(image_rel)
			suffix = image_path.suffix.lower()
			mime = "image/png" if suffix == ".png" else "image/jpeg"
			items.append(
				{
					"sequence_index": sequence_index,
					"frame_index": expected_t,
					"split": split,
					"episode": sequence.get("episode"),
					"source": sequence.get("source"),
					"source_frame_index": frame.get("source_frame_index"),
					"image": image_rel,
					"image_sha256": actual_rgb_sha,
					"data_uri": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}",
					"points": frame.get("points", {}),
				}
			)
	if not items:
		raise ValueError("No annotation frames were found.")
	return items


def _html(payload, items, audit_sha):
	audit_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
	items_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
	return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Small 感知审计标注</title>
<style>
body {{ margin:0; background:#111827; color:#e5e7eb; font-family:system-ui,sans-serif; }}
main {{ max-width:1040px; margin:auto; padding:18px; display:grid; grid-template-columns:minmax(520px,640px) 1fr; gap:22px; }}
canvas {{ width:640px; height:640px; max-width:100%; image-rendering:pixelated; background:#000; cursor:crosshair; border:1px solid #475569; }}
.panel {{ background:#1f2937; padding:16px; border-radius:12px; align-self:start; }}
button {{ margin:4px 4px 4px 0; padding:9px 12px; border:0; border-radius:7px; cursor:pointer; }}
.primary {{ background:#22c55e; color:#052e16; font-weight:700; }}
.warn {{ background:#f59e0b; color:#451a03; }}
.role {{ font-size:1.2rem; font-weight:700; color:#67e8f9; }}
.small {{ font-size:.86rem; color:#94a3b8; word-break:break-all; }}
.progress {{ height:8px; background:#374151; border-radius:8px; overflow:hidden; margin:10px 0; }}
.bar {{ height:100%; background:#22c55e; width:0; }}
@media(max-width:900px) {{ main {{ grid-template-columns:1fr; }} canvas {{ width:100%; height:auto; aspect-ratio:1; }} }}
</style>
</head>
<body><main>
<canvas id="canvas" width="640" height="640"></canvas>
<section class="panel">
<h2>Visual Small 感知审计</h2>
<div id="meta" class="small"></div>
<div class="progress"><div id="bar" class="bar"></div></div>
<p>当前点击：<span id="role" class="role"></span></p>
<p>基座自动设为 [31.5, 31.5]。请依次点击：肘关节、控制尖端、目标中心。点击会量化到 0.5 像素。</p>
<div>
<button id="undo" class="warn">撤销本张</button>
<button id="prev">上一张</button>
<button id="next" class="primary">确认并下一张</button>
</div>
<div><button id="export" class="primary">导出已完成 annotations JSON</button></div>
<p id="status"></p>
<p class="small">只标 RGB；页面不含模型预测、reward、physics 或 test split。标签会保存在本浏览器 localStorage，同时只有点击“导出”才生成文件。</p>
</section>
</main>
<script>
const AUDIT_SHA = {json.dumps(audit_sha)};
const STORAGE_KEY = 'visual-small-perception-audit-' + AUDIT_SHA;
const audit = {audit_json};
const items = {items_json};
const roleOrder = ['elbow','control_tip','goal'];
const colors = {{base:'#22c55e',elbow:'#f0abfc',control_tip:'#22d3ee',goal:'#ef4444'}};
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch (_) {{ saved = {{}}; }}
let current = 0;
let image = new Image();
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function key(item) {{ return item.sequence_index + ':' + item.frame_index; }}
function normalizeExisting(item) {{
  const p = item.points || {{}};
  if (p.elbow && p.control_tip && p.goal) return {{base:p.base || [31.5,31.5], elbow:p.elbow, control_tip:p.control_tip, goal:p.goal}};
  return {{base:[31.5,31.5]}};
}}
function labels(item) {{
  if (!saved[key(item)]) saved[key(item)] = normalizeExisting(item);
  return saved[key(item)];
}}
function nextRole(item) {{ const p=labels(item); return roleOrder.find(r => !p[r]) || null; }}
function drawCross(point, color, name) {{
  const s=10, x=point[0]*10, y=point[1]*10;
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=3;
  ctx.beginPath(); ctx.moveTo(x-s,y); ctx.lineTo(x+s,y); ctx.moveTo(x,y-s); ctx.lineTo(x,y+s); ctx.stroke();
  ctx.font='bold 15px system-ui'; ctx.fillText(name+' '+JSON.stringify(point), Math.min(x+12,470), Math.max(y-10,20));
}}
function redraw() {{
  ctx.imageSmoothingEnabled=false; ctx.clearRect(0,0,640,640); ctx.drawImage(image,0,0,640,640);
  const p=labels(items[current]); Object.keys(p).forEach(r => drawCross(p[r],colors[r],r));
  const role=nextRole(items[current]); document.getElementById('role').textContent=role || '已完成';
  document.getElementById('bar').style.width=((current+1)/items.length*100)+'%';
  const it=items[current];
  document.getElementById('meta').textContent=`${{current+1}}/${{items.length}} | ${{it.split}} | episode=${{it.episode}} | t=${{it.frame_index}} | ${{it.source}}:${{it.source_frame_index}}`;
  const done=items.filter(it => roleOrder.every(r => labels(it)[r])).length;
  document.getElementById('status').textContent=`已完成 ${{done}} / ${{items.length}} 张`;
  localStorage.setItem(STORAGE_KEY,JSON.stringify(saved));
}}
function loadCurrent() {{ image.onload=redraw; image.src=items[current].data_uri; }}
canvas.addEventListener('click', event => {{
  const role=nextRole(items[current]); if (!role) return;
  const rect=canvas.getBoundingClientRect();
  const rawX=(event.clientX-rect.left)/rect.width*64;
  const rawY=(event.clientY-rect.top)/rect.height*64;
  const q=v=>Math.max(0,Math.min(63.5,Math.round(v*2)/2));
  labels(items[current])[role]=[q(rawX),q(rawY)]; redraw();
}});
document.getElementById('undo').onclick=()=>{{ saved[key(items[current])]={{base:[31.5,31.5]}}; redraw(); }};
document.getElementById('prev').onclick=()=>{{ current=Math.max(0,current-1); loadCurrent(); }};
document.getElementById('next').onclick=()=>{{
  if (nextRole(items[current])) {{ alert('请先完成肘、尖端和目标三个点。'); return; }}
  current=Math.min(items.length-1,current+1); loadCurrent();
}};
document.getElementById('export').onclick=()=>{{
  const incomplete=items.filter(it => roleOrder.some(r => !labels(it)[r]));
  if (incomplete.length) {{ alert(`还有 ${{incomplete.length}} 张未完成。`); return; }}
  const out=JSON.parse(JSON.stringify(audit));
  items.forEach(it => {{
    const frame=out.sequences[it.sequence_index].frames[it.frame_index];
    frame.points=labels(it);
    frame.annotation={{policy:'manual_rgb_only',roles:['base','elbow','control_tip','goal']}};
  }});
  out.annotation={{status:'complete',policy:'manual_rgb_only',audit_sha256:AUDIT_SHA,annotated_times:[0,5,10,15]}};
  const blob=new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='annotations.completed.json'; a.click(); URL.revokeObjectURL(a.href);
}};
loadCurrent();
</script></body></html>"""


def main():
	args = parse_args()
	audit_path = args.audit.expanduser().resolve()
	payload = _read_payload(audit_path)
	items = _validated_items(payload, audit_path)
	audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
	output = (
		args.output.expanduser().resolve()
		if args.output is not None
		else audit_path.with_name("annotate_points.html")
	)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(_html(payload, items, audit_sha), encoding="utf-8")
	print(
		"PERCEPTION_ANNOTATOR_READY",
		json.dumps(
			{
				"audit": str(audit_path),
				"output": str(output),
				"frames_to_annotate": len(items),
				"splits": sorted({item["split"] for item in items}),
				"test_frames": 0,
			},
			ensure_ascii=False,
		),
	)


if __name__ == "__main__":
	main()
