"""Browser UI for the shlee-branch 3rd trial and main-branch 4th trial."""
from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import types
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template_string, request


ROOT = Path(__file__).resolve().parent
FOURTH_PATH = ROOT / "결과" / "4차 시도" / "rebar_grid_counter.py"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
PENDING_IMAGES: dict[str, bytes] = {}
PENDING_RESULTS: dict[str, dict[str, dict]] = {}


def load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_shlee_third_trial() -> types.ModuleType:
    """Load the 3rd-trial source from the shlee branch itself.

    The current main branch contains a different 3rd-trial file, so using
    ``git show`` here prevents the web app from silently mixing methodologies.
    A checked-out file is used as a fallback for copied, non-git deployments.
    """
    source_path = "결과/3차 시도/rebar_grid_counter.py"
    try:
        source = subprocess.check_output(
            ["git", "show", f"shlee:{source_path}"],
            cwd=ROOT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return load_module(ROOT / source_path, "shlee_third_trial_fallback")

    module = types.ModuleType("shlee_third_trial")
    module.__file__ = f"shlee:{source_path}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


third = load_shlee_third_trial()
fourth = load_module(FOURTH_PATH, "main_fourth_trial")
app = Flask(__name__)


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rebar Grid Lab</title>
  <style>
    :root{--ink:#172033;--muted:#697386;--line:#e2e7f0;--blue:#5367f5;--bg:#f5f7fb}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif}
    .wrap{max-width:1240px;margin:auto;padding:42px 20px 64px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.14em}
    h1{font-size:clamp(30px,5vw,52px);line-height:1.05;margin:10px 0 12px;letter-spacing:-.05em}.lead{color:var(--muted);margin:0 0 28px}
    .panel,.card{background:white;border:1px solid var(--line);border-radius:20px;box-shadow:0 12px 36px #22304c0d}.panel{padding:24px;margin-bottom:22px}
    .formgrid{display:grid;grid-template-columns:1fr 1.3fr;gap:18px}.field label{display:block;font-weight:750;margin-bottom:8px}.field small,.hint{color:var(--muted);font-size:13px}
    input[type=file],input[type=text]{width:100%;padding:12px;border:1px solid #d5dce8;border-radius:11px;background:#fbfcff;font:inherit}button{border:0;border-radius:11px;padding:13px 20px;background:var(--blue);color:#fff;font-weight:800;cursor:pointer;margin-top:18px}
    button:hover{filter:brightness(.94)}button:disabled{opacity:.45;cursor:not-allowed;filter:none}.roi-actions{display:flex;gap:28px;align-items:center;margin-top:20px}.roi-actions form{margin:0}.roi-actions button{min-width:150px;margin:0}.ready{border-color:#cfd6ff;background:#fbfcff}.error{padding:14px;border:1px solid #f2c3c3;background:#fff2f2;color:#9b2d2d;border-radius:12px;margin-bottom:20px}
    .results{display:block}.results-tabs{width:100%}.tab-buttons{display:flex;gap:10px;margin-bottom:14px;border-bottom:1px solid var(--line);padding-bottom:12px}.tab-button{margin:0;background:#e9edf5;color:#47536a;min-width:150px}.tab-button.active{background:var(--blue);color:#fff}.card{width:100%;padding:22px}.tag{display:inline-block;background:#eef0ff;color:var(--blue);border-radius:999px;padding:5px 10px;font-size:12px;font-weight:800;margin-bottom:10px}
    .card h2{font-size:21px;margin:0 0 6px}.counts{display:flex;gap:10px;margin:16px 0}.count{flex:1;padding:13px;background:#f5f7fb;border-radius:13px}.count b{display:block;font-size:30px}.count span{font-size:12px;color:var(--muted)}
    img{width:100%;display:block;border-radius:14px;background:#edf0f6}.meta{margin-top:14px}.meta summary{cursor:pointer;color:var(--muted);font-size:13px}pre{white-space:pre-wrap;overflow:auto;font-size:11px;color:#536075}
    .note{margin-top:20px;color:var(--muted);font-size:13px}.badge{color:#16805c;font-weight:750}
    @media(max-width:760px){.formgrid,.results{grid-template-columns:1fr}.wrap{padding-top:28px}}
  </style>
</head>
<body><main class="wrap">
  <div class="eyebrow">COMPUTER VISION · METHODOLOGY COMPARISON</div>
  <h1>철근망 카운터</h1>
  <p class="lead">분석 방법을 선택한 뒤 철근망 개수를 확인합니다.</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <section class="panel"><form id="upload-form" method="post" enctype="multipart/form-data">
    <div class="formgrid">
      <div class="field"><label for="image">분석할 이미지</label><input id="image" type="file" name="image" accept="image/*" required><div class="hint">JPG, PNG 등 · 최대 16MB</div></div>
      <div class="field"><label>이미지 준비</label><div class="hint">이미지를 한 번 업로드한 후, 같은 이미지에 수동 ROI와 자동 ROI를 각각 실행할 수 있습니다.</div></div>
    </div><input type="hidden" name="mode" value="upload"><button type="submit" id="upload-button">이미지 업로드</button>
  </form></section>
  {% if image_token %}<section class="panel ready"><strong>이미지가 준비되었습니다.</strong><p class="hint">같은 이미지를 사용해 원하는 방법론을 각각 실행하세요. 한쪽 결과를 본 뒤 다른 버튼을 눌러도 이미지를 다시 첨부할 필요가 없습니다.</p><div class="roi-actions"><form method="post" action="/run"><input type="hidden" name="image_token" value="{{ image_token }}"><button type="submit" name="mode" value="manual">수동 ROI</button></form><form method="post" action="/run"><input type="hidden" name="image_token" value="{{ image_token }}"><button type="submit" name="mode" value="auto">자동 ROI</button></form></div></section>{% endif %}
  {% if results %}<section class="results"><div class="results-tabs"><div class="tab-buttons">{% for item in results %}<button type="button" class="tab-button{% if loop.first %} active{% endif %}" data-tab="result-{{ loop.index }}">{{ item.label }}</button>{% endfor %}</div>
  {% for item in results %}<article class="card result-pane" id="result-{{ loop.index }}"{% if not loop.first %} hidden{% endif %}>
    <span class="tag">{{ item.label }}</span><h2>{{ item.title }}</h2><p class="hint">{{ item.description }}</p>
    <div class="counts"><div class="count"><b>{{ item.horizontal }}</b><span>가로 방향</span></div><div class="count"><b>{{ item.vertical }}</b><span>세로 방향</span></div></div>
    <img src="data:image/png;base64,{{ item.image }}" alt="{{ item.title }} 결과">
    <details class="meta"><summary>처리 정보</summary><pre>{{ item.details }}</pre></details>
  </article>{% endfor %}</div></section><script>document.querySelectorAll('.tab-button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.tab-button').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.result-pane').forEach(p=>p.hidden=true);button.classList.add('active');document.getElementById(button.dataset.tab).hidden=false;}));</script>{% endif %}
  <p class="note"><span class="badge">방법론 보존</span> 3차시도는 shlee 브랜치 소스를, 4차시도는 현재 main 브랜치 소스를 직접 사용합니다.</p>
</main></body></html>"""


ROI_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>수동 ROI 지정 · Rebar Grid Lab</title>
<style>
:root{--ink:#172033;--muted:#697386;--line:#e2e7f0;--blue:#5367f5;--bg:#f5f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif}
.wrap{max-width:1100px;margin:auto;padding:38px 20px 64px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.14em}h1{margin:10px 0;font-size:clamp(28px,5vw,46px);letter-spacing:-.05em}.lead,.hint{color:var(--muted);font-size:14px}.panel{background:#fff;border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 12px 36px #22304c0d}.canvas-box{overflow:auto;background:#edf0f6;border-radius:14px;margin:18px 0}.canvas-box canvas{display:block;max-width:100%;height:auto;cursor:crosshair}.buttons{display:flex;gap:10px;flex-wrap:wrap}button{border:0;border-radius:11px;padding:13px 20px;background:var(--blue);color:#fff;font-weight:800;cursor:pointer}button.secondary{background:#e9edf5;color:#47536a}.error{padding:14px;border:1px solid #f2c3c3;background:#fff2f2;color:#9b2d2d;border-radius:12px;margin-bottom:20px}
</style></head><body><main class="wrap"><div class="eyebrow">MANUAL ROI · 3RD TRIAL</div><h1>철근망 외곽점 지정</h1><p class="lead">철근망의 네 모서리를 순서대로 클릭하세요: 좌상단 → 우상단 → 우하단 → 좌하단</p>
{% if error %}<div class="error">{{ error }}</div>{% endif %}<section class="panel"><div class="canvas-box"><canvas id="canvas"></canvas></div><p class="hint"><span id="count">0</span>/4개 선택됨 · 잘못 찍었으면 초기화하세요.</p>
<form method="post" action="/analyze-manual"><input type="hidden" name="image_token" value="{{ image_token }}"><input type="hidden" id="points" name="points"><div class="buttons"><button type="submit" id="submit" disabled>이 점으로 분석하기</button><button type="button" class="secondary" id="reset">초기화</button><button type="button" class="secondary" onclick="location.href='/'">처음으로</button></div></form></section></main>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),img=new Image(),points=[];
img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0)};
img.src='data:image/png;base64,{{ image_data }}';
function redraw(){ctx.drawImage(img,0,0);points.forEach((p,i)=>{ctx.fillStyle='#e53935';ctx.beginPath();ctx.arc(p[0],p[1],Math.max(7,canvas.width/140),0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='bold '+Math.max(16,canvas.width/55)+'px sans-serif';ctx.fillText(i+1,p[0]+10,p[1]-10)});document.getElementById('count').textContent=points.length;document.getElementById('submit').disabled=points.length!==4;document.getElementById('points').value=points.map(p=>p.map(v=>Math.round(v)).join(',')).join(' ')}
canvas.addEventListener('click',e=>{if(points.length>=4)return;const r=canvas.getBoundingClientRect();points.push([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]);redraw()});
document.getElementById('reset').onclick=()=>{points.length=0;redraw()};
</script></body></html>"""


def png_data(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("결과 이미지를 만들 수 없습니다.")
    return base64.b64encode(encoded).decode("ascii")


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지 파일을 읽을 수 없습니다.")
    return image


def render_home(results=None, error=None, image_token=None):
    return render_template_string(HTML, results=results, error=error, points="", image_token=image_token)


def render_manual_picker(image: np.ndarray, image_token: str, error=None):
    return render_template_string(
        ROI_HTML,
        image_data=png_data(image),
        image_token=image_token,
        error=error,
    )


def parse_points(value: str) -> np.ndarray:
    return third.DEFAULT_TARGET_POINTS.copy() if not value.strip() else third.parse_points(value)


def run_third(image: np.ndarray, points: np.ndarray) -> dict:
    warped, perspective, ordered = third.warp_by_four_points(image, points, output_width=900)
    x_angle, y_angle, _ = third.estimate_grid_axes(warped)
    rectified, affine = third.affine_rectify_axes(warped, x_angle, y_angle)
    polygon = third.grid_polygon_in_rectified(warped.shape, affine)
    horizontal, vertical, *_ = third.build_profile_clusters(rectified, polygon)
    overlay = third.draw_profile_overlay(rectified, horizontal, vertical, polygon)
    info = {"ordered_points": ordered.tolist(), "x_angle": x_angle, "y_angle": y_angle,
            "method": "directional profile clustering"}
    return {"label":"shlee 브랜치 · 3차시도", "title":"수동 외곽 지정 + clustering",
            "description":"원근 보정 후 Hough/방향 profile을 이용해 두 방향의 철근을 clustering합니다.",
            "horizontal":len(horizontal), "vertical":len(vertical), "image":png_data(overlay),
            "details":json.dumps(info, ensure_ascii=False, indent=2)}


def run_fourth(image: np.ndarray) -> dict:
    points, debug_image, detection = fourth.auto_detect_outer_quad(image)
    warped_full, perspective, ordered = fourth.warp_by_four_points(image, points, output_width=900)
    warped, warp_bbox = fourth.trim_valid_region(warped_full)
    try:
        x_angle, y_angle, edges = fourth.estimate_grid_axes(warped)
    except RuntimeError:
        x_angle, y_angle, edges = fourth.estimate_grid_axes(warped_full)
    rectified_full, affine = fourth.affine_rectify_axes(warped, x_angle, y_angle)
    rectified, rect_bbox = fourth.trim_valid_region(rectified_full)
    x_profile, y_profile = fourth.make_profiles(rectified)
    x_peaks, x_threshold = fourth.find_profile_peaks(x_profile, max(45, int(rectified.shape[0] * .09)))
    y_peaks, y_threshold = fourth.find_profile_peaks(y_profile, max(45, int(rectified.shape[1] * .05)))
    overlay = fourth.draw_count_overlay(rectified, x_peaks, y_peaks)
    info = {"quad_detection_method": detection.get("quad_detection_method"),
            "quad_detection_score": detection.get("quad_detection_score"),
            "ordered_points": ordered.tolist(), "x_angle": x_angle, "y_angle": y_angle,
            "x_threshold": x_threshold, "y_threshold": y_threshold,
            "warp_bbox": list(warp_bbox), "rectified_bbox": list(rect_bbox),
            "method": "automatic Hough quad + directional profile peak"}
    return {"label":"main 브랜치 · 4차시도", "title":"자동 외곽 검출 + profile peak",
            "description":"Hough 직선 조합으로 외곽을 자동 검출한 뒤 축 보정과 profile peak로 카운트합니다.",
            "horizontal":len(x_peaks), "vertical":len(y_peaks), "image":png_data(overlay),
            "details":json.dumps(info, ensure_ascii=False, indent=2)}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_home()
    try:
        upload = request.files.get("image")
        if not upload or not upload.filename:
            raise ValueError("이미지를 먼저 선택한 뒤 ROI 버튼을 눌러주세요.")
        raw = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("이미지는 16MB 이하만 업로드할 수 있습니다.")
        image = decode_image(raw)
        mode = request.form.get("mode")
        if mode == "upload":
            image_token = uuid.uuid4().hex
            PENDING_IMAGES[image_token] = raw
            PENDING_RESULTS[image_token] = {}
            return render_home(image_token=image_token)
        raise ValueError("이미지를 업로드한 뒤 ROI 방법을 선택하세요.")
    except (RuntimeError, ValueError, cv2.error) as exc:
        return render_home(error=str(exc)), 400


@app.route("/run", methods=["POST"])
def run_method():
    try:
        image_token = request.form.get("image_token", "")
        raw = PENDING_IMAGES.get(image_token)
        if raw is None:
            raise ValueError("이미지가 만료되었습니다. 이미지를 다시 업로드하세요.")
        image = decode_image(raw)
        mode = request.form.get("mode")
        if mode == "manual":
            return render_manual_picker(image, image_token)
        if mode == "auto":
            PENDING_RESULTS.setdefault(image_token, {})["auto"] = run_fourth(image)
            return render_home(list(PENDING_RESULTS[image_token].values()), image_token=image_token)
        raise ValueError("ROI 방법을 선택하세요.")
    except (RuntimeError, ValueError, cv2.error) as exc:
        return render_home(error=str(exc)), 400


@app.route("/analyze-manual", methods=["POST"])
def analyze_manual():
    try:
        image_token = request.form.get("image_token", "")
        raw = PENDING_IMAGES.get(image_token)
        if raw is None:
            raise ValueError("분석할 이미지가 없습니다. 처음부터 다시 시도하세요.")
        image = decode_image(raw)
        points_text = request.form.get("points", "")
        points = third.parse_points(points_text)
        PENDING_RESULTS.setdefault(image_token, {})["manual"] = run_third(image, points)
        return render_home(list(PENDING_RESULTS[image_token].values()), image_token=image_token)
    except (RuntimeError, ValueError, cv2.error, base64.binascii.Error) as exc:
        return render_home(error=str(exc)), 400


if __name__ == "__main__":
    # Keep one server process so the browser always talks to the current code.
    app.run(host="127.0.0.1", port=5000, debug=False)
