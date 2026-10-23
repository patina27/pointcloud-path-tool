// Point-cloud path tool: lift images to metric point clouds (MoGe on the GPU pod), fuse views placed at chosen
// cameras, draw a camera path through the cloud, preview it, and export OpenCV cameras + point-render guide frames.
//
// Conventions. Three.js world/camera frames (camera looks down -Z, +Y up). A lifted view keeps its points in OpenCV
// camera coordinates (x right, y down, z forward, metres); its object matrix is c2w * scale * FLIP. Exports use the
// OpenCV frame of the first view as world: c2w_cv = FLIP * c2w_three * FLIP (FLIP = diag(1, -1, -1)).
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { TransformControls } from "three/addons/TransformControls.js";

const $ = (id) => document.getElementById(id);
const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);
const HELPERS = 1;   // layer for frustums, path and gizmos: drawn in the viewport, never in path/guide renders
const CLEAR = 0x0a0a0c;
const COLOR = { view: 0xf2b441, viewHover: 0xffffff, key: 0x8db8ff, keySel: 0xf2b441, keyHover: 0xffffff, path: 0xe8f0ff };

// ---------------------------------------------------------------- renderer, scene, viewport camera
const canvas = $("canvas");
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(CLEAR, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 2000);
camera.layers.enable(HELPERS);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
const pathCamera = new THREE.PerspectiveCamera(50, 16 / 9, 0.01, 2000);   // renders points only (layer 0)

const uniforms = { projScale: { value: 500 }, sizeScale: { value: 1.6 } };
const pointMaterial = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: `
    attribute vec3 rgb; attribute float radius; uniform float projScale; uniform float sizeScale; varying vec3 vColor;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      // radius: the pixel footprint of the point in its source view, in object units; modelViewMatrix carries scale
      float world = radius * length(modelViewMatrix[0].xyz);
      gl_PointSize = clamp(sizeScale * world * projScale / max(-mv.z, 1e-4), 1.0, 96.0);
      vColor = rgb;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() { vec2 c = gl_PointCoord * 2.0 - 1.0; if (dot(c, c) > 1.0) discard; gl_FragColor = vec4(vColor, 1.0); }`,
});

function setProjScale(cam, heightPx) {
  uniforms.projScale.value = heightPx / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2));
}

// ---------------------------------------------------------------- helpers
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.style.display = "none"), ms);
}
const hfovOf = (vfov, aspect) => THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(vfov) / 2) * aspect));
const vfovOf = (hfov, aspect) => THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(hfov) / 2) / aspect));
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function frustum(vfov, aspect, depth, color, texture = null) {
  const g = new THREE.Group();
  const y = depth * Math.tan(THREE.MathUtils.degToRad(vfov) / 2), x = y * aspect;
  const c = [[-x, -y], [x, -y], [x, y], [-x, y]];
  const v = [];
  for (const [a, b] of c) v.push(0, 0, 0, a, b, -depth);
  for (let i = 0; i < 4; i++) { const [a, b] = c[i], [d, e] = c[(i + 1) % 4]; v.push(a, b, -depth, d, e, -depth); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
  g.add(lines);
  if (texture) {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * x, 2 * y),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false }));
    plane.position.z = -depth;
    g.add(plane);
  }
  g.traverse((o) => o.layers.set(HELPERS));
  g.userData.lines = lines;
  return g;
}

function disposeTree(o) {
  o.traverse((c) => { c.geometry?.dispose(); if (c.material) { c.material.map?.dispose(); if (c.material !== pointMaterial) c.material.dispose(); } });
}

function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
const b64 = {
  enc(buf) { const u = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength); let s = ""; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); },
  dec(str) { const s = atob(str); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; },
};

// ---------------------------------------------------------------- activity: one persistent progress/status slot
// Long operations (lift, multi-view, export) show a spinner, elapsed time and, when known, a progress bar. Failures stay
// until dismissed and can offer a retry.
const activity = { t0: 0, timer: null, hideTimer: null, retry: null, running: false };
const act = (sel) => $("activity").querySelector(sel);
function jobStart(label, detail = "", determinate = false) {
  const a = $("activity");
  clearTimeout(activity.hideTimer); clearInterval(activity.timer);
  a.hidden = false; a.className = "running";
  act(".act-label").textContent = label; act(".act-detail").textContent = detail;
  act(".progress").classList.toggle("indeterminate", !determinate);
  act(".progress i").style.width = determinate ? "0%" : "";
  act(".act-retry").hidden = true; act(".act-close").hidden = true;
  activity.t0 = performance.now(); activity.running = true;
  const tick = () => (act(".act-time").textContent = secs(performance.now() - activity.t0));
  tick(); activity.timer = setInterval(tick, 100);
  updateUi();
}
function jobProgress(frac, detail) {
  act(".progress i").style.width = `${(100 * frac).toFixed(1)}%`;
  if (detail !== undefined) act(".act-detail").textContent = detail;
}
function jobEnd(cls, label, detail, retry) {
  clearInterval(activity.timer); activity.running = false;
  const a = $("activity");
  a.className = cls; act(".act-label").textContent = label; act(".act-detail").textContent = detail || "";
  act(".act-time").textContent = secs(performance.now() - activity.t0);
  act(".act-close").hidden = false;
  activity.retry = retry || null; act(".act-retry").hidden = !retry;
  if (cls === "done") activity.hideTimer = setTimeout(() => (a.hidden = true), 12000);
  updateUi();
}
const jobDone = (label, detail) => jobEnd("done", label, detail);
const jobFail = (label, detail, retry) => { jobEnd("err", label, detail, retry); checkHealth(); };
act(".act-close").onclick = () => { clearTimeout(activity.hideTimer); $("activity").hidden = true; };
act(".act-retry").onclick = () => { const r = activity.retry; $("activity").hidden = true; r?.(); };

// ---------------------------------------------------------------- views (lifted images)
const views = [];   // {name, dataURL, header, index: Uint32Array, points: Float32Array, colors: Uint8Array, c2w: Matrix4, scale, visible, object, frustum}
let viewCounter = 0;
let sceneVersion = 0;   // bumps whenever the points change (for keyframe thumbnails)
let hoverView = null;

function parseLift(buffer) {
  const n = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  let o = 4 + n;
  const count = header.count;
  const index = new Uint32Array(buffer.slice(o, o + 4 * count)); o += 4 * count;
  const points = new Float32Array(buffer.slice(o, o + 12 * count)); o += 12 * count;
  const colors = new Uint8Array(buffer.slice(o, o + 3 * count));
  return { header, index, points, colors };
}

async function apiError(r) {
  const detail = (await r.json().catch(() => ({}))).detail;
  return new Error(typeof detail === "string" ? detail : `${r.status} ${r.statusText}`);
}

async function liftImage(file, fovX) {
  const form = new FormData();
  form.append("image", file);
  form.append("model", $("model").value);
  form.append("max_side", $("maxSide").value);
  if (fovX) form.append("fov_x", String(fovX));
  const r = await fetch("api/lift", { method: "POST", body: form });
  if (!r.ok) throw await apiError(r);
  return parseLift(await r.arrayBuffer());
}

function buildViewObject(view) {
  const { points, colors, header } = view;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(points, 3));
  geo.setAttribute("rgb", new THREE.BufferAttribute(colors, 3, true));
  const fx = header.K[0][0], radius = new Float32Array(points.length / 3);
  for (let i = 0; i < radius.length; i++) radius[i] = points[3 * i + 2] / fx;
  geo.setAttribute("radius", new THREE.BufferAttribute(radius, 1));
  geo.computeBoundingSphere();
  const obj = new THREE.Points(geo, pointMaterial);
  obj.matrixAutoUpdate = false;
  obj.frustumCulled = false;
  view.object = obj;
  scene.add(obj);
  const tex = new THREE.TextureLoader().load(view.dataURL);
  tex.colorSpace = THREE.SRGBColorSpace;
  const aspect = header.width / header.height, vfov = vfovOf(header.fov_x, aspect);
  view.frustum = frustum(vfov, aspect, 0.25, COLOR.view, tex);
  view.frustum.matrixAutoUpdate = false;
  view.labelLocal = new THREE.Vector3(0, 0.25 * Math.tan(THREE.MathUtils.degToRad(vfov) / 2), -0.25);   // top edge of the image plane
  scene.add(view.frustum);
  updateViewMatrix(view);
}

function updateViewMatrix(view) {
  const s = new THREE.Matrix4().makeScale(view.scale, view.scale, view.scale);
  view.object.matrix.copy(view.c2w).multiply(s).multiply(FLIP);
  view.object.matrixWorldNeedsUpdate = true;
  view.object.visible = view.visible;
  const depth = Math.max(0.05, 0.08 * sceneRadius());
  view.frustum.matrix.copy(view.c2w).multiply(new THREE.Matrix4().makeScale(depth / 0.25, depth / 0.25, depth / 0.25));
  view.frustum.matrixWorldNeedsUpdate = true;
  view.frustum.visible = view.visible && $("showFrusta").checked;
  view.frustum.userData.lines.material.color.setHex(view === hoverView ? COLOR.viewHover : COLOR.view);
  sceneVersion++; scheduleThumbs();
}

function sceneRadius() {
  // median depth of the first view, scaled: sets frustum sizes and fly speed
  const v = views[0];
  if (!v) return 1;
  if (!v.medianDepth) {
    const z = []; for (let i = 2; i < v.points.length; i += 3 * 97) z.push(v.points[i]);
    z.sort((a, b) => a - b); v.medianDepth = z[z.length >> 1] || 1;
  }
  return v.medianDepth * v.scale;
}

// Scale of a new view so its depths match the existing points it sees: median over pixel cells covered by both of
// (existing depth along the new camera) / (new view's predicted depth).
function alignScale(view) {
  const { header, index, points } = view;
  const w = header.width, h = header.height, K = header.K, cell = 4;
  const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
  const zbuf = new Float32Array(gw * gh).fill(Infinity);
  const toCam = new THREE.Matrix4().copy(FLIP).multiply(view.c2w.clone().invert());   // world -> new OpenCV camera
  const m = new THREE.Matrix4();
  for (const other of views) {
    if (other === view || !other.visible) continue;
    m.copy(toCam).multiply(other.object.matrix);
    const e = m.elements, P = other.points;
    for (let i = 0; i < P.length; i += 3) {
      const x = P[i], y = P[i + 1], zz = P[i + 2];
      const cx = e[0] * x + e[4] * y + e[8] * zz + e[12], cy = e[1] * x + e[5] * y + e[9] * zz + e[13], cz = e[2] * x + e[6] * y + e[10] * zz + e[14];
      if (cz <= 1e-3) continue;
      const u = K[0][0] * cx / cz + K[0][2], v = K[1][1] * cy / cz + K[1][2];
      if (u < 0 || v < 0 || u >= w || v >= h) continue;
      const k = Math.floor(v / cell) * gw + Math.floor(u / cell);
      if (cz < zbuf[k]) zbuf[k] = cz;
    }
  }
  const ratios = [];
  for (let i = 0; i < index.length; i++) {
    const px = index[i] % w, py = (index[i] / w) | 0;
    const z = zbuf[Math.floor(py / cell) * gw + Math.floor(px / cell)];
    if (z < Infinity) ratios.push(z / points[3 * i + 2]);
  }
  if (ratios.length < 200) return { scale: null, support: ratios.length };
  ratios.sort((a, b) => a - b);
  return { scale: ratios[ratios.length >> 1], support: ratios.length };
}

const fileToDataURL = (file) => new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file); });

function clearViews() {
  for (const v of [...views]) { scene.remove(v.object); scene.remove(v.frustum); disposeTree(v.object); disposeTree(v.frustum); }
  views.length = 0;
  hoverView = null;
}

// A new scene replaces the views; keyframes live in the old scene's world frame, so they go too.
function startNewScene(firstName, multi) {
  const dropped = keys.length;
  clearViews();
  keys.length = 0; selectedKey = -1; gizmo.detach(); guidePose = null;
  if (!touched.exportName) $("exportName").value = exportNameFrom(firstName, multi);
  return dropped ? ` · cleared ${dropped} keyframe${dropped > 1 ? "s" : ""} from the previous scene` : "";
}

function afterFirstView() {
  if (!touched.size) matchInputAspect();
  views.forEach(updateViewMatrix);
  resetViewToInput(views[0]);
  renderViewList(); rebuildPath();
}

// Several photos with unknown cameras: the service estimates cameras (VGGT) and lifts each photo (MoGe); each blob's
// header carries its OpenCV c2w (world = first photo's camera, metres) and the scale for its points.
async function addMultiView(files) {
  const form = new FormData();
  for (const f of files) form.append("images", f);
  form.append("model", $("model").value);
  form.append("max_side", $("maxSide").value);
  jobStart(`Reconstructing ${files.length} photos`, "Estimating cameras (VGGT) and lifting each photo (MoGe). Usually 2–25 s.");
  const t0 = performance.now();
  const r = await fetch("api/lift_multi", { method: "POST", body: form });
  if (!r.ok) throw await apiError(r);
  const buf = await r.arrayBuffer(), dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  const note = startNewScene(files[0].name, true);
  let o = 4;
  for (let i = 0; i < n; i++) {
    const len = dv.getUint32(o, true);
    const lifted = parseLift(buf.slice(o + 4, o + 4 + len));
    o += 4 + len;
    const cv = new THREE.Matrix4().set(...lifted.header.c2w.flat());
    const view = { name: files[i].name, dataURL: await fileToDataURL(files[i]), ...lifted, scale: lifted.header.scale,
                   visible: true, id: viewCounter++, c2w: new THREE.Matrix4().copy(FLIP).multiply(cv).multiply(FLIP) };
    buildViewObject(view);
    views.push(view);
  }
  afterFirstView();
  const pts = views.reduce((a, v) => a + v.header.count, 0);
  jobDone(`${n} photos → ${(pts / 1e6).toFixed(1)}M points in ${secs(performance.now() - t0)}`,
    `Per-photo scale ${views.map((v) => v.scale.toFixed(2)).join(", ")}${note}`);
}

async function addView(file, placeAtViewport) {
  const dataURL = await fileToDataURL(file);
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("not a readable image")); i.src = dataURL; });
  const aspect = img.width / img.height;
  const fovX = placeAtViewport ? hfovOf(camera.fov, aspect) : null;
  const c2w = placeAtViewport ? camera.matrixWorld.clone() : new THREE.Matrix4();
  jobStart(`Lifting ${file.name}`, `${$("model").value}, max side ${$("maxSide").value} px${health.models.includes($("model").value) ? "" : ". First use loads the model, which takes longer."}`);
  const t0 = performance.now();
  const lifted = await liftImage(file, fovX);
  const note = placeAtViewport ? "" : startNewScene(file.name, false);
  const view = { name: file.name || `view${viewCounter}`, dataURL, ...lifted, scale: 1, visible: true, id: viewCounter++, c2w };
  buildViewObject(view);
  views.push(view);
  let detail = `fov_x ${lifted.header.fov_x.toFixed(1)}°`;
  if (placeAtViewport && views.length > 1 && $("autoAlign").checked) {
    const { scale, support } = alignScale(view);
    if (scale) { view.scale = scale; detail += ` · scale ×${scale.toFixed(3)} from ${support} px of overlap`; }
    else detail += ` · too little overlap to align (${support} px), kept scale 1`;
  }
  if (placeAtViewport) guidePose = null;   // the guide round trip is complete
  if (views.length === 1) afterFirstView();
  else { views.forEach(updateViewMatrix); renderViewList(); }
  jobDone(`${placeAtViewport ? `Added view ${views.length}` : "Lifted"}: ${lifted.header.count.toLocaleString()} points in ${secs(performance.now() - t0)}`,
    `${view.name} · ${detail}${note}`);
}

// Entry points used by buttons, file inputs and drag-and-drop. Errors stay visible and can be retried.
async function runLiftNew(file) {
  try { await addView(file, false); }
  catch (e) { jobFail(`Couldn't lift ${file.name}`, liftHint(e), () => runLiftNew(file)); }
}
async function runLiftAdd(file) {
  if (!views.length) return toast("Lift a first image before adding views to it");
  try { await addView(file, true); }
  catch (e) { jobFail(`Couldn't add ${file.name}`, liftHint(e), () => runLiftAdd(file)); }
}
async function runMulti(files) {
  if (files.length < 2) return toast("Choose at least 2 photos of the same place");
  try { await addMultiView(files); }
  catch (e) { jobFail(`Couldn't reconstruct ${files.length} photos`, liftHint(e), () => runMulti(files)); }
}
function liftHint(e) {
  const msg = e.message || String(e);
  return health.ok ? msg : `${msg}. The GPU service is not reachable; lifting works again once it is back.`;
}

function resetViewToInput(view) {
  const aspect = view.header.width / view.header.height;
  camera.fov = vfovOf(view.header.fov_x, aspect);
  $("fov").value = camera.fov.toFixed(0);
  camera.updateProjectionMatrix();
  view.c2w.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, sceneRadius());
  controls.update();
  updateFovLabel();
}

function setHoverView(v) {
  if (hoverView === v) return;
  const prev = hoverView; hoverView = v;
  for (const x of [prev, v]) if (x) x.frustum.userData.lines.material.color.setHex(x === hoverView ? COLOR.viewHover : COLOR.view);
  document.querySelectorAll("#views .item").forEach((el, i) => el.classList.toggle("hover", views[i] === v));
  labels.views.forEach((el, i) => el.classList.toggle("hover", views[i] === v));
}

function renderViewList() {
  const box = $("views");
  box.innerHTML = "";
  views.forEach((v, i) => {
    const d = document.createElement("div");
    d.className = "item" + (v.visible ? "" : " off");
    const vfov = vfovOf(v.header.fov_x, v.header.width / v.header.height);
    d.innerHTML = `<span class="thumbwrap"><img src="${v.dataURL}" alt=""><span class="tag">${i + 1}</span></span>
      <div class="meta"><span class="name" title="${esc(v.name)}">${esc(v.name)}</span>
        <span class="info2"><span class="info" title="${v.header.count.toLocaleString()} points, vertical fov ${vfov.toFixed(1)}°">${(v.header.count / 1e3).toFixed(0)}k · ${vfov.toFixed(0)}°</span>
        <span class="scale" title="Scale of this view's points">×<input type="number" step="0.01" value="${v.scale.toFixed(3)}" aria-label="scale"></span></span></div>
      <div class="acts"><button data-a="go" title="Look from this camera">◎</button><button data-a="vis" class="${v.visible ? "on" : ""}" title="${v.visible ? "Hide" : "Show"} these points">${v.visible ? "●" : "○"}</button><button data-a="del" title="Remove this view">✕</button></div>`;
    d.querySelector("input").onchange = (e) => { v.scale = parseFloat(e.target.value) || 1; updateViewMatrix(v); };
    d.querySelector('[data-a="go"]').onclick = () => resetViewToInput(v);
    d.querySelector('[data-a="vis"]').onclick = () => { v.visible = !v.visible; updateViewMatrix(v); renderViewList(); };
    d.querySelector('[data-a="del"]').onclick = () => {
      if (hoverView === v) hoverView = null;
      scene.remove(v.object); scene.remove(v.frustum); disposeTree(v.object); disposeTree(v.frustum);
      views.splice(views.indexOf(v), 1); sceneVersion++; scheduleThumbs(); renderViewList();
    };
    d.onmouseenter = () => setHoverView(v);
    d.onmouseleave = () => setHoverView(null);
    box.appendChild(d);
  });
  syncLabels();
  updateUi();
}

// ---------------------------------------------------------------- camera path
const keys = [];   // {position, quaternion, fov}
let selectedKey = -1, hoverKey = -1;
const pathGroup = new THREE.Group();
scene.add(pathGroup);
let pathFrames = [];   // {position, quaternion, fov}
let keyFrames = [];    // frame index of each keyframe along the path
let keyLabelPos = [];  // world position of each keyframe's label
const gizmoTarget = new THREE.Object3D();
scene.add(gizmoTarget);
const gizmo = new TransformControls(camera, canvas);
gizmo.setSize(0.8);
const gizmoHelper = gizmo.getHelper ? gizmo.getHelper() : gizmo;
gizmoHelper.traverse((o) => o.layers.set(HELPERS));
scene.add(gizmoHelper);
gizmo.addEventListener("dragging-changed", (e) => (controls.enabled = !e.value));
gizmo.addEventListener("objectChange", () => {
  if (selectedKey < 0) return;
  keys[selectedKey].position.copy(gizmoTarget.position);
  keys[selectedKey].quaternion.copy(gizmoTarget.quaternion);
  rebuildPath();
});

const poseHere = () => ({ position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov: camera.fov });

function addKey() {
  keys.push(poseHere());
  rebuildPath();
  setFrame(keyFrames[keys.length - 1] ?? 0);
  toast(`Added keyframe ${keys.length}${keys.length === 1 ? ". Fly somewhere else and press K again to make a path" : ""}`, 2500);
}

function selectKey(i) {
  selectedKey = i;
  if (i < 0) gizmo.detach();
  else { gizmoTarget.position.copy(keys[i].position); gizmoTarget.quaternion.copy(keys[i].quaternion); gizmo.attach(gizmoTarget); }
  rebuildPath();
}

function deleteKey(i) { keys.splice(i, 1); selectedKey = -1; gizmo.detach(); rebuildPath(); }

// Keyframe thumbnails: the points seen from each keyframe, re-rendered (debounced) when the key or the scene changes.
const thumbs = new WeakMap();   // key -> {sig, url}
let thumbTimer = null, thumbBusy = false;
const thumbSig = (k) => `${k.position.toArray().join()}|${k.quaternion.toArray().join()}|${k.fov}|${sceneVersion}|${outAspect().toFixed(3)}`;
function scheduleThumbs() { clearTimeout(thumbTimer); thumbTimer = setTimeout(renderThumbs, 300); }
async function renderThumbs() {
  if (thumbBusy || exporting) return scheduleThumbs();
  thumbBusy = true;
  try {
    const W = 128, H = Math.max(16, Math.round(W / outAspect()));
    const cam = new THREE.PerspectiveCamera(50, W / H, 0.01, 2000);
    for (const k of [...keys]) {
      const sig = thumbSig(k), old = thumbs.get(k);
      if (old?.sig === sig) continue;
      cam.position.copy(k.position); cam.quaternion.copy(k.quaternion); cam.fov = k.fov; cam.updateMatrixWorld();
      const blob = await renderOffscreen(cam, W, H);
      if (old) URL.revokeObjectURL(old.url);
      thumbs.set(k, { sig, url: URL.createObjectURL(blob) });
      const i = keys.indexOf(k), img = document.querySelectorAll("#keys .item .thumb")[i];
      if (img) img.src = thumbs.get(k).url;
    }
  } finally { thumbBusy = false; }
}

function renderKeyList() {
  const box = $("keys");
  box.innerHTML = "";
  const fps = parseInt($("fps").value) || 24;
  keys.forEach((k, i) => {
    const d = document.createElement("div");
    d.className = "item key" + (i === selectedKey ? " sel" : "");
    const f = keyFrames[i] ?? 0, url = thumbs.get(k)?.url;
    d.innerHTML = `<span class="thumbwrap"><img class="thumb" alt="" title="Jump to this keyframe on the timeline" ${url ? `src="${url}"` : ""}><span class="tag">K${i + 1}</span></span>
      <div class="meta"><span class="name">frame ${f}</span><span class="info">${(f / fps).toFixed(2)} s · fov ${k.fov.toFixed(0)}°</span></div>
      <div class="acts"><button data-a="sel" class="${i === selectedKey ? "on" : ""}" title="${i === selectedKey ? "Stop editing (Esc)" : "Move / rotate with the gizmo (T / R)"}">✥</button><button data-a="go" title="Look from this keyframe">◎</button><button data-a="set" title="Replace with the current view">⟲</button><button data-a="del" title="Delete keyframe">✕</button></div>`;
    d.querySelector(".thumb").onclick = () => setFrame(f);
    d.querySelector(".meta").onclick = () => setFrame(f);
    d.querySelector('[data-a="sel"]').onclick = () => selectKey(i === selectedKey ? -1 : i);
    d.querySelector('[data-a="go"]').onclick = () => { lookFrom(k); setFrame(f); };
    d.querySelector('[data-a="set"]').onclick = () => { keys[i] = poseHere(); selectedKey = -1; gizmo.detach(); rebuildPath(); };
    d.querySelector('[data-a="del"]').onclick = () => deleteKey(i);
    d.onmouseenter = () => setHoverKey(i);
    d.onmouseleave = () => setHoverKey(-1);
    box.appendChild(d);
  });
}

function setHoverKey(i) {
  if (hoverKey === i) return;
  hoverKey = i;
  pathGroup.children.forEach((c) => { if (c.userData.key !== undefined) c.userData.lines.material.color.setHex(keyColor(c.userData.key)); });
  document.querySelectorAll("#keys .item").forEach((el, j) => el.classList.toggle("hover", j === i));
  labels.keys.forEach((el, j) => el.classList.toggle("hover", j === i));
}
const keyColor = (i) => (i === hoverKey ? COLOR.keyHover : i === selectedKey ? COLOR.keySel : COLOR.key);

function lookFrom(pose) {
  camera.position.copy(pose.position); camera.quaternion.copy(pose.quaternion);
  camera.fov = pose.fov; camera.updateProjectionMatrix(); $("fov").value = pose.fov.toFixed(0); updateFovLabel();
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, Math.max(0.1, 0.3 * sceneRadius()));
  camera.lookAt(controls.target);   // keeps OrbitControls consistent (roll is dropped while orbiting)
  camera.quaternion.copy(pose.quaternion);
}

function interpolate(n) {
  keyFrames = keys.map(() => 0);
  if (keys.length === 0) return [];
  if (keys.length === 1) return Array.from({ length: n }, () => ({ ...keys[0] }));
  const curve = new THREE.CatmullRomCurve3(keys.map((k) => k.position), false, "centripetal");
  const quats = keys.map((k) => k.quaternion.clone());
  for (let i = 1; i < quats.length; i++) if (quats[i].dot(quats[i - 1]) < 0) quats[i].set(-quats[i].x, -quats[i].y, -quats[i].z, -quats[i].w);
  const faceForward = $("faceForward").checked, up = new THREE.Vector3(0, 1, 0), m = new THREE.Matrix4();
  const out = [];
  for (let f = 0; f < n; f++) {
    const u = n === 1 ? 0 : f / (n - 1);
    const t = curve.getUtoTmapping(u);
    const seg = Math.min(Math.floor(t * (keys.length - 1)), keys.length - 2), local = t * (keys.length - 1) - seg;
    const position = curve.getPoint(t);
    let quaternion;
    if (faceForward) {
      const tangent = curve.getTangent(t);
      m.lookAt(position, position.clone().add(tangent), up);
      quaternion = new THREE.Quaternion().setFromRotationMatrix(m);
    } else {
      const s = local * local * (3 - 2 * local);   // ease within a segment; continuous at keyframes
      quaternion = quats[seg].clone().slerp(quats[seg + 1], s);
    }
    out.push({ position, quaternion, fov: THREE.MathUtils.lerp(keys[seg].fov, keys[seg + 1].fov, local) });
  }
  // keyframe i sits at curve parameter t = i / (K - 1); map it to an arc-length fraction, then to a frame
  const lengths = curve.getLengths(), D = lengths.length - 1, total = lengths[D] || 1;
  keyFrames = keys.map((_, i) => {
    const x = (i / (keys.length - 1)) * D, j = Math.min(Math.floor(x), D - 1), a = x - j;
    const u = (lengths[j] * (1 - a) + lengths[j + 1] * a) / total;
    return Math.round(u * (n - 1));
  });
  return out;
}

function rebuildPath() {
  for (const c of [...pathGroup.children]) { pathGroup.remove(c); disposeTree(c); }
  const n = Math.max(2, parseInt($("frames").value) || 120);
  pathFrames = interpolate(n);
  $("frame").max = String(Math.max(0, pathFrames.length - 1));
  keyLabelPos = [];
  if (pathFrames.length) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pathFrames.map((p) => p.position)),
      new THREE.LineBasicMaterial({ color: COLOR.path }));
    pathGroup.add(line);
    const aspect = outAspect(), depth = Math.max(0.03, 0.05 * sceneRadius());
    keys.forEach((k, i) => {
      const f = frustum(k.fov, aspect, depth * 1.4, keyColor(i));
      f.position.copy(k.position); f.quaternion.copy(k.quaternion); f.userData.key = i; pathGroup.add(f);
      const y = depth * 1.4 * Math.tan(THREE.MathUtils.degToRad(k.fov) / 2);
      keyLabelPos.push(new THREE.Vector3(0, y, -depth * 1.4).applyQuaternion(k.quaternion).add(k.position));
    });
  }
  pathGroup.traverse((o) => o.layers.set(HELPERS));
  renderKeyList();
  renderTicks();
  syncLabels();
  onFrameChange();
  scheduleThumbs();
  updateUi();
}

const aheadGroup = new THREE.Group();
scene.add(aheadGroup);
function updatePathAhead() {
  for (const c of [...aheadGroup.children]) { aheadGroup.remove(c); disposeTree(c); }
  if (!pathFrames.length) return;
  const cur = currentFrame(), ahead = parseInt($("ahead").value) || 0, aspect = outAspect();
  const depth = Math.max(0.03, 0.05 * sceneRadius()), step = Math.max(1, Math.round(ahead / 8));
  for (let f = cur; f <= Math.min(pathFrames.length - 1, cur + ahead); f += step) {
    const p = pathFrames[f];
    const fr = frustum(p.fov, aspect, depth, f === cur ? 0xffffff : 0x9fb6d8);
    fr.position.copy(p.position); fr.quaternion.copy(p.quaternion);
    aheadGroup.add(fr);
  }
  aheadGroup.traverse((o) => o.layers.set(HELPERS));
}

const currentFrame = () => Math.min(parseInt($("frame").value) || 0, Math.max(0, pathFrames.length - 1));
const outAspect = () => (parseInt($("outW").value) || 16) / (parseInt($("outH").value) || 9);

function setPathCamera(i) {
  const p = pathFrames[i];
  pathCamera.position.copy(p.position); pathCamera.quaternion.copy(p.quaternion);
  pathCamera.fov = p.fov; pathCamera.aspect = outAspect(); pathCamera.updateProjectionMatrix(); pathCamera.updateMatrixWorld();
}

// ---------------------------------------------------------------- timeline
function setFrame(f) {
  if (!pathFrames.length) return;
  $("frame").value = String(THREE.MathUtils.clamp(Math.round(f), 0, pathFrames.length - 1));
  onFrameChange();
}

function renderTicks() {
  const box = $("ticks");
  box.innerHTML = "";
  const n = pathFrames.length;
  if (!n) return;
  keys.forEach((_, i) => {
    const b = document.createElement("button");
    b.className = "tick" + (i === selectedKey ? " sel" : "");
    b.textContent = `K${i + 1}`;
    b.title = `Keyframe ${i + 1} · frame ${keyFrames[i]}. Click to jump, double-click to edit.`;
    b.style.left = `calc(6px + (100% - 12px) * ${n > 1 ? keyFrames[i] / (n - 1) : 0})`;
    b.onclick = () => setFrame(keyFrames[i]);
    b.ondblclick = () => selectKey(i);
    b.onmouseenter = () => setHoverKey(i);
    b.onmouseleave = () => setHoverKey(-1);
    box.appendChild(b);
  });
}

function onFrameChange() {
  const f = currentFrame(), n = pathFrames.length, fps = parseInt($("fps").value) || 24;
  $("frameVal").textContent = String(f).padStart(String(Math.max(0, n - 1)).length, "0");
  $("frameMax").textContent = String(Math.max(0, n - 1));
  $("timeVal").textContent = `${(f / fps).toFixed(2)} s / ${(n / fps).toFixed(2)} s`;
  updatePathAhead();
}

function setPlaying(p) {
  playing = p && pathFrames.length > 0;
  $("btnPlay").textContent = playing ? "❚❚" : "▶";
  $("btnPlay").setAttribute("aria-label", playing ? "Pause" : "Play");
}

// ---------------------------------------------------------------- 3D labels: numbered input cameras and keyframes
const labels = { views: [], keys: [] };
function syncLabels() {
  const box = $("labels");
  box.innerHTML = "";
  labels.views = views.map((v, i) => {
    const el = document.createElement("div");
    el.className = "lbl view" + (v === hoverView ? " hover" : "");
    el.textContent = String(i + 1);
    el.title = `${v.name}: click to look from this camera`;
    el.onclick = () => resetViewToInput(v);
    el.onmouseenter = () => setHoverView(v);
    el.onmouseleave = () => setHoverView(null);
    box.appendChild(el);
    return el;
  });
  labels.keys = keys.map((_, i) => {
    const el = document.createElement("div");
    el.className = "lbl key" + (i === selectedKey ? " sel" : "");
    el.textContent = `K${i + 1}`;
    el.title = `Keyframe ${i + 1}: click to edit with the gizmo`;
    el.onclick = () => { selectKey(i === selectedKey ? -1 : i); setFrame(keyFrames[i] ?? 0); };
    el.onmouseenter = () => setHoverKey(i);
    el.onmouseleave = () => setHoverKey(-1);
    box.appendChild(el);
    return el;
  });
}

const _p = new THREE.Vector3();
function placeLabel(el, world, show) {
  if (show) {
    _p.copy(world).project(camera);
    show = _p.z > -1 && _p.z < 1 && Math.abs(_p.x) < 1.05 && Math.abs(_p.y) < 1.05 && world.distanceTo(camera.position) > 1e-3;
  }
  if (!show) { el.style.display = "none"; return; }
  el.style.display = "";
  el.style.transform = `translate(${((_p.x + 1) / 2) * canvas.clientWidth}px, ${((1 - _p.y) / 2) * canvas.clientHeight}px) translate(-50%, -100%) translateY(-3px)`;
}

function updateLabels(look) {
  $("labels").hidden = look;
  if (look) return;
  views.forEach((v, i) => labels.views[i] && placeLabel(labels.views[i], _w.copy(v.labelLocal).applyMatrix4(v.frustum.matrix), v.frustum.visible));
  keys.forEach((_, i) => labels.keys[i] && keyLabelPos[i] && placeLabel(labels.keys[i], keyLabelPos[i], true));
}
const _w = new THREE.Vector3();

// ---------------------------------------------------------------- rendering: viewport, picture-in-picture, offscreen
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

const clock = new THREE.Clock();
let playing = false, playAccum = 0;
const keysDown = new Set();
window.addEventListener("keydown", (e) => {
  const tag = e.target.tagName, type = e.target.type;
  const typing = tag === "SELECT" || tag === "TEXTAREA" || (tag === "INPUT" && type !== "range" && type !== "checkbox");
  if (typing) { if (e.key === "Escape") e.target.blur(); return; }
  if (type === "range" && e.key.startsWith("Arrow")) return;   // the slider steps itself
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (!$("help").hidden) { if (e.key === "Escape" || e.key === "?") showHelp(false); return; }
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keysDown.add(e.key.toLowerCase());
  const step = e.shiftKey ? 10 : 1;
  switch (k) {
    case "t": gizmo.setMode("translate"); break;
    case "r": gizmo.setMode("rotate"); break;
    case "Escape": selectKey(-1); break;
    case "k": addKey(); break;
    case "l": $("lookThrough").checked = !$("lookThrough").checked; break;
    case "?": showHelp(true); break;
    case "f": frameAll(); break;
    case " ":
      e.preventDefault(); if (tag === "BUTTON" || tag === "INPUT") e.target.blur();
      setPlaying(!playing); break;
    case "ArrowLeft": e.preventDefault(); setFrame(currentFrame() - step); break;
    case "ArrowRight": e.preventDefault(); setFrame(currentFrame() + step); break;
    case "[": { const f = currentFrame(), prev = keyFrames.filter((x) => x < f); if (prev.length) setFrame(Math.max(...prev)); break; }
    case "]": { const f = currentFrame(), next = keyFrames.filter((x) => x > f); if (next.length) setFrame(Math.min(...next)); break; }
    case "Delete": case "Backspace": if (selectedKey >= 0) { e.preventDefault(); deleteKey(selectedKey); } break;
  }
});
window.addEventListener("keyup", (e) => keysDown.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => keysDown.clear());

// Pull the viewport camera back along its view direction until every keyframe and input camera is in view.
function frameAll() {
  const pts = [...keys.map((k) => k.position), ...views.filter((v) => v.visible).map((v) => new THREE.Vector3().setFromMatrixPosition(v.c2w))];
  if (!pts.length) return;
  const sphere = new THREE.Sphere().setFromPoints(pts);
  const r = Math.max(sphere.radius, 0.3 * sceneRadius());
  const fov = THREE.MathUtils.degToRad(Math.min(camera.fov, hfovOf(camera.fov, camera.aspect))) / 2;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  camera.position.copy(sphere.center).addScaledVector(fwd, -1.15 * r / Math.sin(fov));
  controls.target.copy(sphere.center); controls.update();
}

function fly(dt) {
  const speed = sceneRadius() * (keysDown.has("shift") ? 1.5 : 0.4) * dt;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion), right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion), d = new THREE.Vector3();
  if (keysDown.has("w")) d.add(fwd); if (keysDown.has("s")) d.sub(fwd);
  if (keysDown.has("d")) d.add(right); if (keysDown.has("a")) d.sub(right);
  if (keysDown.has("e")) d.add(up); if (keysDown.has("q")) d.sub(up);
  if (d.lengthSq()) { d.normalize().multiplyScalar(speed); camera.position.add(d); controls.target.add(d); }
}

function fitRect(aspect, maxW, maxH) {   // largest aspect-correct rect inside maxW x maxH
  let w = maxW, h = w / aspect;
  if (h > maxH) { h = maxH; w = h * aspect; }
  return { w: Math.round(w), h: Math.round(h) };
}

function placeBox(el, x, yFromBottom, w, h) {   // CSS box over the canvas from GL-style coordinates
  el.style.left = `${x}px`; el.style.top = `${canvas.clientHeight - yFromBottom - h}px`;
  el.style.width = `${w}px`; el.style.height = `${h}px`;
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  fly(dt);
  if (playing && pathFrames.length) {
    playAccum += dt * (parseInt($("fps").value) || 24);
    if (playAccum >= 1) {
      let f = currentFrame() + Math.floor(playAccum); playAccum %= 1;
      if (f >= pathFrames.length) f = 0;
      $("frame").value = String(f); onFrameChange();
    }
  }
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const look = $("lookThrough").checked && pathFrames.length > 0;
  if (!look) controls.update();
  const pr = renderer.getPixelRatio();
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, cw, ch);
  if (look) {   // letterboxed to the output aspect: exactly what the export frame sees
    renderer.clear();
    setPathCamera(currentFrame());
    const { w, h } = fitRect(outAspect(), cw, ch), x = Math.round((cw - w) / 2), y = Math.round((ch - h) / 2);
    renderer.setScissorTest(true); renderer.setScissor(x, y, w, h); renderer.setViewport(x, y, w, h);
    renderer.setClearColor(0x000000, 1); renderer.clear();
    setProjScale(pathCamera, h * pr); renderer.render(scene, pathCamera);
    renderer.setScissorTest(false); renderer.setClearColor(CLEAR, 1);
    placeBox($("safeFrame"), x, y, w, h);
    $("safeLabel").textContent = `path camera · frame ${currentFrame()} / ${pathFrames.length - 1} · L to leave`;
  } else {
    setProjScale(camera, ch * pr); renderer.render(scene, camera);
  }
  $("safeFrame").hidden = !look;
  if (pathFrames.length && !look) {   // picture-in-picture: the path camera's view of the points
    const { w: pw, h: ph } = fitRect(outAspect(), Math.round(cw * 0.28), Math.round(ch * 0.42));
    const x = cw - pw - 12, y = 12;
    setPathCamera(currentFrame());
    renderer.setScissorTest(true); renderer.setScissor(x, y, pw, ph); renderer.setViewport(x, y, pw, ph);
    renderer.setClearColor(0x000000, 1); renderer.clear();
    setProjScale(pathCamera, ph * pr); renderer.render(scene, pathCamera);
    renderer.setScissorTest(false); renderer.setClearColor(CLEAR, 1);
    $("pip").hidden = false;
    placeBox($("pip"), x, y, pw, ph);
    $("pipLabel").textContent = `path camera · frame ${currentFrame()} / ${pathFrames.length - 1}`;
  } else $("pip").hidden = true;
  updateLabels(look);
  updateGuideNote();
}

// Render the points (layer 0 only) from `cam` at w x h into RGBA pixels; alpha = covered by a point.
const readTarget = { rt: null };
function renderOffscreen(cam, w, h) {
  if (!readTarget.rt || readTarget.rt.width !== w || readTarget.rt.height !== h) {
    readTarget.rt?.dispose();
    readTarget.rt = new THREE.WebGLRenderTarget(w, h, { samples: 0 });
    readTarget.rt.texture.colorSpace = THREE.SRGBColorSpace;
  }
  const c = cam.clone(); c.layers.set(0); c.aspect = w / h; c.updateProjectionMatrix();
  renderer.setRenderTarget(readTarget.rt);
  renderer.setClearColor(0x000000, 0); renderer.clear();
  setProjScale(c, h); renderer.render(scene, c);
  const px = new Uint8Array(w * h * 4);
  renderer.readRenderTargetPixels(readTarget.rt, 0, 0, w, h, px);
  renderer.setRenderTarget(null); renderer.setClearColor(CLEAR, 1);
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) img.data.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);   // GL rows are bottom-up
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob(res, "image/png"));
}

// ---------------------------------------------------------------- grow: guide render and the add-back round trip
let guidePose = null;   // viewport camera when the last guide was saved
function updateGuideNote() {
  let moved = false;
  if (guidePose) {
    const d = camera.position.distanceTo(guidePose.position), a = camera.quaternion.angleTo(guidePose.quaternion);
    moved = d > 1e-3 * sceneRadius() || a > 0.003 || Math.abs(camera.fov - guidePose.fov) > 0.05;
  }
  if ($("guideNote").hidden === moved) $("guideNote").hidden = !moved;
}

// ---------------------------------------------------------------- export
const cvPose = (position, quaternion) => {
  const m = new THREE.Matrix4().compose(position, quaternion, new THREE.Vector3(1, 1, 1));
  return new THREE.Matrix4().copy(FLIP).multiply(m).multiply(FLIP);
};
const rows = (m) => { const e = m.elements; return [0, 1, 2, 3].map((r) => [e[r], e[4 + r], e[8 + r], e[12 + r]]); };
const Kof = (vfov, w, h) => { const f = h / 2 / Math.tan(THREE.MathUtils.degToRad(vfov) / 2); return [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]; };

function camerasJson() {
  const W = parseInt($("outW").value), H = parseInt($("outH").value);
  const inputs = views.map((v, i) => {
    const p = new THREE.Vector3(), q = new THREE.Quaternion(); v.c2w.decompose(p, q, new THREE.Vector3());
    return { image: `inputs/${String(i).padStart(2, "0")}.png`, name: v.name, c2w: rows(cvPose(p, q)), K: v.header.K,
             width: v.header.width, height: v.header.height, scale: v.scale, model: v.header.model };
  });
  const c2w = pathFrames.map((p) => rows(cvPose(p.position, p.quaternion)));
  const K = pathFrames.map((p) => Kof(p.fov, W, H));
  return {
    convention: "OpenCV c2w (x right, y down, z forward); world = the first input view's camera frame; metres (MoGe metric scale)",
    fps: parseInt($("fps").value), inputs,
    targets: { path: { kind: "views", width: W, height: H, c2w, K, guide: pathFrames.map((_, i) => `guide/${String(i).padStart(4, "0")}.png`) } },
    keyframes: keys.map((k) => ({ c2w: rows(cvPose(k.position, k.quaternion)), vfov: k.fov })),
  };
}

async function putFile(name, path, body) {
  const r = await fetch(`api/export/${encodeURIComponent(name)}?path=${encodeURIComponent(path)}`, { method: "PUT", body });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

const NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;
const nameValid = () => NAME_RE.test($("exportName").value.trim()) && !$("exportName").value.trim().startsWith(".");
let exporting = false, lastExport = "";

async function exportToServer() {
  if (!pathFrames.length) return toast("Add at least one keyframe first");
  if (!nameValid()) { $("exportName").focus(); return toast("Fix the export name first"); }
  const name = $("exportName").value.trim();
  const W = parseInt($("outW").value), H = parseInt($("outH").value);
  const total = views.length + pathFrames.length + 1;
  let done = 0;
  const status = $("exportStatus"), bar = $("exportProgress");
  const progress = (what) => {
    const frac = done / total, el = performance.now() - activity.t0;
    const eta = done > 2 ? ` · about ${Math.ceil((el / done) * (total - done) / 1000)} s left` : "";
    jobProgress(frac, `${what}${eta}`);
    bar.firstElementChild.style.width = `${(100 * frac).toFixed(1)}%`;
  };
  exporting = true; bar.hidden = false; status.className = "status-line";
  jobStart(`Exporting "${name}"`, "", true);
  try {
    for (let i = 0; i < views.length; i++) {
      const blob = await (await fetch(views[i].dataURL)).blob();
      await putFile(name, `inputs/${String(i).padStart(2, "0")}.png`, blob);
      done++; progress(`input image ${i + 1} / ${views.length}`);
    }
    for (let i = 0; i < pathFrames.length; i++) {
      setPathCamera(i);
      await putFile(name, `guide/${String(i).padStart(4, "0")}.png`, await renderOffscreen(pathCamera, W, H));
      done++; progress(`guide frame ${i + 1} / ${pathFrames.length} at ${W}×${H}`);
      status.textContent = `guide ${i + 1} / ${pathFrames.length}`;
    }
    const r = await putFile(name, "cameras.json", new Blob([JSON.stringify(camerasJson(), null, 1)], { type: "application/json" }));
    done++;
    status.textContent = `exported ${pathFrames.length} frames → ${r.dir}`;
    status.className = "status-line ok";
    lastExport = name;
    jobDone(`Exported ${pathFrames.length} frames`, r.dir);
  } catch (e) {
    status.textContent = `export failed: ${e.message}`;
    status.className = "status-line err";
    jobFail(`Export "${name}" failed`, e.message, exportToServer);
  } finally { exporting = false; bar.hidden = true; updateUi(); }
}

function exportPly() {
  let n = 0;
  for (const v of views) if (v.visible) n += v.points.length / 3;
  const header = `ply\nformat binary_little_endian 1.0\ncomment OpenCV world = first input camera\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const head = new TextEncoder().encode(header), body = new ArrayBuffer(n * 15), dv = new DataView(body);
  let o = 0;
  for (const v of views) {
    if (!v.visible) continue;
    const m = new THREE.Matrix4().copy(FLIP).multiply(v.object.matrix), e = m.elements, P = v.points, C = v.colors;
    for (let i = 0; i < P.length / 3; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      dv.setFloat32(o, e[0] * x + e[4] * y + e[8] * z + e[12], true);
      dv.setFloat32(o + 4, e[1] * x + e[5] * y + e[9] * z + e[13], true);
      dv.setFloat32(o + 8, e[2] * x + e[6] * y + e[10] * z + e[14], true);
      dv.setUint8(o + 12, C[3 * i]); dv.setUint8(o + 13, C[3 * i + 1]); dv.setUint8(o + 14, C[3 * i + 2]);
      o += 15;
    }
  }
  download("points.ply", new Blob([head, body]));
  toast(`Downloaded points.ply (${(n / 1e6).toFixed(2)}M points)`);
}

// ---------------------------------------------------------------- project save / load
function saveProject() {
  const project = {
    version: 1,
    views: views.map((v) => ({ name: v.name, dataURL: v.dataURL, header: v.header, scale: v.scale, visible: v.visible,
      c2w: v.c2w.elements, index: b64.enc(v.index), points: b64.enc(v.points), colors: b64.enc(v.colors) })),
    keys: keys.map((k) => ({ position: k.position.toArray(), quaternion: k.quaternion.toArray(), fov: k.fov })),
    settings: Object.fromEntries(["frames", "fps", "outW", "outH", "exportName", "ahead"].map((id) => [id, $(id).value]).concat([["faceForward", $("faceForward").checked]])),
    viewport: { position: camera.position.toArray(), target: controls.target.toArray(), fov: camera.fov },
  };
  const file = `${$("exportName").value || "project"}.pctool.json`;
  download(file, new Blob([JSON.stringify(project)], { type: "application/json" }));
  toast(`Saved ${file}`);
}

async function loadProject(file) {
  let p;
  try { p = JSON.parse(await file.text()); } catch (e) { return jobFail(`Couldn't open ${file.name}`, `Not a project file: ${e.message}`); }
  clearViews(); keys.length = 0; guidePose = null;
  for (const s of p.views) {
    const v = { name: s.name, dataURL: s.dataURL, header: s.header, scale: s.scale, visible: s.visible, id: viewCounter++,
      index: new Uint32Array(b64.dec(s.index)), points: new Float32Array(b64.dec(s.points)), colors: new Uint8Array(b64.dec(s.colors)),
      c2w: new THREE.Matrix4().fromArray(s.c2w) };
    buildViewObject(v); views.push(v);
  }
  for (const k of p.keys) keys.push({ position: new THREE.Vector3().fromArray(k.position), quaternion: new THREE.Quaternion().fromArray(k.quaternion), fov: k.fov });
  for (const [id, val] of Object.entries(p.settings || {})) { if (id === "faceForward") $(id).checked = val; else $(id).value = val; }
  touched.size = touched.exportName = true;
  if (p.viewport) {
    camera.position.fromArray(p.viewport.position); controls.target.fromArray(p.viewport.target);
    camera.fov = p.viewport.fov; camera.updateProjectionMatrix(); $("fov").value = camera.fov.toFixed(0); controls.update(); updateFovLabel();
  }
  views.forEach(updateViewMatrix); renderViewList(); selectKey(-1); updateGuideSize();
  toast(`Opened ${file.name}: ${views.length} views, ${keys.length} keyframes`);
}

// ---------------------------------------------------------------- defaults
const touched = { size: false, exportName: false };   // user-edited fields keep their values
const even = (x) => Math.max(64, 2 * Math.round(x / 2));
function matchInputAspect() {
  const v = views[0];
  if (!v) return;
  const a = v.header.width / v.header.height;
  if (a >= 1) { $("outW").value = 1280; $("outH").value = even(1280 / a); }
  else { $("outH").value = 1280; $("outW").value = even(1280 * a); }
  updateGuideSize();
}
function exportNameFrom(file, multi) {
  const stem = (file || "").replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[.-]+|-+$/g, "").slice(0, 40);
  return stem ? `${stem}${multi ? "-multi" : ""}-path` : "path01";
}
function updateGuideSize() { $("guideSize").textContent = `${$("outW").value}×${$("outH").value} PNG, transparent where there are no points`; }

// ---------------------------------------------------------------- UI state
function updateUi() {
  const hasScene = views.length > 0, hasPath = pathFrames.length > 0, busy = activity.running && !exporting;
  $("empty").hidden = hasScene;
  $("hud").hidden = !hasScene;
  $("exportHint").hidden = hasPath;
  const step = (id, { locked = false, done = false, current = false }) => {
    const el = $(id); el.classList.toggle("locked", locked); el.classList.toggle("done", done); el.classList.toggle("current", current);
  };
  step("step1", { done: hasScene, current: !hasScene });
  step("step2", { locked: !hasScene, done: views.length > 1 });
  step("step3", { locked: !hasScene, done: keys.length >= 2, current: hasScene && keys.length < 2 });
  step("step4", { done: !!lastExport, current: keys.length >= 2 });
  const pts = views.reduce((a, v) => a + v.header.count, 0);
  $("badge1").textContent = hasScene ? `${views.length} view${views.length > 1 ? "s" : ""} · ${(pts / 1e6).toFixed(2)}M pts` : "";
  $("badge2").textContent = hasScene ? (views.length > 1 ? `${views.length} views` : "optional") : "";
  $("badge3").textContent = keys.length ? `${keys.length} key${keys.length > 1 ? "s" : ""} · ${pathFrames.length} frames` : "";
  $("badge4").textContent = lastExport ? `✓ ${lastExport}` : `${$("outW").value}×${$("outH").value}`;
  for (const id of ["btnFirst", "btnMulti", "emptyFirst", "emptyMulti"]) $(id).disabled = busy;
  $("btnAdd").disabled = !hasScene || busy;
  $("btnGuide").disabled = !hasScene;
  $("btnKey").disabled = !hasScene;
  $("btnClearPath").disabled = !keys.length;
  $("btnFrameAll").disabled = !hasScene;
  $("btnExport").disabled = !hasPath || exporting || !nameValid();
  $("btnCams").disabled = !hasScene;
  $("btnPly").disabled = !hasScene;
  $("btnSave").disabled = !hasScene && !keys.length;
  $("btnPlay").disabled = !hasPath;
  $("timeline").classList.toggle("empty", !hasPath);
  $("nameError").hidden = nameValid();
  if (!hasPath && playing) setPlaying(false);
}

function showHelp(show) { $("help").hidden = !show; if (show) $("helpClose").focus(); }

// ---------------------------------------------------------------- drag and drop
// Empty scene: one image = lift, several = multi-view. With a scene: drop on "New scene" or "Add at this view".
// A .json file opens a project.
let dragDepth = 0;
const dragCount = (e) => [...(e.dataTransfer?.items || [])].filter((i) => i.kind === "file").length;
const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
function dropZoneAt(e) {
  const z = e.target.closest?.("#dropOverlay .zone");
  return z && !z.classList.contains("disabled") ? z.dataset.zone : null;
}
function showDropUi(e, on) {
  const n = dragCount(e), overlay = $("dropOverlay");
  if (!views.length) { $("dropZone").classList.toggle("over", on); overlay.hidden = true; return; }
  overlay.hidden = !on;
  if (!on) return;
  overlay.querySelector('[data-zone="new"] .zone-sub').textContent =
    n > 1 ? `Reconstruct these ${n} photos together (replaces the scene and its keyframes)` : "Lift this image as a new scene (replaces the scene and its keyframes)";
  overlay.querySelector('[data-zone="add"]').classList.toggle("disabled", n > 1);
  overlay.querySelectorAll(".zone").forEach((z) => z.classList.toggle("over", z.dataset.zone === dropZoneAt(e)));
}
stage.addEventListener("dragenter", (e) => { if (!isFileDrag(e)) return; e.preventDefault(); dragDepth++; showDropUi(e, true); });
stage.addEventListener("dragover", (e) => { if (!isFileDrag(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; showDropUi(e, true); });
stage.addEventListener("dragleave", (e) => { if (!isFileDrag(e)) return; if (--dragDepth <= 0) { dragDepth = 0; showDropUi(e, false); } });
stage.addEventListener("drop", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  const zone = dropZoneAt(e);
  dragDepth = 0; showDropUi(e, false);
  handleFiles([...e.dataTransfer.files], views.length ? zone : "new");
});
window.addEventListener("dragover", (e) => e.preventDefault());   // never navigate away to a dropped file
window.addEventListener("drop", (e) => e.preventDefault());

function handleFiles(files, zone) {
  const project = files.find((f) => f.name.endsWith(".json"));
  if (project) return loadProject(project);
  const images = files.filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|gif|tiff?|heic)$/i.test(f.name));
  if (!images.length) return toast("Drop image files (or a .pctool.json project)");
  if (!zone) return;
  if (zone === "add") return runLiftAdd(images[0]);
  if (images.length === 1) return runLiftNew(images[0]);
  return runMulti(images);
}

// ---------------------------------------------------------------- UI wiring
function updateFovLabel() { $("fovVal").textContent = `${(+$("fov").value).toFixed(0)}°`; }

$("fov").oninput = () => { camera.fov = +$("fov").value; camera.updateProjectionMatrix(); updateFovLabel(); };
$("pointSize").oninput = () => { uniforms.sizeScale.value = +$("pointSize").value; $("pointSizeVal").textContent = $("pointSize").value; };
$("showFrusta").onchange = () => views.forEach(updateViewMatrix);
$("btnFirst").onclick = $("emptyFirst").onclick = () => $("fileFirst").click();
$("btnMulti").onclick = $("emptyMulti").onclick = () => $("fileMulti").click();
$("btnAdd").onclick = () => { if (!views.length) return toast("Lift a first image before adding views to it"); $("fileAdd").click(); };
$("fileMulti").onchange = (e) => { const files = [...e.target.files]; e.target.value = ""; if (files.length) runMulti(files); };
$("fileFirst").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) runLiftNew(f); };
$("fileAdd").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) runLiftAdd(f); };
$("btnGuide").onclick = async () => {
  const W = parseInt($("outW").value), H = parseInt($("outH").value);
  const c = camera.clone(); c.updateMatrixWorld();
  guidePose = { ...poseHere(), target: controls.target.clone() };
  download("guide.png", await renderOffscreen(c, W, H));
  toast(`Saved guide.png (${W}×${H}). Generate an image from it, then add it here without moving the camera.`, 6000);
};
$("btnGuideReturn").onclick = () => {
  if (!guidePose) return;
  lookFrom(guidePose); controls.target.copy(guidePose.target); camera.quaternion.copy(guidePose.quaternion);
};
$("btnKey").onclick = addKey;
$("btnFrameAll").onclick = frameAll;
$("btnClearPath").onclick = () => {
  const n = keys.length; keys.length = 0; selectKey(-1);
  if (n) toast(`Cleared ${n} keyframe${n > 1 ? "s" : ""}`);
};
for (const id of ["frames", "faceForward", "outW", "outH"]) $(id).onchange = () => { rebuildPath(); updateGuideSize(); };
$("outW").oninput = $("outH").oninput = () => { touched.size = true; };
$("btnMatchAspect").onclick = () => { if (!views.length) return toast("Lift an image first"); matchInputAspect(); touched.size = true; rebuildPath(); };
$("exportName").oninput = () => { touched.exportName = true; updateUi(); };
$("fps").onchange = () => { onFrameChange(); renderKeyList(); };
$("ahead").onchange = updatePathAhead;
$("frame").oninput = onFrameChange;
$("btnPlay").onclick = () => setPlaying(!playing);
$("lookThrough").onchange = () => $("lookThrough").blur();
$("pip").onclick = () => { $("lookThrough").checked = true; };
$("btnExport").onclick = exportToServer;
$("btnCams").onclick = () => download("cameras.json", new Blob([JSON.stringify(camerasJson(), null, 1)], { type: "application/json" }));
$("btnPly").onclick = exportPly;
$("btnSave").onclick = saveProject;
$("btnLoad").onclick = $("emptyLoad").onclick = () => $("fileProject").click();
$("fileProject").onchange = async (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) await loadProject(f); };
$("btnHelp").onclick = $("hintKeys").onclick = () => showHelp(true);
$("helpClose").onclick = () => showHelp(false);
$("help").onclick = (e) => { if (e.target === $("help")) showHelp(false); };

canvas.addEventListener("dblclick", (e) => {   // orbit around the clicked point
  const r = canvas.getBoundingClientRect(), ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, camera);
  ray.params.Points.threshold = 0.01 * sceneRadius();
  const hits = ray.intersectObjects(views.filter((v) => v.visible).map((v) => v.object), false);
  if (hits.length) { controls.target.copy(hits[0].point); controls.update(); }
});

// ---------------------------------------------------------------- GPU service health
const health = { ok: true, models: [] };
async function checkHealth() {
  try {
    const r = await fetch("api/health"), j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.detail || r.statusText);
    health.ok = true; health.models = j.models;
    $("status").className = "ok"; $("status").textContent = `GPU service · ${j.device}`;
    $("status").title = `Loaded models: ${j.models.join(", ") || "none yet"}`;
    if (!$("model").options.length) for (const m of j.available) $("model").add(new Option(m, m, false, m === j.default));
    $("emptyError").hidden = true;
  } catch (e) {
    health.ok = false;
    $("status").className = "err"; $("status").textContent = "GPU service unavailable";
    $("status").title = e.message;
    if (!$("model").options.length) $("model").add(new Option("moge-3-vitl", "moge-3-vitl"));
    const note = $("emptyError");
    note.hidden = false;
    note.innerHTML = `<span>The GPU service is not reachable (${esc(e.message)}), so images can't be lifted right now. You can still open a saved project.</span>`;
    const b = document.createElement("button"); b.className = "ghost small"; b.textContent = "Check again"; b.onclick = checkHealth;
    note.appendChild(b);
  }
}

resize();
updateFovLabel();
updateGuideSize();
$("pointSizeVal").textContent = $("pointSize").value;
rebuildPath();
checkHealth();
setInterval(checkHealth, 30000);
animate();
window.pctool = { THREE, scene, camera, controls, views, keys, pathGroup, aheadGroup, get pathFrames() { return pathFrames; },   // console debugging
  get keyFrames() { return keyFrames; }, addKey, selectKey, setFrame, handleFiles };
