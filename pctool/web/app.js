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

// ---------------------------------------------------------------- renderer, scene, viewport camera
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x07080b, 1);
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

// ---------------------------------------------------------------- views (lifted images)
const views = [];   // {name, dataURL, header, index: Uint32Array, points: Float32Array, colors: Uint8Array, c2w: Matrix4, scale, visible, object, frustum}
let viewCounter = 0;

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

async function liftImage(file, fovX) {
  const form = new FormData();
  form.append("image", file);
  form.append("model", $("model").value);
  form.append("max_side", $("maxSide").value);
  if (fovX) form.append("fov_x", String(fovX));
  const r = await fetch("api/lift", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
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
  view.frustum = frustum(vfov, aspect, 0.25, 0xf0c674, tex);
  view.frustum.matrixAutoUpdate = false;
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
}

// Several photos with unknown cameras: the service estimates cameras (VGGT) and lifts each photo (MoGe); each blob's
// header carries its OpenCV c2w (world = first photo's camera, metres) and the scale for its points.
async function addMultiView(files) {
  const form = new FormData();
  for (const f of files) form.append("images", f);
  form.append("model", $("model").value);
  form.append("max_side", $("maxSide").value);
  toast(`reconstructing ${files.length} photos (cameras + points)…`, 600000);
  const t0 = performance.now();
  const r = await fetch("api/lift_multi", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  const buf = await r.arrayBuffer(), dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  clearViews();
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
  views.forEach(updateViewMatrix);
  resetViewToInput(views[0]);
  renderViewList();
  const pts = views.reduce((a, v) => a + v.header.count, 0);
  toast(`${n} photos → ${(pts / 1e6).toFixed(1)}M points in ${((performance.now() - t0) / 1000).toFixed(1)} s · per-photo scale ${views.map((v) => v.scale.toFixed(2)).join(", ")}`, 8000);
}

async function addView(file, placeAtViewport) {
  const dataURL = await fileToDataURL(file);
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = dataURL; });
  const aspect = img.width / img.height;
  const fovX = placeAtViewport ? hfovOf(camera.fov, aspect) : null;
  toast(`lifting ${file.name} with ${$("model").value}…`, 60000);
  const t0 = performance.now();
  const lifted = await liftImage(file, fovX);
  const view = { name: file.name || `view${viewCounter}`, dataURL, ...lifted, scale: 1, visible: true, id: viewCounter++ };
  view.c2w = placeAtViewport ? camera.matrixWorld.clone() : new THREE.Matrix4();
  buildViewObject(view);
  views.push(view);
  views.forEach(updateViewMatrix);
  let msg = `${view.name}: ${lifted.header.count.toLocaleString()} points, fov_x ${lifted.header.fov_x.toFixed(1)}°, ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  if (placeAtViewport && views.length > 1 && $("autoAlign").checked) {
    const { scale, support } = alignScale(view);
    if (scale) { view.scale = scale; updateViewMatrix(view); msg += ` · scale ×${scale.toFixed(3)} from ${support} px`; }
    else msg += ` · too little overlap to align (${support} px): scale 1`;
  }
  if (views.length === 1) resetViewToInput(view);
  renderViewList();
  toast(msg);
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

function renderViewList() {
  const box = $("views");
  box.innerHTML = "";
  for (const v of views) {
    const d = document.createElement("div");
    d.className = "item";
    d.innerHTML = `<img src="${v.dataURL}"><span class="name" title="${v.name}">${v.name} · ${(v.header.count / 1e3).toFixed(0)}k</span>
      <input type="number" step="0.01" value="${v.scale.toFixed(3)}" title="scale">
      <button data-a="go" title="look from this camera">◎</button><button data-a="vis">${v.visible ? "hide" : "show"}</button><button data-a="del">✕</button>`;
    d.querySelector("input").onchange = (e) => { v.scale = parseFloat(e.target.value) || 1; updateViewMatrix(v); };
    d.querySelector('[data-a="go"]').onclick = () => resetViewToInput(v);
    d.querySelector('[data-a="vis"]').onclick = () => { v.visible = !v.visible; updateViewMatrix(v); renderViewList(); };
    d.querySelector('[data-a="del"]').onclick = () => {
      scene.remove(v.object); scene.remove(v.frustum); disposeTree(v.object); disposeTree(v.frustum);
      views.splice(views.indexOf(v), 1); renderViewList();
    };
    box.appendChild(d);
  }
}

// ---------------------------------------------------------------- camera path
const keys = [];   // {position, quaternion, fov}
let selectedKey = -1;
const pathGroup = new THREE.Group();
scene.add(pathGroup);
let pathFrames = [];   // {position, quaternion, fov}
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

function addKey() {
  keys.push({ position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov: camera.fov });
  rebuildPath(); renderKeyList();
}

function selectKey(i) {
  selectedKey = i;
  if (i < 0) gizmo.detach();
  else { gizmoTarget.position.copy(keys[i].position); gizmoTarget.quaternion.copy(keys[i].quaternion); gizmo.attach(gizmoTarget); }
  renderKeyList();
}

function renderKeyList() {
  const box = $("keys");
  box.innerHTML = "";
  keys.forEach((k, i) => {
    const d = document.createElement("div");
    d.className = "item" + (i === selectedKey ? " sel" : "");
    d.innerHTML = `<span class="name">key ${i} · fov ${k.fov.toFixed(0)}°</span>
      <button data-a="sel">${i === selectedKey ? "done" : "edit"}</button><button data-a="go">◎</button>
      <button data-a="set" title="replace with the current view">⟲</button><button data-a="del">✕</button>`;
    d.querySelector('[data-a="sel"]').onclick = () => selectKey(i === selectedKey ? -1 : i);
    d.querySelector('[data-a="go"]').onclick = () => lookFrom(k);
    d.querySelector('[data-a="set"]').onclick = () => { keys[i] = { position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov: camera.fov }; selectKey(-1); rebuildPath(); };
    d.querySelector('[data-a="del"]').onclick = () => { keys.splice(i, 1); selectKey(-1); rebuildPath(); };
    box.appendChild(d);
  });
}

function lookFrom(pose) {
  camera.position.copy(pose.position); camera.quaternion.copy(pose.quaternion);
  camera.fov = pose.fov; camera.updateProjectionMatrix(); $("fov").value = pose.fov.toFixed(0); updateFovLabel();
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, Math.max(0.1, 0.3 * sceneRadius()));
  camera.lookAt(controls.target);   // keeps OrbitControls consistent (roll is dropped while orbiting)
  camera.quaternion.copy(pose.quaternion);
}

function interpolate(n) {
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
  return out;
}

function rebuildPath() {
  for (const c of [...pathGroup.children]) { pathGroup.remove(c); disposeTree(c); }
  const n = Math.max(2, parseInt($("frames").value) || 120);
  pathFrames = interpolate(n);
  $("frame").max = String(Math.max(0, pathFrames.length - 1));
  if (pathFrames.length) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pathFrames.map((p) => p.position)),
      new THREE.LineBasicMaterial({ color: 0xe8f0ff }));
    pathGroup.add(line);
    const aspect = outAspect(), depth = Math.max(0.03, 0.05 * sceneRadius());
    keys.forEach((k, i) => {
      const f = frustum(k.fov, aspect, depth * 1.4, i === selectedKey ? 0xf0c674 : 0x7fb2ff);
      f.position.copy(k.position); f.quaternion.copy(k.quaternion); pathGroup.add(f);
    });
  }
  pathGroup.traverse((o) => o.layers.set(HELPERS));
  updatePathAhead();
  renderKeyList();
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

// ---------------------------------------------------------------- rendering: viewport, picture-in-picture, offscreen
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

const clock = new THREE.Clock();
let playing = false, playAccum = 0;
const keysDown = new Set();
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  keysDown.add(e.key.toLowerCase());
  if (e.key === "t") gizmo.setMode("translate");
  if (e.key === "r") gizmo.setMode("rotate");
  if (e.key === "Escape") selectKey(-1);
  if (e.key === "k") addKey();
});
window.addEventListener("keyup", (e) => keysDown.delete(e.key.toLowerCase()));

function fly(dt) {
  const speed = sceneRadius() * (keysDown.has("shift") ? 1.5 : 0.4) * dt;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion), right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion), d = new THREE.Vector3();
  if (keysDown.has("w")) d.add(fwd); if (keysDown.has("s")) d.sub(fwd);
  if (keysDown.has("d")) d.add(right); if (keysDown.has("a")) d.sub(right);
  if (keysDown.has("e")) d.add(up); if (keysDown.has("q")) d.sub(up);
  if (d.lengthSq()) { d.normalize().multiplyScalar(speed); camera.position.add(d); controls.target.add(d); }
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
  const look = $("lookThrough").checked && pathFrames.length;
  if (!look) controls.update();
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (look) {
    setPathCamera(currentFrame());
    const view = pathCamera.clone(); view.aspect = canvas.clientWidth / canvas.clientHeight; view.updateProjectionMatrix();
    setProjScale(view, size.y); renderer.render(scene, view);
  } else {
    setProjScale(camera, size.y); renderer.render(scene, camera);
  }
  if (pathFrames.length && !look) {   // picture-in-picture: the path camera's view of the points
    const pw = Math.round(canvas.clientWidth * 0.28), ph = Math.round(pw / outAspect()), x = canvas.clientWidth - pw - 12, y = 28;
    setPathCamera(currentFrame());
    renderer.setScissorTest(true); renderer.setScissor(x, y, pw, ph); renderer.setViewport(x, y, pw, ph);
    renderer.setClearColor(0x000000, 1); renderer.clear();
    setProjScale(pathCamera, ph * renderer.getPixelRatio()); renderer.render(scene, pathCamera);
    renderer.setScissorTest(false); renderer.setClearColor(0x07080b, 1);
    $("pipLabel").style.display = "block";
    $("pipLabel").textContent = `path camera · frame ${currentFrame()} / ${pathFrames.length - 1}`;
  } else $("pipLabel").style.display = "none";
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
  renderer.setRenderTarget(null); renderer.setClearColor(0x07080b, 1);
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) img.data.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);   // GL rows are bottom-up
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob(res, "image/png"));
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

async function exportToServer() {
  if (!pathFrames.length) return toast("add at least one keyframe first");
  const name = $("exportName").value.trim();
  const W = parseInt($("outW").value), H = parseInt($("outH").value);
  $("btnExport").disabled = true;
  try {
    for (let i = 0; i < views.length; i++) {
      const blob = await (await fetch(views[i].dataURL)).blob();
      await putFile(name, `inputs/${String(i).padStart(2, "0")}.png`, blob);
    }
    for (let i = 0; i < pathFrames.length; i++) {
      setPathCamera(i);
      await putFile(name, `guide/${String(i).padStart(4, "0")}.png`, await renderOffscreen(pathCamera, W, H));
      if (i % 10 === 0) $("exportStatus").textContent = `guide ${i + 1} / ${pathFrames.length}`;
    }
    const r = await putFile(name, "cameras.json", new Blob([JSON.stringify(camerasJson(), null, 1)], { type: "application/json" }));
    $("exportStatus").textContent = `exported ${pathFrames.length} frames → ${r.dir}`;
  } catch (e) {
    $("exportStatus").textContent = `export failed: ${e.message}`;
  } finally { $("btnExport").disabled = false; }
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
  download(`${$("exportName").value || "project"}.pctool.json`, new Blob([JSON.stringify(project)], { type: "application/json" }));
}

async function loadProject(file) {
  const p = JSON.parse(await file.text());
  clearViews(); keys.length = 0;
  for (const s of p.views) {
    const v = { name: s.name, dataURL: s.dataURL, header: s.header, scale: s.scale, visible: s.visible, id: viewCounter++,
      index: new Uint32Array(b64.dec(s.index)), points: new Float32Array(b64.dec(s.points)), colors: new Uint8Array(b64.dec(s.colors)),
      c2w: new THREE.Matrix4().fromArray(s.c2w) };
    buildViewObject(v); views.push(v);
  }
  for (const k of p.keys) keys.push({ position: new THREE.Vector3().fromArray(k.position), quaternion: new THREE.Quaternion().fromArray(k.quaternion), fov: k.fov });
  for (const [id, val] of Object.entries(p.settings || {})) { if (id === "faceForward") $(id).checked = val; else $(id).value = val; }
  if (p.viewport) {
    camera.position.fromArray(p.viewport.position); controls.target.fromArray(p.viewport.target);
    camera.fov = p.viewport.fov; camera.updateProjectionMatrix(); $("fov").value = camera.fov.toFixed(0); controls.update(); updateFovLabel();
  }
  views.forEach(updateViewMatrix); renderViewList(); selectKey(-1); rebuildPath();
  toast(`loaded ${views.length} views, ${keys.length} keyframes`);
}

// ---------------------------------------------------------------- UI wiring
function updateFovLabel() { $("fovVal").textContent = `${(+$("fov").value).toFixed(0)}°`; }
function onFrameChange() { $("frameVal").textContent = `${currentFrame()}`; updatePathAhead(); }

$("fov").oninput = () => { camera.fov = +$("fov").value; camera.updateProjectionMatrix(); updateFovLabel(); };
$("pointSize").oninput = () => { uniforms.sizeScale.value = +$("pointSize").value; $("pointSizeVal").textContent = $("pointSize").value; };
$("showFrusta").onchange = () => views.forEach(updateViewMatrix);
$("btnFirst").onclick = () => $("fileFirst").click();
$("btnAdd").onclick = () => { if (!views.length) return toast("lift a first image first"); $("fileAdd").click(); };
$("btnMulti").onclick = () => $("fileMulti").click();
$("fileMulti").onchange = async (e) => {
  const files = [...e.target.files]; e.target.value = "";
  if (files.length < 2) return toast("choose at least 2 photos of the same place");
  try { await addMultiView(files); } catch (err) { toast(`reconstruction failed: ${err.message}`, 8000); }
};
$("fileFirst").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  clearViews();
  try { await addView(f, false); } catch (err) { toast(`lift failed: ${err.message}`, 8000); }
};
$("fileAdd").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) try { await addView(f, true); } catch (err) { toast(`lift failed: ${err.message}`, 8000); }
};
$("btnGuide").onclick = async () => {
  const W = parseInt($("outW").value), H = parseInt($("outH").value);
  const c = camera.clone(); c.updateMatrixWorld();
  download("guide.png", await renderOffscreen(c, W, H));
  toast(`guide ${W}×${H} at the current view (alpha = covered). Generate an image there, then "Add at current view" with the same viewport.`, 7000);
};
$("btnKey").onclick = addKey;
$("btnClearPath").onclick = () => { keys.length = 0; selectKey(-1); rebuildPath(); };
for (const id of ["frames", "faceForward", "outW", "outH"]) $(id).onchange = rebuildPath;
$("ahead").onchange = updatePathAhead;
$("frame").oninput = onFrameChange;
$("btnPlay").onclick = () => { playing = !playing; $("btnPlay").textContent = playing ? "❚❚ Pause" : "▶ Play"; };
$("btnExport").onclick = exportToServer;
$("btnCams").onclick = () => download("cameras.json", new Blob([JSON.stringify(camerasJson(), null, 1)], { type: "application/json" }));
$("btnPly").onclick = exportPly;
$("btnSave").onclick = saveProject;
$("btnLoad").onclick = () => $("fileProject").click();
$("fileProject").onchange = async (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) await loadProject(f); };

canvas.addEventListener("dblclick", (e) => {   // orbit around the clicked point
  const r = canvas.getBoundingClientRect(), ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, camera);
  ray.params.Points.threshold = 0.01 * sceneRadius();
  const hits = ray.intersectObjects(views.filter((v) => v.visible).map((v) => v.object), false);
  if (hits.length) { controls.target.copy(hits[0].point); controls.update(); }
});

async function checkHealth() {
  try {
    const r = await fetch("api/health"), j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.detail || r.statusText);
    $("status").className = "ok"; $("status").textContent = `GPU service: ${j.device} · loaded: ${j.models.join(", ") || "none yet"}`;
    if (!$("model").options.length) for (const m of j.available) $("model").add(new Option(m, m, false, m === j.default));
  } catch (e) {
    $("status").className = "err"; $("status").textContent = `GPU service unavailable: ${e.message}`;
    if (!$("model").options.length) $("model").add(new Option("moge-3-vitl", "moge-3-vitl"));
  }
}

resize();
updateFovLabel();
$("pointSizeVal").textContent = $("pointSize").value;
onFrameChange();
checkHealth();
setInterval(checkHealth, 30000);
animate();
window.pctool = { THREE, scene, camera, controls, views, keys, pathGroup, aheadGroup, get pathFrames() { return pathFrames; } };   // console debugging
