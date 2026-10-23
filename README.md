# pointcloud-path-tool

A small browser tool for planning camera paths through a scene that started as a single image.

1. **Lift.** An image becomes a metric point cloud with [MoGe](https://github.com/microsoft/MoGe). MoGe-3 is the default; MoGe-2 is also available. Several photos of one place (cameras unknown) are reconstructed together: [VGGT](https://github.com/facebookresearch/vggt) estimates every photo's camera and field of view, MoGe lifts each photo at that field of view, and per-photo depth ratios put cameras and points in metres in the first photo's frame.
2. **Grow.** Fly to a new viewpoint and download the point render there. Make an image for that view with any generator, then add it back. It is lifted with the viewport's field of view, placed at the viewport camera, and scale-aligned to the existing points.
3. **Plan.** Place keyframes and edit them with a gizmo. The camera path is a centripetal Catmull–Rom spline with eased rotation interpolation, or it can face along the path. A picture-in-picture shows what the path camera sees, and frustums for the frames ahead are drawn in the scene.
4. **Export.** The export contains per-frame OpenCV cameras and point-render guide frames (RGBA; alpha marks covered pixels), ready to condition a novel-view or video model. A merged PLY and a self-contained project file are also available.

Rendering is plain three.js, with points drawn as discs sized by each point's pixel footprint in its source view. Everything else is a small FastAPI server.

## Run

```bash
pip install -e .            # web server only
pip install -e ".[moge]"    # + MoGe for lifting in-process (needs a GPU for interactive speed)
pip install -e ".[moge,multiview]"   # + VGGT for multi-photo scenes (VGGT-1B weights: non-commercial licence)
# MoGe-3 additionally needs FlexGEMM: pip install --no-build-isolation git+https://github.com/JeffreyXiang/FlexGEMM.git
```

Everything on one GPU machine:

```bash
pctool-server --port 23200            # open http://localhost:23200
```

The web UI on one machine and lifting on a GPU box:

```bash
# on the GPU box
pctool-service --host 127.0.0.1 --port 23190
# locally, with a tunnel (ssh -L 23190:127.0.0.1:23190 gpu-box, kubectl port-forward, ...)
pctool-server --port 23200 --backend http://127.0.0.1:23190
```

Lifting a 1024 px image takes about 0.4 s with MoGe-2 ViT-L and 0.7 s with MoGe-3 ViT-L on a recent datacenter GPU, plus model loading on first use.

## Controls

| | |
| --- | --- |
| drag / right-drag / wheel | orbit / pan / zoom |
| W A S D, Q E (shift: faster) | fly |
| double-click a point | orbit around it |
| drop files on the viewport | one image: lift · several: multi-view · `.pctool.json`: open project |
| K | add a keyframe at the current view |
| T / R / Esc | gizmo: move / rotate / deselect the edited keyframe |
| Delete | delete the selected keyframe |
| Space | play / pause the path |
| ← / → (shift: 10), [ / ] | previous / next frame, previous / next keyframe |
| L | look through the path camera (letterboxed to the output size) |
| F | frame the path and the input cameras |
| ? | show all shortcuts |

## Export format

**Export to server** writes `<exports>/<name>/`, containing:
- `cameras.json`;
- `inputs/NN.png`: the lifted images;
- `guide/NNNN.png`: one point render per path frame, at the chosen output size.

`cameras.json`:

```jsonc
{
  "convention": "OpenCV c2w (x right, y down, z forward); world = the first input view's camera frame; metres",
  "fps": 24,
  "inputs": [{"image": "inputs/00.png", "c2w": [[...4x4...]], "K": [[...3x3...]], "width": 1024, "height": 576, "scale": 1.0, "model": "moge-3-vitl"}],
  "targets": {"path": {"kind": "views", "width": 1280, "height": 720, "c2w": [...per frame...], "K": [...per frame...], "guide": ["guide/0000.png", ...]}},
  "keyframes": [{"c2w": [[...]], "vfov": 50.0}]
}
```

Notes on the fields:
- `K` is in pixels.
- Input `K` refers to the lifted resolution, which is the image downscaled to at most *Max side*.
- Path frames use square pixels and a centred principal point.

## Layout

```
pctool/lift.py      MoGe model cache and the binary lift format
pctool/multiview.py VGGT cameras + per-photo MoGe points for several photos
pctool/service.py   GPU service (GET /health, POST /lift, POST /lift_multi)
pctool/server.py    web server: static files, export writer, in-process or proxied lifting
pctool/web/         index.html, app.js (all client logic), style.css, vendored three.js r170 (MIT)
```
