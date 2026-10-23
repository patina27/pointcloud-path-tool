"""Web server: serves the tool, writes exports, and lifts images either in-process (MoGe on this machine's GPU) or
by forwarding to a remote `pctool-service` (--backend URL).

    pctool-server --port 23200                                  # lift in-process
    pctool-server --port 23200 --backend http://127.0.0.1:23190 # lift on a GPU box (e.g. through an ssh/kubectl tunnel)
"""
import argparse
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).resolve().parent / "web"


def create_app(backend: str | None, exports: Path) -> FastAPI:
    app = FastAPI(title="pctool")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    if backend:
        import httpx

        @app.get("/api/health")
        async def health():
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(f"{backend}/health")
                return JSONResponse(r.json(), status_code=r.status_code)
            except httpx.HTTPError as error:
                return JSONResponse({"ok": False, "detail": f"lift service unreachable: {error!r}"}, status_code=503)

        @app.post("/api/{endpoint}")
        async def lift(endpoint: str, request: Request):
            if endpoint not in ("lift", "lift_multi"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            try:
                async with httpx.AsyncClient(timeout=1200) as client:
                    r = await client.post(f"{backend}/{endpoint}", content=await request.body(),
                                          headers={"content-type": request.headers["content-type"]})
            except httpx.HTTPError as error:
                return JSONResponse({"detail": f"lift service unreachable: {error!r}"}, status_code=503)
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    else:
        from pctool.service import app as service

        app.mount("/api", service)

    @app.put("/api/export/{name}")
    async def export_file(name: str, path: str, request: Request):
        """Write one exported file (cameras.json, inputs/NN.png, guide/NNNN.png) under <exports>/<name>/."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name) or name.startswith("."):
            return JSONResponse({"detail": "export name: letters, digits, . _ - only"}, status_code=400)
        parts = Path(path).parts
        if not parts or ".." in parts or path.startswith("/") or not re.fullmatch(r"[A-Za-z0-9._/-]+", path):
            return JSONResponse({"detail": f"bad path {path!r}"}, status_code=400)
        target = exports / name / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await request.body())
        return {"dir": str(exports / name), "path": str(target)}

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=23200)
    parser.add_argument("--backend", help="URL of a pctool-service; default: lift in this process")
    parser.add_argument("--exports", type=Path, default=Path("exports"), help="directory for exported paths")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.backend.rstrip("/") if args.backend else None, args.exports.resolve()),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
