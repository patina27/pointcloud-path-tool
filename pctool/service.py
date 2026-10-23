"""GPU lift service: run this where MoGe and a GPU are, and point `pctool-server --backend` at it.

    pctool-service --host 127.0.0.1 --port 23190
GET /health, POST /lift (multipart: image; form: model, fov_x, max_side) -> see pctool.lift.
"""
import argparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from pctool import lift as lifting

app = FastAPI(title="pctool lift service")


@app.get("/health")
def health():
    return lifting.health()


@app.post("/lift")
def lift(image: UploadFile = File(...), model: str = Form(lifting.DEFAULT_MODEL),
         fov_x: float | None = Form(None), max_side: int = Form(1024)):
    import torch

    try:
        body = lifting.lift(image.file.read(), model, fov_x, max_side)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        return JSONResponse({"detail": f"CUDA out of memory: {error}"}, status_code=503)
    return Response(body, media_type="application/octet-stream")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23190)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
