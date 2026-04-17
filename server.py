import re
import secrets
import shutil
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from waldo_pipeline import run_waldo_pipeline


ROOT = Path(__file__).resolve().parent
GAME_DIR = ROOT / "waldo-game"
GENERATED_ROOT = ROOT / "generated_rounds"
UPLOAD_ROOT = ROOT / "uploaded_images"
DEFAULT_WALDO_PATH = ROOT / "waldo.png"
DEFAULT_PIPELINE_ORDER = "stylize_bg_place_raw_waldo"

GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Where's Waldo Generator")


def sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-")
    return cleaned or "scene"


def build_level_payload(manifest: dict) -> dict:
    placement = manifest["placement"]
    final_image_path = Path(manifest["artifacts"]["final_image"]).resolve()
    if not final_image_path.exists():
        raise HTTPException(status_code=500, detail="Generated image not found.")

    image_width, image_height = placement["image_size_px"]
    center_x, center_y = placement["center"]
    waldo_width, waldo_height = placement["waldo_size_px"]
    hit_radius = max(waldo_width, waldo_height) * 0.40 / max(1, image_width)

    image_rel = final_image_path.relative_to(GENERATED_ROOT.resolve()).as_posix()
    return {
        "id": manifest["round_id"],
        "type": "image",
        "title": manifest["title"],
        "src": f"/generated/{image_rel}",
        "waldo": {
            "x": center_x / image_width,
            "y": center_y / image_height,
        },
        "hitRadius": hit_radius,
    }


@app.get("/api/health")
def healthcheck():
    return {"ok": True}


@app.post("/api/generate")
async def generate_round(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image filename provided.")

    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use PNG, JPG, JPEG, or WEBP.")

    round_id = secrets.token_hex(8)
    title = f"Generated: {sanitize_stem(image.filename)}"
    upload_path = UPLOAD_ROOT / f"{round_id}{suffix}"
    output_dir = GENERATED_ROOT / round_id

    with upload_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    await image.close()

    try:
        manifest = await run_in_threadpool(
            run_waldo_pipeline,
            upload_path,
            waldo_path=DEFAULT_WALDO_PATH,
            output_dir=output_dir,
            pipeline_order=DEFAULT_PIPELINE_ORDER,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    manifest["round_id"] = round_id
    manifest["title"] = title
    level = build_level_payload(manifest)
    return JSONResponse({"level": level, "manifest": manifest})


app.mount("/generated", StaticFiles(directory=GENERATED_ROOT), name="generated")
app.mount("/", StaticFiles(directory=GAME_DIR, html=True), name="game")
