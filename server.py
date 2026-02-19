import uuid
import time
import logging
import threading
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from handler import handler, handler_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Generation API")

# In-memory job store
jobs: dict = {}
jobs_lock = threading.Lock()


class GenerateRequest(BaseModel):
    # Image source (one required)
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    # Optional end frame
    end_image_path: Optional[str] = None
    end_image_url: Optional[str] = None
    end_image_base64: Optional[str] = None
    # Generation params
    prompt: str
    negative_prompt: Optional[str] = None
    seed: int
    cfg: float = 2.0
    width: int = 480
    height: int = 832
    length: int = Field(default=81, description="Number of frames")
    steps: int = 10
    context_overlap: int = 24
    context_frames: int = 81
    # Model overrides (filename on volume, or _url to download on-demand)
    diffusion_model_high: Optional[str] = None
    diffusion_model_high_url: Optional[str] = None
    diffusion_model_low: Optional[str] = None
    diffusion_model_low_url: Optional[str] = None
    vae_model: Optional[str] = None
    vae_model_url: Optional[str] = None
    text_encoder: Optional[str] = None
    text_encoder_url: Optional[str] = None
    clip_vision: Optional[str] = None
    clip_vision_url: Optional[str] = None
    # LoRA pairs
    lora_pairs: Optional[list] = None


class ModelDownload(BaseModel):
    url: str
    model_type: str
    filename: Optional[str] = None


class GenerateV2Request(BaseModel):
    workflow: dict
    image_base64: str
    image_node_id: str = "244"
    end_image_base64: Optional[str] = None
    end_image_node_id: str = "617"
    model_downloads: Optional[List[ModelDownload]] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Optional[float] = None
    progress_step: Optional[int] = None
    progress_max: Optional[int] = None
    current_node: Optional[str] = None


def run_job(job_id: str, job_input: dict, handler_fn=handler):
    """Run video generation in a background thread."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

    def on_progress(step, max_steps, node_id):
        with jobs_lock:
            jobs[job_id]["progress"] = round(step / max_steps * 100, 1)
            jobs[job_id]["progress_step"] = int(step)
            jobs[job_id]["progress_max"] = int(max_steps)
            jobs[job_id]["current_node"] = node_id

    logger.info(f"Starting generation for job {job_id}")

    try:
        result = handler_fn(job_input, progress_callback=on_progress)
        with jobs_lock:
            if "error" in result:
                jobs[job_id]["status"] = "failed"
            else:
                jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
            jobs[job_id]["completed_at"] = time.time()
        logger.info(f"Job {job_id} finished with status: {jobs[job_id]['status']}")
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["result"] = {"error": str(e)}
            jobs[job_id]["completed_at"] = time.time()
        logger.error(f"Job {job_id} failed with exception: {e}")


@app.post("/generate")
def generate(request: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_input = request.model_dump(exclude_none=True)
    job_input["job_id"] = job_id

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "result": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "progress": None,
            "progress_step": None,
            "progress_max": None,
            "current_node": None,
        }

    thread = threading.Thread(target=run_job, args=(job_id, job_input), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.post("/generate/v2")
def generate_v2(request: GenerateV2Request):
    job_id = str(uuid.uuid4())
    job_input = request.model_dump(exclude_none=True)
    job_input["job_id"] = job_id

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "result": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "progress": None,
            "progress_step": None,
            "progress_max": None,
            "current_node": None,
        }

    thread = threading.Thread(target=run_job, args=(job_id, job_input, handler_v2), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job)


@app.get("/health")
def health():
    comfyui_ok = False
    server_address = __import__("os").getenv("SERVER_ADDRESS", "127.0.0.1")
    try:
        urllib.request.urlopen(f"http://{server_address}:8188/", timeout=5)
        comfyui_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "comfyui": "reachable" if comfyui_ok else "unreachable",
        "active_jobs": sum(1 for j in jobs.values() if j["status"] in ("pending", "running")),
    }
