# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A video generation service wrapping ComfyUI with the Wan2.2 I2V (image-to-video) model, exposed via a FastAPI HTTP API. Designed to run as a RunPod worker with NVIDIA GPU. Supports single-frame (I2V) and multi-frame (FLF2V) video generation with optional LoRA pairs.

## Build & Run

```bash
# Build and run with Docker Compose (requires NVIDIA GPU)
docker compose up --build

# Or build manually
docker build -t video-generator .
docker run --gpus all -p 8080:8080 -v ./models:/runpod-volume video-generator
```

The container startup sequence (entrypoint.sh):
1. Downloads default models to `/runpod-volume/` if not already present (with corruption detection via min file size checks)
2. Starts ComfyUI on port 8188 (internal, with `--use-sage-attention`)
3. Waits up to 120s for ComfyUI health check
4. Starts FastAPI/Uvicorn on port 8080

Mount `./models:/runpod-volume` to persist model downloads across container restarts.

## Architecture

**Two source files, no tests, no linter configured.**

- **server.py** — FastAPI app with three endpoints:
  - `POST /generate` — accepts image + generation params, spawns a background thread, returns `job_id`
  - `GET /status/{job_id}` — polls job status and retrieves base64-encoded video result
  - `GET /health` — checks ComfyUI reachability
  - Jobs are stored in an in-memory dict with a threading lock. The `GenerateRequest` Pydantic model defines all accepted parameters.

- **handler.py** — Core generation logic called by `server.py`:
  1. Resolves image input (file path, URL via wget, or base64 decode)
  2. Downloads any on-demand models (via `_url` fields) to `/runpod-volume/`
  3. Loads the appropriate ComfyUI workflow JSON and injects parameters into specific node IDs
  4. Queues the workflow to ComfyUI via HTTP (`POST /prompt`)
  5. Monitors execution via WebSocket (`ws://127.0.0.1:8188/ws`)
  6. Retrieves output video from ComfyUI history, returns as base64

**Workflow selection:** If an end image is provided, uses `new_Wan22_flf2v_api.json` (FLF2V mode); otherwise uses `new_Wan22_api.json` (single-frame I2V).

## Model Storage

Models are stored on a persistent volume at `/runpod-volume/` with this structure:
- `/runpod-volume/models/` — diffusion models (HIGH/LOW fp8)
- `/runpod-volume/loras/` — LoRA weights
- `/runpod-volume/clip_vision/` — CLIP vision encoder
- `/runpod-volume/text_encoders/` — text encoder (umt5-xxl)
- `/runpod-volume/vae/` — VAE decoder

Default models are downloaded at first startup by `entrypoint.sh`. Additional models can be downloaded on-demand at runtime by passing `_url` fields in the API request. The `extra_model_paths.yaml` configures ComfyUI to search both `/ComfyUI/models/` and `/runpod-volume/` paths.

## Key ComfyUI Node IDs (hardcoded in handler.py)

These numeric node IDs map to specific workflow nodes and must stay in sync with the workflow JSON files:

| Node ID | Purpose |
|---------|---------|
| 122 | HIGH diffusion model loader |
| 549 | LOW diffusion model loader |
| 129 | VAE model loader |
| 136 | Text encoder loader |
| 173 | CLIP vision model loader |
| 244 | Input image path |
| 617 | End image path (FLF2V only) |
| 541 | Frame count (`num_frames`) |
| 135 | Positive/negative prompt |
| 220, 540 | Seed and CFG |
| 235, 236 | Width and height |
| 498 | Context overlap and frames |
| 834 | Steps |
| 829 | Low steps (60% of steps) |
| 279 | High LoRA node |
| 553 | Low LoRA node |

## Dependencies

Python packages (no requirements.txt — installed in Dockerfile):
- `fastapi`, `uvicorn[standard]`, `websocket-client`
- ComfyUI and its custom nodes are cloned/installed in the Docker image (shallow clones, .git dirs stripped)

## Environment Variables

- `SERVER_ADDRESS` — ComfyUI host (default: `127.0.0.1`). Both `handler.py` and `server.py` use this.

## Conventions

- Resolution values are auto-adjusted to the nearest multiple of 16 (ComfyUI requirement)
- Max 4 LoRA pairs per request; each has high/low model paths and weights (plus optional `high_url`/`low_url` for on-demand download)
- Videos are returned as base64-encoded strings in the JSON response
- Model overrides accept either a filename (already on the volume) or a `_url` variant that triggers download to `/runpod-volume/`
