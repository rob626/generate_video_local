#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Download default models to persistent volume if not already present ---
# Models on /runpod-volume/ survive container restarts.
# After first cold start, this section completes instantly.
echo "Checking for required models..."

download_if_missing() {
    local url="$1"
    local dest="$2"
    local min_bytes="${3:-1000000}"  # minimum expected size (default 1MB)
    if [ -f "$dest" ]; then
        local size
        size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
        if [ "$size" -lt "$min_bytes" ]; then
            echo "  Corrupted (${size} bytes < ${min_bytes} min), re-downloading: $dest"
            rm -f "$dest"
        else
            echo "  Already exists: $dest ($(du -h "$dest" | cut -f1))"
            return 0
        fi
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  Downloading: $dest"
    local tmp="${dest}.tmp"
    rm -f "$tmp"
    wget --tries=3 --retry-connrefused --waitretry=5 -q --show-progress "$url" -O "$tmp"
    mv "$tmp" "$dest"
    echo "  Done: $dest ($(du -h "$dest" | cut -f1))"
}

# Diffusion models (~14GB each)
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
    "/runpod-volume/models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
    10000000000

download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
    "/runpod-volume/models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
    10000000000

# LoRAs (~200MB each)
download_if_missing \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
    "/runpod-volume/loras/high_noise_model.safetensors" \
    100000000

download_if_missing \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" \
    "/runpod-volume/loras/low_noise_model.safetensors" \
    100000000

# CLIP vision (~1.7GB)
download_if_missing \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "/runpod-volume/clip_vision/clip_vision_h.safetensors" \
    1000000000

# Text encoder (~9.5GB)
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" \
    "/runpod-volume/text_encoders/umt5-xxl-enc-bf16.safetensors" \
    5000000000

# VAE (~300MB)
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
    "/runpod-volume/vae/Wan2_1_VAE_bf16.safetensors" \
    100000000

echo "All models ready."

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the FastAPI server as the main process
echo "Starting the API server..."
exec uvicorn server:app --host 0.0.0.0 --port 8080
