import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())


def to_nearest_multiple_of_16(value):
    """Round the given value to the nearest multiple of 16, minimum 16."""
    try:
        numeric_value = float(value)
    except Exception as e:
        raise Exception(f"width/height value is not a number: {value} (original error: {e})")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def process_input(input_data, temp_dir, output_filename, input_type):
    """Process input data and return a file path."""
    if input_type == "path":
        logger.info(f"Processing path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"Processing URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Processing Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url, output_path):
    """Download a file from a URL."""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)

        if result.returncode == 0:
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else "unknown"
            logger.info(f"Successfully downloaded file from URL: {url} -> {output_path} ({file_size} bytes)")
            return output_path
        else:
            logger.error(f"wget download failed (returncode={result.returncode}): stderr={result.stderr}, stdout={result.stdout}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise Exception(f"Error during download: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Save Base64 data to a file."""
    try:
        decoded_data = base64.b64decode(base64_data)

        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)

        logger.info(f"Saved Base64 input to file: '{file_path}' ({len(decoded_data)} bytes)")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decoding failed: {e}")
        raise Exception(f"Base64 decoding failed: {e}")

MODEL_DIRS = {
    "loras": "/runpod-volume/loras",
    "diffusion_models": "/runpod-volume/models",
    "vae": "/runpod-volume/vae",
    "text_encoders": "/runpod-volume/text_encoders",
    "clip_vision": "/runpod-volume/clip_vision",
}

def download_model(url, model_type, filename=None):
    """Download a model file to the persistent volume if not already present.
    Returns the filename (not full path) for use in workflow nodes."""
    dest_dir = MODEL_DIRS.get(model_type)
    if not dest_dir:
        raise Exception(f"Unknown model type: {model_type}")

    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)

    if not filename.endswith(".safetensors"):
        filename += ".safetensors"

    dest = os.path.join(dest_dir, filename)

    if os.path.exists(dest):
        logger.info(f"{model_type} model already exists, skipping download: {dest}")
        return filename

    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Downloading {model_type} model: {url} -> {dest}")
    download_file_from_url(url, dest)
    return filename


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    try:
        response_data = urllib.request.urlopen(req).read()
        result = json.loads(response_data)
        logger.info(f"Prompt queued successfully, prompt_id: {result.get('prompt_id', 'UNKNOWN')}")
        return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace') if e.fp else "no response body"
        logger.error(f"Failed to queue prompt - HTTP {e.code}: {error_body}")
        raise Exception(f"Failed to queue prompt - HTTP {e.code}: {error_body}")
    except urllib.error.URLError as e:
        logger.error(f"Failed to connect to ComfyUI to queue prompt: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse queue_prompt response as JSON: {e}")
        raise


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            logger.info(f"History response contains {len(data)} entries")
            return data
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to get history for prompt_id={prompt_id} - HTTP {e.code}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"Failed to connect to ComfyUI for history: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse history response as JSON: {e}")
        raise

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_videos(ws, prompt):
    logger.info("Starting video generation pipeline...")

    try:
        queue_result = queue_prompt(prompt)
        prompt_id = queue_result['prompt_id']
        logger.info(f"Prompt queued with ID: {prompt_id}")
    except KeyError:
        logger.error(f"queue_prompt response missing 'prompt_id' key. Response: {queue_result}")
        raise Exception(f"ComfyUI queue_prompt response missing 'prompt_id'. Response: {json.dumps(queue_result)}")
    except Exception as e:
        logger.error(f"Failed to queue prompt: {e}\n{traceback.format_exc()}")
        raise

    logger.info("Waiting for ComfyUI to finish execution (listening on WebSocket)...")
    message_count = 0
    start_time = time.time()
    current_node = None

    while True:
        try:
            out = ws.recv()
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"WebSocket recv() failed after {elapsed:.1f}s and {message_count} messages: {e}")
            raise Exception(f"WebSocket connection lost during execution after {elapsed:.1f}s: {e}")

        if isinstance(out, str):
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON WebSocket message (length={len(out)}): {out[:200]}")
                continue

            msg_type = message.get('type', 'unknown')
            msg_data = message.get('data', {})
            message_count += 1

            if msg_type == 'executing':
                node = msg_data.get('node')
                msg_prompt_id = msg_data.get('prompt_id')

                if node is not None:
                    current_node = node
                    logger.info(f"Executing node: {node} (message #{message_count}, elapsed: {time.time() - start_time:.1f}s)")

                if node is None and msg_prompt_id == prompt_id:
                    elapsed = time.time() - start_time
                    logger.info(f"Execution complete. Total messages: {message_count}, elapsed: {elapsed:.1f}s")
                    break
            elif msg_type == 'execution_error':
                error_info = msg_data
                node_id = error_info.get('node_id', 'unknown')
                node_type = error_info.get('node_type', 'unknown')
                exception_message = error_info.get('exception_message', 'no message')
                exception_type = error_info.get('exception_type', 'unknown')
                logger.error(
                    f"ComfyUI execution error on node {node_id} ({node_type}): "
                    f"[{exception_type}] {exception_message}"
                )
                traceback_lines = error_info.get('traceback', [])
                if traceback_lines:
                    logger.error(f"ComfyUI traceback:\n{''.join(traceback_lines)}")
                raise Exception(
                    f"ComfyUI execution failed on node {node_id} ({node_type}): "
                    f"[{exception_type}] {exception_message}"
                )
            elif msg_type == 'progress':
                value = msg_data.get('value', '?')
                max_val = msg_data.get('max', '?')
                if message_count % 5 == 0:
                    logger.info(f"Progress: {value}/{max_val} (node: {current_node}, elapsed: {time.time() - start_time:.1f}s)")
            elif msg_type == 'execution_cached':
                cached_nodes = msg_data.get('nodes', [])
                logger.info(f"Cached nodes (skipped): {cached_nodes}")
            elif msg_type == 'status':
                queue_remaining = msg_data.get('status', {}).get('exec_info', {}).get('queue_remaining', '?')
                logger.info(f"Queue status: {queue_remaining} remaining")
            else:
                logger.debug(f"WebSocket message type={msg_type}: {json.dumps(msg_data)[:200]}")
        else:
            logger.debug(f"Received binary WebSocket data ({len(out)} bytes)")
            continue

    logger.info(f"Retrieving execution history for prompt_id: {prompt_id}")
    try:
        history_response = get_history(prompt_id)
    except Exception as e:
        logger.error(f"Failed to retrieve history for prompt_id={prompt_id}: {e}\n{traceback.format_exc()}")
        raise Exception(f"Failed to retrieve execution history: {e}")

    if prompt_id not in history_response:
        logger.error(
            f"prompt_id '{prompt_id}' not found in history response. "
            f"Available keys: {list(history_response.keys())}"
        )
        raise Exception(f"prompt_id '{prompt_id}' not found in execution history")

    history = history_response[prompt_id]
    output_nodes = history.get('outputs', {})
    logger.info(f"History contains {len(output_nodes)} output node(s): {list(output_nodes.keys())}")

    output_videos = {}
    for node_id in output_nodes:
        node_output = output_nodes[node_id]
        videos_output = []

        output_keys = list(node_output.keys())
        logger.info(f"Node {node_id} output keys: {output_keys}")

        if 'gifs' in node_output:
            gif_entries = node_output['gifs']
            logger.info(f"Node {node_id} has {len(gif_entries)} video(s) in 'gifs' output")

            for idx, video in enumerate(gif_entries):
                fullpath = video.get('fullpath', 'NO_PATH')
                filename = video.get('filename', 'unknown')
                subfolder = video.get('subfolder', '')
                vid_type = video.get('type', 'unknown')

                logger.info(
                    f"  Video {idx}: filename={filename}, subfolder={subfolder}, "
                    f"type={vid_type}, fullpath={fullpath}"
                )

                if not os.path.exists(fullpath):
                    logger.error(
                        f"Video file does not exist: {fullpath}. "
                        f"Video metadata: {json.dumps(video)}"
                    )
                    continue

                file_size = os.path.getsize(fullpath)
                logger.info(f"  Reading video file: {fullpath} ({file_size} bytes)")

                try:
                    with open(fullpath, 'rb') as f:
                        video_data = base64.b64encode(f.read()).decode('utf-8')
                    logger.info(f"  Encoded video to base64 ({len(video_data)} chars)")
                    videos_output.append(video_data)
                except Exception as e:
                    logger.error(f"  Failed to read/encode video file {fullpath}: {e}\n{traceback.format_exc()}")
        else:
            logger.info(f"Node {node_id} has no 'gifs' key (keys present: {output_keys})")

        output_videos[node_id] = videos_output

    total_videos = sum(len(v) for v in output_videos.values())
    logger.info(f"Total videos collected: {total_videos} across {len(output_videos)} node(s)")

    return output_videos


def load_workflow(workflow_path):
    logger.info(f"Loading workflow from: {workflow_path}")
    try:
        with open(workflow_path, 'r') as file:
            workflow = json.load(file)
        logger.info(f"Workflow loaded successfully ({len(workflow)} nodes)")
        return workflow
    except FileNotFoundError:
        logger.error(f"Workflow file not found: {workflow_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Workflow file is not valid JSON: {workflow_path}: {e}")
        raise


def handler(job_input):
    """Process a video generation job. Accepts a plain dict of input parameters."""
    job_id = job_input.get("job_id", f"local_{uuid.uuid4().hex[:8]}")

    logger.info(f"=== Job {job_id} started ===")
    logger.info(f"Job input keys: {list(job_input.keys())}")
    task_id = f"task_{uuid.uuid4()}"

    try:
        # Image input processing
        image_path = None
        if "image_path" in job_input:
            image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
        elif "image_url" in job_input:
            image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
        elif "image_base64" in job_input:
            image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
        else:
            image_path = "/example_image.png"
            logger.info("No image input provided, using default image: /example_image.png")

        logger.info(f"Image path resolved to: {image_path}")

        # End image input processing
        end_image_path_local = None
        if "end_image_path" in job_input:
            end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
        elif "end_image_url" in job_input:
            end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
        elif "end_image_base64" in job_input:
            end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")

        if end_image_path_local:
            logger.info(f"End image path resolved to: {end_image_path_local}")
        else:
            logger.info("No end image provided (single-frame mode)")

        # LoRA configuration
        lora_pairs = job_input.get("lora_pairs", [])
        lora_count = min(len(lora_pairs), 4)
        if len(lora_pairs) > 4:
            logger.warning(f"Received {len(lora_pairs)} LoRA pairs but only up to 4 are supported. Using the first 4.")
            lora_pairs = lora_pairs[:4]

        # Select workflow file
        workflow_file = "/new_Wan22_flf2v_api.json" if end_image_path_local else "/new_Wan22_api.json"
        logger.info(f"Using {'FLF2V' if end_image_path_local else 'single'} workflow with {lora_count} LoRA pairs")

        prompt = load_workflow(workflow_file)

        # Model overrides — pass filenames or URLs for models on the persistent volume
        # If a _url variant is provided, the model is downloaded on-demand to /runpod-volume/
        if "diffusion_model_high_url" in job_input:
            job_input["diffusion_model_high"] = download_model(job_input["diffusion_model_high_url"], "diffusion_models", job_input.get("diffusion_model_high"))
        if "diffusion_model_high" in job_input:
            prompt["122"]["inputs"]["model"] = job_input["diffusion_model_high"]
            logger.info(f"HIGH diffusion model overridden: {job_input['diffusion_model_high']}")

        if "diffusion_model_low_url" in job_input:
            job_input["diffusion_model_low"] = download_model(job_input["diffusion_model_low_url"], "diffusion_models", job_input.get("diffusion_model_low"))
        if "diffusion_model_low" in job_input:
            prompt["549"]["inputs"]["model"] = job_input["diffusion_model_low"]
            logger.info(f"LOW diffusion model overridden: {job_input['diffusion_model_low']}")

        if "vae_model_url" in job_input:
            job_input["vae_model"] = download_model(job_input["vae_model_url"], "vae", job_input.get("vae_model"))
        if "vae_model" in job_input:
            prompt["129"]["inputs"]["model_name"] = job_input["vae_model"]
            logger.info(f"VAE model overridden: {job_input['vae_model']}")

        if "text_encoder_url" in job_input:
            job_input["text_encoder"] = download_model(job_input["text_encoder_url"], "text_encoders", job_input.get("text_encoder"))
        if "text_encoder" in job_input:
            prompt["136"]["inputs"]["model_name"] = job_input["text_encoder"]
            logger.info(f"Text encoder overridden: {job_input['text_encoder']}")

        if "clip_vision_url" in job_input:
            job_input["clip_vision"] = download_model(job_input["clip_vision_url"], "clip_vision", job_input.get("clip_vision"))
        if "clip_vision" in job_input:
            prompt["173"]["inputs"]["clip_name"] = job_input["clip_vision"]
            logger.info(f"CLIP vision model overridden: {job_input['clip_vision']}")

        length = job_input.get("length", 81)
        steps = job_input.get("steps", 10)

        logger.info(f"Video parameters: length={length}, steps={steps}")

        prompt["244"]["inputs"]["image"] = image_path
        prompt["541"]["inputs"]["num_frames"] = length
        prompt["135"]["inputs"]["positive_prompt"] = job_input["prompt"]
        prompt["135"]["inputs"]["negative_prompt"] = job_input.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
        prompt["220"]["inputs"]["seed"] = job_input["seed"]
        prompt["540"]["inputs"]["seed"] = job_input["seed"]
        prompt["540"]["inputs"]["cfg"] = job_input["cfg"]

        # Adjust resolution to nearest multiple of 16
        original_width = job_input["width"]
        original_height = job_input["height"]
        adjusted_width = to_nearest_multiple_of_16(original_width)
        adjusted_height = to_nearest_multiple_of_16(original_height)
        if adjusted_width != original_width:
            logger.info(f"Width adjusted to nearest multiple of 16: {original_width} -> {adjusted_width}")
        if adjusted_height != original_height:
            logger.info(f"Height adjusted to nearest multiple of 16: {original_height} -> {adjusted_height}")
        prompt["235"]["inputs"]["value"] = adjusted_width
        prompt["236"]["inputs"]["value"] = adjusted_height
        prompt["498"]["inputs"]["context_overlap"] = job_input.get("context_overlap", 48)
        prompt["498"]["inputs"]["context_frames"] = length

        # Apply step settings
        if "834" in prompt:
            prompt["834"]["inputs"]["steps"] = steps
            logger.info(f"Steps set to: {steps}")
            lowsteps = int(steps * 0.6)
            prompt["829"]["inputs"]["step"] = lowsteps
            logger.info(f"LowSteps set to: {lowsteps}")

        # Apply end image to node 617 if provided (FLF2V only)
        if end_image_path_local:
            prompt["617"]["inputs"]["image"] = end_image_path_local

        # Apply LoRA settings
        if lora_count > 0:
            high_lora_node_id = "279"
            low_lora_node_id = "553"

            for i, lora_pair in enumerate(lora_pairs):
                if i < 4:
                    lora_high = lora_pair.get("high")
                    lora_low = lora_pair.get("low")
                    lora_high_weight = lora_pair.get("high_weight", 1.0)
                    lora_low_weight = lora_pair.get("low_weight", 1.0)

                    # Download LoRAs from URL if provided (persists to /runpod-volume/loras/)
                    if "high_url" in lora_pair:
                        lora_high = download_model(lora_pair["high_url"], "loras", lora_high)
                    if "low_url" in lora_pair:
                        lora_low = download_model(lora_pair["low_url"], "loras", lora_low)

                    if lora_high:
                        prompt[high_lora_node_id]["inputs"][f"lora_{i}"] = lora_high
                        prompt[high_lora_node_id]["inputs"][f"strength_{i}"] = lora_high_weight
                        logger.info(f"LoRA {i} HIGH applied to node 279: {lora_high} with weight {lora_high_weight}")

                    if lora_low:
                        prompt[low_lora_node_id]["inputs"][f"lora_{i}"] = lora_low
                        prompt[low_lora_node_id]["inputs"][f"strength_{i}"] = lora_low_weight
                        logger.info(f"LoRA {i} LOW applied to node 553: {lora_low} with weight {lora_low_weight}")

        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")

        # HTTP connection check (max 3 minutes)
        http_url = f"http://{server_address}:8188/"
        logger.info(f"Checking HTTP connection to: {http_url}")
        max_http_attempts = 180
        for http_attempt in range(max_http_attempts):
            try:
                urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"HTTP connection successful (attempt {http_attempt+1})")
                break
            except Exception as e:
                logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
                if http_attempt == max_http_attempts - 1:
                    raise Exception("Cannot connect to ComfyUI server. Please verify the server is running.")
                time.sleep(1)

        ws = websocket.WebSocket()
        max_attempts = int(180 / 5)
        for attempt in range(max_attempts):
            try:
                ws.connect(ws_url)
                logger.info(f"WebSocket connection successful (attempt {attempt+1})")
                break
            except Exception as e:
                logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise Exception("WebSocket connection timed out (3 minutes)")
                time.sleep(5)

        videos = get_videos(ws, prompt)
        ws.close()
        logger.info("WebSocket closed")

        for node_id in videos:
            if videos[node_id]:
                video_count = len(videos[node_id])
                video_size = len(videos[node_id][0])
                logger.info(f"=== Job {job_id} completed successfully === Returning video from node {node_id} ({video_size} base64 chars, {video_count} total videos)")
                return {"video": videos[node_id][0]}

        node_summary = {nid: len(vids) for nid, vids in videos.items()}
        logger.error(
            f"No videos found in any output node. "
            f"Node summary (node_id: video_count): {node_summary}."
        )
        return {
            "error": "No video found in ComfyUI output. Execution completed but no nodes produced video data.",
            "details": {
                "prompt_id": task_id,
                "nodes_checked": list(videos.keys()),
                "node_video_counts": node_summary
            }
        }

    except Exception as e:
        logger.error(f"=== Job {job_id} failed === {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return {"error": f"Job failed: {type(e).__name__}: {str(e)}"}
