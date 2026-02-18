FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

WORKDIR /

# Install Python deps in a single layer, no cache
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websocket-client

# Clone ComfyUI + all custom nodes, install deps, strip .git dirs — single layer
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt && \
    cd /ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone --depth 1 https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone --depth 1 https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && mv * ../ && \
    find /ComfyUI -name ".git" -type d -exec rm -rf {} + 2>/dev/null; true

# Models are stored on the persistent volume (/runpod-volume/) and downloaded
# on-demand by the handler. Default models are downloaded at first startup
# only if not already present on the volume.

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh

EXPOSE 8080

CMD ["/entrypoint.sh"]
