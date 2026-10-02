FROM python:3.11-slim-bookworm

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

# Enroot uses these image settings to inject the node's NVIDIA driver.
# Slurm must restrict GPU access to the job's allocation.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
# Keep the large CUDA wheel in its own layer. Changes to requirements.txt or
# application code can then reuse this layer from the Docker build cache.
RUN python -m pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.10.0
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY train_search.py summarize.py prepare_data.py dataset.py config.py ./

ENTRYPOINT ["python", "/app/train_search.py"]
CMD ["--dataset-path", "/dataset", "--search-config", "/search/search.json", "--device", "auto", "--output-dir", "/results"]
