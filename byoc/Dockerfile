# Bring Your Own Container (BYOC) image for the workshop.
#
# This image is built fully from scratch on top of the official slim Python
# image, rather than extending the AWS-managed XGBoost container. It installs
# only the dependencies this workshop actually uses, plus the SageMaker
# training toolkit for script-mode training. Inference is served by our own
# lightweight FastAPI server (see `serve`) — the archived sagemaker-inference
# toolkit and its JVM-based Multi Model Server are NOT used.
#
# Runtime contract that SageMaker expects and this image honors:
#   - Training: `docker run <image> train` -> sagemaker-training runs the
#     user script referenced by $SAGEMAKER_PROGRAM. Input channels land in
#     /opt/ml/input/data/<channel>/, hyperparameters in
#     /opt/ml/input/config/hyperparameters.json, model artifacts written to
#     /opt/ml/model/.
#   - Inference: `docker run <image> serve` -> our `serve` script starts a
#     FastAPI/uvicorn server on :8080 exposing GET /ping and POST
#     /invocations, delegating to model_fn / input_fn / predict_fn /
#     output_fn from the inference.py uploaded by the SDK (resolved via
#     $SAGEMAKER_PROGRAM and $SAGEMAKER_SUBMIT_DIRECTORY).
#   - Processing: the SDK runs your script directly; no special entrypoint.
#
# Base image is pulled from the AWS ECR Public Gallery mirror of the official
# Python image. Using the public.ecr.aws mirror avoids Docker Hub anonymous
# rate limits and works reliably inside SageMaker Studio's Docker proxy.
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

# OS-level dependencies:
#   build-essential, libgomp1: XGBoost + scikit-learn runtime (OpenMP)
#   curl, ca-certificates:     S3/TLS, general tooling
#   sudo:                      required by SageMaker SDK @remote / @step
#                              bootstrap scripts to install helpers at runtime
# (No JVM: the old Multi Model Server needed default-jre-headless; the native
# FastAPI serving stack does not.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
         libgomp1 \
         curl \
         ca-certificates \
         sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies pinned in requirements.txt.
# requirements.txt adds the PyTorch CPU index via --extra-index-url. uv defaults
# to the `first-index` strategy (only considers the first index that publishes a
# given package, to prevent dependency-confusion attacks). That makes transitive
# deps such as importlib-metadata resolve to the only version the PyTorch index
# hosts (7.1.0), which conflicts with sagemaker-core (<7.0). `unsafe-best-match`
# restores pip's behavior: consider all (trusted) indexes and pick the best
# compatible version.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache --index-strategy unsafe-best-match -r /tmp/requirements.txt

# SageMaker toolkit + serving stack:
#   sagemaker-training   provides the `train` entrypoint and SAGEMAKER_PROGRAM
#                        handling so training jobs can pass a user script.
#   fastapi + uvicorn    power our own `serve` script (below), which implements
#                        the SageMaker inference contract (/ping, /invocations)
#                        natively. This replaces the archived sagemaker-inference
#                        toolkit + multi-model-server (and their JVM dependency).
RUN uv pip install --system --no-cache \
      sagemaker-training \
      "fastapi>=0.115,<1" \
      "uvicorn>=0.30,<1"

# Install our custom serving stack. SageMaker invokes `docker run <image> serve`
# to start inference, so the launcher script must be on $PATH and executable.
# The FastAPI app lives in serving_app.py, installed to /opt/serving (added to
# PYTHONPATH): uvicorn needs the "serving_app:build_app" import string to spawn
# multiple worker processes — one per vCPU — so every processor on the hosting
# instance serves requests.
COPY serve /usr/local/bin/serve
COPY serving_app.py /opt/serving/serving_app.py
RUN chmod +x /usr/local/bin/serve
ENV PYTHONPATH="/opt/serving:${PYTHONPATH}"

# Sanity check: every import the container needs at runtime must resolve,
# including the serving app module the uvicorn workers import.
RUN python -c "import fastapi, uvicorn, sagemaker_training, xgboost, sklearn, mlflow, pandas, numpy, serving_app; print('ok')"

# SageMaker convention: user code may be copied into /opt/program.
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program
