# Bring Your Own Container (BYOC) image for the workshop.
#
# This image is built fully from scratch on top of the official slim Python
# image, rather than extending the AWS-managed XGBoost container. It installs
# only the dependencies this workshop actually uses, plus the SageMaker
# training and inference toolkits that make the container script-mode
# compatible (so notebooks can pass a `train.py` / `inference.py` at job time).
#
# Runtime contract that SageMaker expects and this image honors:
#   - Training: `docker run <image> train` -> sagemaker-training runs the
#     user script referenced by $SAGEMAKER_PROGRAM. Input channels land in
#     /opt/ml/input/data/<channel>/, hyperparameters in
#     /opt/ml/input/config/hyperparameters.json, model artifacts written to
#     /opt/ml/model/.
#   - Inference: `docker run <image> serve` -> our `serve` script starts
#     Multi Model Server (via sagemaker-inference) on :8080 exposing
#     GET /ping and POST /invocations, delegating to model_fn / input_fn /
#     predict_fn / output_fn from inference.py uploaded by the SDK.
#   - Processing: the SDK runs your script directly; no special entrypoint.
#
# Base image is pulled from the AWS ECR Public Gallery mirror of the official
# Python image. Using the public.ecr.aws mirror avoids Docker Hub anonymous
# rate limits and works reliably inside SageMaker Studio's Docker proxy.
FROM public.ecr.aws/docker/library/python:3.10-slim-bookworm

# OS-level dependencies:
#   build-essential, libgomp1: XGBoost + scikit-learn runtime (OpenMP)
#   curl, ca-certificates:     S3/TLS, general tooling
#   sudo:                      required by SageMaker SDK @remote / @step
#                              bootstrap scripts to install helpers at runtime
#   default-jre-headless:      Multi Model Server (used by sagemaker-inference
#                              to serve predictions) runs on the JVM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
         libgomp1 \
         curl \
         ca-certificates \
         sudo \
         default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies pinned in requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# SageMaker toolkits:
#   sagemaker-training   provides the `train` entrypoint and SAGEMAKER_PROGRAM
#                        handling so training jobs can pass a user script.
#   sagemaker-inference  provides helpers used by our `serve` script below.
#   multi-model-server   the JVM-based model server that sagemaker-inference
#                        wraps. Required for `serve` to actually start an
#                        HTTP server on :8080.
RUN pip install --no-cache-dir \
      sagemaker-training \
      sagemaker-inference \
      multi-model-server

# Install our custom `serve` script. SageMaker invokes `docker run <image> serve`
# to start inference, so the script must be on $PATH and executable.
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# SageMaker convention: user code may be copied into /opt/program.
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program
