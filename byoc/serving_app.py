"""
SageMaker inference application — implemented natively, without the (archived)
sagemaker-inference toolkit or the JVM-based Multi Model Server.

This module holds the FastAPI application and the SageMaker handler dispatch
logic. It is kept as an importable module (installed to /opt/serving, which is
on PYTHONPATH) so the ``serve`` launcher can pass the ``serving_app:build_app``
import string to uvicorn and scale to multiple worker processes — one per
CPU — the same way Multi Model Server defaulted to one worker per vCPU.
uvicorn can only fork/spawn workers from an import string, not from an app
object, which is why this code lives here rather than in the ``serve`` script.

SageMaker starts the inference container with ``docker run <image> serve`` and
expects an HTTP server on port 8080 (or $SAGEMAKER_BIND_TO_PORT) handling:

    GET  /ping         -> 200 when healthy
    POST /invocations  -> model predictions

Requests are dispatched to the standard SageMaker handler functions defined in
the user's ``inference.py`` (uploaded by the SDK at deploy time and located via
$SAGEMAKER_PROGRAM and $SAGEMAKER_SUBMIT_DIRECTORY):

    model_fn(model_dir)                 -> model object          (required)
    input_fn(body, content_type)        -> deserialized payload  (optional)
    predict_fn(data, model)             -> prediction            (optional)
    output_fn(prediction, accept)       -> serialized response   (optional)

Defaults are provided for the optional functions, mirroring the behavior the
sagemaker-inference toolkit used to supply.

Note on workers: ``build_app()`` runs once per worker process, so each worker
loads its own copy of the model (again matching MMS semantics). One-time,
non-parallel-safe setup (installing code/requirements.txt) belongs in
``prepare()``, which the launcher calls exactly once before spawning workers.
"""
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Request, Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("serve")

# SageMaker container contract paths. MODEL_DIR is overridable only to allow
# local smoke tests outside a container; SageMaker itself always uses /opt/ml/model.
MODEL_DIR = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
PROGRAM = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")
PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))


def _resolve_code_dir() -> str:
    """Locate the directory holding the user's inference script.

    The SDK repacks the entry point into model.tar.gz under code/, which
    SageMaker extracts to /opt/ml/model/code. SAGEMAKER_SUBMIT_DIRECTORY may
    point there directly, or may hold an s3:// URI in some flows — in that
    case the local copy under the model dir is authoritative.
    """
    submit_dir = os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", "")
    if submit_dir and not submit_dir.startswith("s3://"):
        return submit_dir
    return str(Path(MODEL_DIR) / "code")


def prepare() -> None:
    """One-time setup run by the launcher BEFORE workers are spawned.

    Installs code/requirements.txt if present (toolkit-compatible behavior).
    Must not run per-worker: concurrent pip installs into the same
    site-packages are racy.
    """
    code_dir = _resolve_code_dir()
    req = Path(code_dir) / "requirements.txt"
    if req.exists():
        logger.info("Installing user requirements from %s", req)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def _load_user_module(code_dir: str):
    """Import the user's inference script named by $SAGEMAKER_PROGRAM."""
    script = Path(code_dir) / PROGRAM
    if not script.exists():
        raise FileNotFoundError(
            f"Inference script {script} not found. The SageMaker SDK uploads it "
            f"at deploy time (entry_point) and sets SAGEMAKER_PROGRAM."
        )
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("user_inference_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded user inference module from %s", script)
    return module


# --- Default handlers (used when inference.py omits a function) --------------

def default_model_fn(model_dir):
    raise NotImplementedError(
        "inference.py must define model_fn(model_dir); this container provides "
        "no framework-specific default model loader."
    )


def default_input_fn(body: bytes, content_type: str):
    if content_type.startswith("application/json"):
        return json.loads(body)
    if content_type.startswith("text/csv"):
        import io

        import numpy as np

        return np.genfromtxt(io.BytesIO(body), delimiter=",")
    raise ValueError(f"Unsupported content type: {content_type}")


def default_predict_fn(data, model):
    return model.predict(data)


def default_output_fn(prediction, accept: str):
    import numpy as np

    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    if accept.startswith("text/csv"):
        if not isinstance(prediction, (list, tuple)):
            prediction = [prediction]
        return ",".join(str(p) for p in prediction), "text/csv"
    # application/json and */* both serialize to JSON
    return json.dumps(prediction), "application/json"


# --- Application factory (runs once per worker process) -----------------------

def build_app() -> FastAPI:
    code_dir = _resolve_code_dir()
    user_module = _load_user_module(code_dir)

    model_fn = getattr(user_module, "model_fn", default_model_fn)
    input_fn = getattr(user_module, "input_fn", default_input_fn)
    predict_fn = getattr(user_module, "predict_fn", default_predict_fn)
    output_fn = getattr(user_module, "output_fn", default_output_fn)

    # Load the model once per worker at startup. If this raises, the worker
    # exits, uvicorn's manager shuts down, and SageMaker marks the endpoint
    # unhealthy — same behavior as the toolkit.
    model = model_fn(MODEL_DIR)
    logger.info("Model loaded from %s (pid %d)", MODEL_DIR, os.getpid())

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/ping")
    def ping():
        # The model loaded successfully at startup; serving == healthy.
        return Response(status_code=200)

    @app.post("/invocations")
    async def invocations(request: Request):
        content_type = request.headers.get("content-type", "application/json")
        accept = request.headers.get("accept", "application/json")
        if accept in ("", "*/*"):
            accept = "application/json"
        body = await request.body()
        try:
            data = input_fn(body, content_type)
            prediction = predict_fn(data, model)
            result = output_fn(prediction, accept)
        except ValueError as e:
            # Unsupported content/accept type -> 415, matching toolkit behavior
            return Response(content=str(e), status_code=415, media_type="text/plain")
        # output_fn may return (payload, content_type) or just the payload
        if isinstance(result, tuple):
            payload, media_type = result
        else:
            payload, media_type = result, accept
        return Response(content=payload, media_type=media_type)

    return app
