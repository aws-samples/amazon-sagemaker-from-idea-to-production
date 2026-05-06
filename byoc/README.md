# BYOC: XGBoost container from scratch

This folder contains a minimal **Bring Your Own Container** image for the
workshop. Unlike the default path in `00-start-here.ipynb` — which *extends*
the AWS-managed XGBoost container — this image is built **from scratch** on
top of `python:3.12-slim-bookworm` and installs only what the workshop uses.

The resulting image is a drop-in replacement for the `xgboost_image` variable
stored by `00-start-here.ipynb`, so notebooks 02–06 work unchanged.

## What's inside

| Layer | What | Why |
|---|---|---|
| `python:3.12-slim-bookworm` | Debian slim + CPython 3.12 | Minimal, matches workshop kernel version |
| `build-essential`, `libgomp1` | OpenMP runtime | Required by XGBoost and scikit-learn wheels |
| `sudo` | sudo binary | Required by SageMaker SDK `@remote` / `@step` bootstrap |
| `xgboost`, `scikit-learn`, `pandas`, `numpy` | ML stack | Training and preprocessing |
| `boto3`, `s3fs` | AWS + S3 | Reading inputs from S3 |
| `sagemaker`, `sagemaker-core`, `sagemaker-train`, `sagemaker-mlops` | SageMaker SDK v3 | Same pins as the rest of the workshop |
| `mlflow`, `sagemaker-mlflow` | Experiment tracking | Notebooks 01+ log to MLflow from inside jobs |
| `sagemaker-training` | Training toolkit | Makes the container script-mode compatible |
| `sagemaker-inference` | Inference toolkit | Provides `/ping` + `/invocations` server |

## How SageMaker uses it

The image implements the SageMaker container contract without any custom
entrypoint code:

- **Training** — SageMaker runs `docker run <image> train`. The
  `sagemaker-training` toolkit installs a `train` executable on `PATH` that
  reads `$SAGEMAKER_PROGRAM` (set by the SDK) and executes the user script
  uploaded to `/opt/ml/code/`. Standard I/O paths apply:
  `/opt/ml/input/data/<channel>/`, `/opt/ml/input/config/hyperparameters.json`,
  `/opt/ml/model/`, `/opt/ml/output/`.
- **Processing** — The SDK invokes your processing script directly; no
  special entrypoint is used.
- **Inference** — SageMaker runs `docker run <image> serve`. The
  `sagemaker-inference` toolkit starts a model server on port `8080` that
  exposes `GET /ping` and `POST /invocations`, delegating to `model_fn`,
  `input_fn`, `predict_fn`, and `output_fn` defined in the `inference.py` the
  SDK uploads at deploy time.

You keep script-mode flexibility: the image has no training or inference
logic baked in.

## Build and push

From a SageMaker Studio terminal (Docker must already be installed — the
`00-start-here.ipynb` notebook handles that):

```bash
cd ~/amazon-sagemaker-from-idea-to-production/byoc
./build_and_push.sh
```

Optional arguments: `./build_and_push.sh <region> <repo_name> <tag>`.

The script:
1. Resolves your AWS account ID.
2. Creates the ECR repository if it doesn't exist.
3. Logs Docker in to your account's ECR.
4. Builds with `--network sagemaker` (required in Studio).
5. Pushes to `<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`.

## Use this image in the workshop

In a notebook cell, replace the value of `xgboost_image` stored by
`00-start-here.ipynb`:

```python
account_id = session.account_id()
region = session.boto_region_name
xgboost_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost-byoc:1.0"
%store xgboost_image
```

Notebooks 02–06 use `%store -r xgboost_image`, so they'll pick up the new
URI automatically. To go back to the extended image, rerun the relevant
cells in `00-start-here.ipynb`.

## Trade-offs vs. the extended image

| | Extended (default) | BYOC (this folder) |
|---|---|---|
| Base | `sagemaker-xgboost:3.0-5` (AWS-managed) | `python:3.12-slim-bookworm` |
| Size | Larger (inherits full AWS image) | Smaller (only what's needed) |
| Cross-account ECR pull | Yes (`246618743249.dkr.ecr...`) | No |
| Transparency | Limited (base is opaque) | Full (you see every layer) |
| Maintenance | AWS maintains base | You maintain every layer |
| Patch cadence | Follows AWS releases | Follows your rebuild cadence |

Both paths produce a SageMaker-compatible image; pick based on whether you
value ownership (BYOC) or managed maintenance (extended).

## Verifying the build

After `docker push` succeeds, a quick sanity check:

```bash
# Confirm the image can start and the toolkits are installed
docker run --rm <image-uri> python -c "import sagemaker_training, sagemaker_inference, xgboost, sklearn, mlflow; print('ok')"
```

End-to-end verification is running notebook `02-sagemaker-containers.ipynb`
against the new image (both local mode and a managed training job).
