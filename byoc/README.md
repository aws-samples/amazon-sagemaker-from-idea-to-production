# BYOC: XGBoost container from scratch

Notebook 00 builds this custom image from `python:3.12-slim-bookworm`, pushes it to ECR, and stores its URI as `xgboost_image`. The image contains the dependencies and entrypoints used by the workshop's remote functions, pipeline steps, processing job, and real-time endpoint.

## What's inside

| Layer | Purpose |
|---|---|
| `python:3.12-slim-bookworm` | Matches the workshop kernel version |
| `build-essential`, `libgomp1` | XGBoost and scikit-learn runtime support |
| `sudo` | Required by the SageMaker SDK `@remote` and `@step` bootstrap |
| XGBoost, scikit-learn, pandas, NumPy | Preprocessing and training |
| boto3, s3fs | Reading and writing S3 data |
| SageMaker SDK v3 packages | Remote functions, pipelines, processing, and resources |
| MLflow and `sagemaker-mlflow` | Experiment tracking and registry sync |
| `sagemaker-training` | The SageMaker `train` entrypoint and script-mode compatibility |
| FastAPI and uvicorn | Native `/ping` and `/invocations` inference server |

SHAP and matplotlib are installed only for explainability jobs from `explainability/requirements.txt`; they are not baked into the base image.

## How the workshop uses the image

- **Remote functions in notebook 01** — `@remote` runs the reusable preprocess, train, and explain functions on SageMaker TrainingJob infrastructure. All three use this image; explainability adds its two job-specific dependencies at startup.
- **Pipeline steps in notebook 03** — `@step` wraps the same functions and uses the same image through `config.yaml`.
- **Monitoring in notebook 04** — `FrameworkProcessor` uses the image for a native SageMaker ProcessingJob and scheduled `ProcessingStep`.
- **Inference in notebooks 04 and 05** — SageMaker starts the image with `serve`; the FastAPI application loads the `inference.py` stored with the registered model.

`ModelTrainer` remains a supported alternative for script-first training with explicit channels and output artifacts, but the current notebook flow does not execute a second training path.

## Container contract

- **Training and remote functions** — SageMaker runs `docker run <image> train`. The `sagemaker-training` toolkit reads `SAGEMAKER_PROGRAM` and executes the code staged under `/opt/ml/code/`. Standard training paths such as `/opt/ml/input/data/` and `/opt/ml/model/` remain available to script-mode jobs.
- **Processing** — `FrameworkProcessor` invokes the supplied processing script directly; no special image entrypoint is required.
- **Inference** — SageMaker runs `docker run <image> serve`. The launcher starts uvicorn on port 8080 and exposes `GET /ping` and `POST /invocations`. `serving_app.py` delegates to `model_fn`, `input_fn`, `predict_fn`, and `output_fn` from the model's `inference.py`.

The image contains no model-specific training or inference implementation. Those functions and scripts are supplied at job or deployment time.

## Build and push

From a SageMaker Studio terminal, after notebook 00 enables Docker access:

```bash
cd ~/aim-408-end-to-end-modelops-on-sagemaker-ai/byoc
./build_and_push.sh
```

Optional arguments:

```bash
./build_and_push.sh <region> <repository-name> <tag>
```

The script resolves the account and Region, creates the ECR repository when needed, authenticates Docker, builds with Studio's `sagemaker` network, and pushes the image.

## Use an existing build

Set and store the image URI if you want to reuse an image instead of rebuilding it:

```python
account_id = session.account_id()
region = session.boto_region_name
xgboost_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost-byoc:1.0"
%store xgboost_image
```

Notebooks 01, 03, and 04 restore this value directly. The inference specification registered by notebooks 01 and 03 carries the same image URI into deployment.

## Trade-offs

| | AWS-managed base image | BYOC image |
|---|---|---|
| Ownership | AWS maintains the base | You maintain every layer |
| Transparency | Base implementation is external | Dockerfile shows the complete runtime |
| Cross-account ECR pull | May be required | No; image is in your account |
| Patch cadence | Follows AWS releases | Follows your rebuild cadence |

## Verify the build

Run a local import check after the image is built:

```bash
docker run --rm <image-uri> python -c "import fastapi, uvicorn, sagemaker_training, xgboost, sklearn, mlflow; print('ok')"
```

The end-to-end verification is the workshop itself: notebook 01 runs the reusable functions with `@remote`, notebook 03 runs them with `@step`, and notebook 04 deploys the registered model and runs native ProcessingJob-based monitoring.
