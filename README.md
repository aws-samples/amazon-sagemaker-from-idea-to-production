# AIM 408 — End-to-end ModelOps on Amazon SageMaker AI

This workshop takes you from developing an ML idea in a simple notebook to a production-ready solution with SageMaker AI processing and training jobs, an automated ML pipeline with model registry, and model monitoring.

It is based on the [Amazon SageMaker MLOps: from idea to production in six steps](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) workshop.

## Workshop flow

Work through the notebooks in order:

| Notebook | What you do |
|---|---|
| [00-start-here.ipynb](00-start-here.ipynb) | Environment setup: SageMaker domain, MLflow tracking, data download |
| [01-mlflow-and-sagemaker-jobs.ipynb](01-mlflow-and-sagemaker-jobs.ipynb) | **Part 1** — experiment locally in the notebook (data processing, training, evaluation in plain Python) with experiment tracking in SageMaker managed MLflow. **Part 2** — scale the same logic with SageMaker AI processing and training jobs (`@remote`, `ModelTrainer`, a BYOC XGBoost container), then register the trained model with a logged inference specification and deploy the auto-synced Model Package version to a real-time endpoint |
| [03-sagemaker-pipeline.ipynb](03-sagemaker-pipeline.ipynb) | Operationalize with a SageMaker pipeline built with the `@step` decorator, and register models in the Model Registry |
| [04-monitoring-with-evidently.ipynb](04-monitoring-with-evidently.ipynb) | Deploy a real-time endpoint with data capture and add model monitoring with Evidently |

> Each capability is exercised exactly once: the preprocessing logic is prototyped in the notebook and then executed once as a managed job, and there is a single model registration and a single real-time endpoint, which serves the registered and approved Model Package.

## Repository structure

```
├── 0x-*.ipynb            # Workshop notebooks (see table above)
├── byoc/                 # Bring-your-own-container: Dockerfile, training and serving code,
│                         # and build_and_push.sh to publish the custom XGBoost image to ECR
├── img/                  # Images and diagrams used in the notebooks
├── pipeline_steps/       # Python functions used as pipeline steps in notebook 03
│                         # (preprocess, train_fn, evaluate, register, inference)
└── setup-py312.sh        # Creates the Python 3.12 kernel required by the notebooks
```

## Prerequisites

- An AWS account with a SageMaker AI domain and a JupyterLab space in SageMaker Studio
- **SageMaker Distribution Image 4.3.2** and **SageMaker Python SDK 3.20.0** — the versions the notebooks were tested with
- **Python 3.12** — all notebooks must run on a Python 3.12 kernel

If your JupyterLab space does not have a Python 3.12 kernel, open a terminal and run:

```bash
./setup-py312.sh
```

Then refresh your browser and select the **Python 3.12** kernel in each notebook.

## Getting started

1. Clone this repository into your JupyterLab space
2. Open [00-start-here.ipynb](00-start-here.ipynb) and follow the instructions
3. Continue with the notebooks in order

## License

See the upstream [amazon-sagemaker-from-idea-to-production](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) repository for license terms (MIT-0).
