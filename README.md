# AIM 408 — End-to-end ModelOps on Amazon SageMaker AI

This workshop takes you from developing an ML idea in a simple notebook to a production-ready solution with SageMaker AI processing and training jobs, an automated ML pipeline with model registry, and model monitoring.

It is based on the [Amazon SageMaker MLOps: from idea to production in six steps](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) workshop.

## Workshop flow

Work through the notebooks in order:

| Notebook | What you do |
|---|---|
| [00-start-here.ipynb](00-start-here.ipynb) | Environment setup: SageMaker domain, MLflow tracking with Model Registry sync, data download, and the **approval gate**: two IAM persona roles (data scientist, governance officer) separated by Model Registry condition keys |
| [01-mlflow-and-sagemaker-jobs.ipynb](01-mlflow-and-sagemaker-jobs.ipynb) | **Part 1** — experiment locally in the notebook (data processing, training, evaluation in plain Python) with experiment tracking in SageMaker managed MLflow. **Part 2** — scale the same logic with SageMaker AI processing and training jobs (`@remote`, `ModelTrainer`, a BYOC XGBoost container), explain the selected model with SHAP and standardized fairness metrics as a managed job, register it with a logged inference specification, group and version tags and the `staging` lifecycle stage, pass the approval gate (data scientist denied, governance officer promotes to `production` and approves), and deploy the approved Model Package version to a real-time endpoint |
| [02-model-card-automation.ipynb](02-model-card-automation.ipynb) | Automate model governance: an EventBridge rule on `SageMaker Model Package State Change` invokes a Lambda function that completes the Model Card of every registered model package version (one card per version, status aligned with the approval workflow) |
| [03-sagemaker-pipeline.ipynb](03-sagemaker-pipeline.ipynb) | Operationalize with a SageMaker pipeline built with the `@step` decorator; the register step stages and tags the version in the Model Registry, and the governance officer promotes and approves it |
| [04-monitoring-with-evidently.ipynb](04-monitoring-with-evidently.ipynb) | Deploy a real-time endpoint with data capture and add model monitoring with Evidently |

> Each capability is exercised exactly once: the preprocessing logic is prototyped in the notebook and then executed once as a managed job, and there is a single model registration and a single real-time endpoint, which serves the registered and approved Model Package.

## Model governance: the approval gate

The workshop follows the single-account pattern from [Govern models with MLflow and Amazon SageMaker AI Model Registry sync](https://aws.amazon.com/blogs/machine-learning/govern-models-with-mlflow-and-amazon-sagemaker-ai-model-registry-sync-part-1/). Every `mlflow.register_model()` call creates a Model Package version in the SageMaker Model Registry; two personas, separated by IAM rather than by accounts, then move it through its lifecycle:

| Persona | Does | Denied |
|---|---|---|
| Data scientist ([policy](governance/data_scientist_policy.json)) | registers versions from MLflow, moves them to `staging/pending` (`UpdateModelPackage` made as this role, mirrored as the `sagemakerlifecycle-staging-pending` MLflow alias), tags versions | `UpdateModelPackage` when `sagemaker:ModelLifeCycle/stage = production`, when `sagemaker:ModelApprovalStatus = Approved`, and on versions tagged `locked=true` |
| Governance officer ([policy](governance/governance_officer_policy.json)) | reviews synced metrics, model card and lineage; sets `ModelLifeCycle = production/active` and `ModelApprovalStatus = Approved`; locks the version | — |

The gate is enforced by IAM only: there is no stage check in the code. Every lifecycle change is an `UpdateModelPackage` call signed as one of the two roles, and the policy attached to that role decides. The pipeline's register step ([`pipeline_steps/register.py`](pipeline_steps/register.py)) makes its lifecycle update as the data-scientist role (`DATA_SCIENTIST_ROLE_ARN`), so a pipeline asking for `production` fails with `AccessDeniedException`.

Both roles are created as part of provisioning of the workshop. The notebooks only look for the roles. Notebook 00 resolves the roles and prints the policy attached to each, notebook 01 exercises the gate for a version registered by hand, and notebook 03 for a version registered by the pipeline. Shared helpers live in [`governance/approval_gate.py`](governance/approval_gate.py); the JSON files above are reference copies of the policies the stack deploys.

## Repository structure

```
├── 0x-*.ipynb            # Workshop notebooks (see table above)
├── byoc/                 # Bring-your-own-container: Dockerfile, training and serving code,
│                         # and build_and_push.sh to publish the custom XGBoost image to ECR
├── governance/           # Approval gate: IAM policy templates for the data-scientist and
│                         # governance-officer personas, and the helpers notebooks 00/01/03 use
├── img/                  # Images and diagrams used in the notebooks
├── model_card_automation/ # Lambda function deployed by notebook 02: completes the Model Card
│                         # of each registered model package version (EventBridge-triggered)
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
