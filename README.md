# AIM 408 — End-to-end ModelOps on Amazon SageMaker AI

This workshop takes you from developing an ML idea in a simple notebook to a production-ready solution with SageMaker AI processing and training jobs, an automated ML pipeline with model registry, and model monitoring.

It is based on the [Amazon SageMaker MLOps: from idea to production in six steps](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) workshop.

## Workshop flow

Work through the notebooks in order:

| Notebook | What you do |
|---|---|
| [00-start-here.ipynb](00-start-here.ipynb) | Environment setup: SageMaker domain, MLflow tracking with Model Registry sync, data download, and the **approval gate**: two IAM persona roles (data scientist, governance officer) separated by Model Registry condition keys |
| [01-mlflow-and-sagemaker-jobs.ipynb](01-mlflow-and-sagemaker-jobs.ipynb) | **Part 1** — experiment locally in the notebook (data processing, training, evaluation in plain Python) with experiment tracking in SageMaker managed MLflow. **Part 2** — scale the same logic with SageMaker AI processing and training jobs (`@remote`, `ModelTrainer`, a BYOC XGBoost container), explain the selected model with SHAP and standardized fairness metrics as a managed job, register it with a logged inference specification, group and version tags and the `staging` lifecycle stage, and pass the approval gate (data scientist denied, governance officer promotes to `production` and approves) |
| [02-model-card-automation.ipynb](02-model-card-automation.ipynb) | Automate model governance: an EventBridge rule on `SageMaker Model Package State Change` invokes a Lambda function that completes the Model Card of every registered model package version (one card per version, status aligned with the approval workflow) |
| [03-sagemaker-pipeline.ipynb](03-sagemaker-pipeline.ipynb) | Operationalize with a SageMaker pipeline built with the `@step` decorator; the register step stages and tags the version in the Model Registry, and the governance officer promotes and approves it |
| [04-inference-and-monitoring.ipynb](04-inference-and-monitoring.ipynb) | **Inference**: resolve the latest approved Model Package version in the registry, validate its logged `inference.py` in the kernel exactly as the serving container runs it, deploy it to a real-time endpoint with **data capture** enabled (no retraining, no repacking), test the endpoint and log the confusion matrix to the model's MLflow run. **Monitoring**: send production traffic and watch it with **Evidently**: data drift against the training baseline and model quality once the ground truth is joined, logged to MLflow; then run the same checks as a scheduled SageMaker pipeline with SNS alerts |
| [05-event-driven-deployment.ipynb](05-event-driven-deployment.ipynb) | Make the approval the deployment: an EventBridge rule on `SageMaker Model Package State Change` with `ModelApprovalStatus = Approved` invokes a Lambda function that re-reads the registry and deploys the version to a stable real-time endpoint with data capture (first approval creates it, later approvals update it blue/green; re-approving an earlier version rolls back). The governance officer's approval is the only thing that moves a model to production; the notebook triggers it, watches the endpoint come up, tests it and proves the function is idempotent |

> Each capability is exercised exactly once: the preprocessing logic is prototyped in the notebook and then executed once as a managed job; notebook 01 registers a version by hand and notebook 03 registers one from the pipeline; notebook 04 deploys the approved Model Package version by hand and notebook 05 lets the `Approved` event do it.

## Model governance: the approval gate

The workshop follows the single-account pattern from [Govern models with MLflow and Amazon SageMaker AI Model Registry sync](https://aws.amazon.com/blogs/machine-learning/govern-models-with-mlflow-and-amazon-sagemaker-ai-model-registry-sync-part-1/). Every `mlflow.register_model()` call creates a Model Package version in the SageMaker Model Registry; two personas, separated by IAM rather than by accounts, then move it through its lifecycle:

| Persona | Does | Denied |
|---|---|---|
| Data scientist ([policy](governance/data_scientist_policy.json)) | registers versions from MLflow, moves them to `staging/pending` with the `sagemakerlifecycle-staging-pending` alias, records their provenance | `UpdateModelPackage` when `sagemaker:ModelLifeCycle/stage = production` or `sagemaker:ModelApprovalStatus = Approved`, and on anything tagged `frozen=true` |
| Governance officer ([policy](governance/governance_officer_policy.json)) | reviews synced metrics, model card and lineage; promotes to `production/active`; sets `ModelApprovalStatus = Approved`; can freeze the group ([freeze policy](governance/freeze_policy.json)) | — |

Lifecycle stage is driven from MLflow: setting the alias `sagemakerlifecycle-<stage>-<status>` on a model version updates the `ModelLifeCycle` of the synced Model Package version. Two consequences shape where a guardrail bites:

- **Alias-driven lifecycle transitions are performed by the MLflow app's service role**, not by the caller. A deny on the notebook role does not stop them, so the notebooks verify that guardrail with `iam:SimulateCustomPolicy` — IAM's own evaluation of the same request — exactly as the blog's [reference notebook](https://github.com/aws-samples/genai-ml-platform-examples/blob/main/operations/sagemaker-mlflow-model-registry-goverance/01_single_account_governance.ipynb) does. Attach the policy to the data-scientist roles *and* to the MLflow app service role to close that route in production.
- **`ModelApprovalStatus` is set by a direct `UpdateModelPackage` call**, so the calling role's policy does apply. The notebooks demonstrate that denial live, against the real data-scientist role.

Provenance follows the shape of the API rather than a preference: SageMaker supports tag operations on Model Package **Groups** only, so group-wide facts (project, use case, team) are resource tags on the group — which is also what `aws:ResourceTag` conditions evaluate — while per-version facts (who registered it, from where, the test AUC, who approved it and when) live in the version's `CustomerMetadataProperties`.

There is no stage check in the workshop code: the register step stops at `staging` and the policies decide the rest. Notebook 05 closes the loop on the other side of the gate: the `Approved` status the governance officer sets is the event that deploys the version, so the approval is not only the permission to deploy but the deployment itself.

Both roles are created as part of provisioning of the workshop; the notebooks only look them up. Notebook 00 resolves the roles and prints the policy attached to each, notebook 01 exercises the gate for a version registered by hand, and notebook 03 for a version registered by the pipeline. Shared helpers live in [`governance/approval_gate.py`](governance/approval_gate.py); the JSON files linked above are reference copies of the deployed policies.

## Repository structure

```
├── 0x-*.ipynb            # Workshop notebooks (see table above)
├── byoc/                 # Bring-your-own-container: Dockerfile, training and serving code,
│                         # and build_and_push.sh to publish the custom XGBoost image to ECR
├── governance/           # Approval gate: IAM policy templates for the data-scientist and
│                         # governance-officer personas, and the helpers notebooks 00/01/03 use
├── img/                  # Images and diagrams used in the notebooks
├── deployment_automation/ # Lambda function deployed by notebook 05: deploys an approved model
│                         # package version to the production endpoint (EventBridge-triggered),
│                         # and the reference IAM policy of its execution role
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
