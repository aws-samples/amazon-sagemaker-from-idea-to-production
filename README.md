# AIM 408 — End-to-end ModelOps on Amazon SageMaker AI

This workshop builds an end-to-end ModelOps reference workflow on SageMaker AI. You prototype and track a model, move the workload to managed jobs, register and govern model versions, automate the workflow with a pipeline, deploy a real-time endpoint, monitor it, and package staged deployment as a SageMaker Project.

It is based on the [Amazon SageMaker MLOps: from idea to production in six steps](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) workshop.

## Workshop flow

Work through the notebooks in order:

| Notebook | What you do |
|---|---|
| [00-start-here.ipynb](00-start-here.ipynb) | Set up the SageMaker domain resources, Python environment, managed MLflow tracking with Model Registry sync, dataset, custom XGBoost container, and data-scientist and governance-officer roles. |
| [01-mlflow-and-sagemaker-jobs.ipynb](01-mlflow-and-sagemaker-jobs.ipynb) | Prototype and track the model locally, then run preprocessing, training, and explanation as managed SageMaker AI jobs. Register one deployable Model Package version and exercise the two-persona approval gate. |
| [02-model-card-automation.ipynb](02-model-card-automation.ipynb) | Create an EventBridge rule and Lambda function that complete each registered version's Model Card and keep its status aligned with the approval workflow. |
| [03-sagemaker-pipeline.ipynb](03-sagemaker-pipeline.ipynb) | Turn the reusable preprocess, train, evaluate, explain, and register functions into a SageMaker pipeline with `@step`. Register only when the evaluation threshold passes, then stage, review, and approve the version. |
| [04-inference-and-monitoring.ipynb](04-inference-and-monitoring.ipynb) | Resolve the latest registry version and require it to be `production` and `Approved`; validate its logged inference code, deploy it with data capture, inspect the endpoint-to-model lineage, and monitor drift and model quality with Evidently and MLflow. Links to examples for automating the checks in production with SageMaker Model Monitor or Evidently pipelines. |
| [05-templatize-deployment.ipynb](05-templatize-deployment.ipynb) | Package [`deployment-seed-code/`](deployment-seed-code/) and publish an organization-owned deployment-only Project template. Its CodePipeline deploys an approved package to staging, runs CSV inference tests, waits for guarded manual approval, then deploys and tests production. |

Complete notebooks 00–04 for the model-development, governance, inference, and monitoring workflow. Notebook 05 adds a separate staged CI/CD path for the same Model Package Group.

> **CodeCommit caveat:** the notebook 05 template retains CodeCommit because it matches the copied AWS toolchain and the workshop account has existing access. AWS labels the original product not supported, and CodeCommit is unavailable to new customers. Treat this implementation as a reference for an organization-owned successor that uses your supported Git provider.

## Model governance: the approval gate

The workshop follows the single-account pattern from [Govern models with MLflow and Amazon SageMaker AI Model Registry sync](https://aws.amazon.com/blogs/machine-learning/govern-models-with-mlflow-and-amazon-sagemaker-ai-model-registry-sync-part-1/). With automatic registry sync enabled, each `mlflow.register_model()` call creates a Model Package version in the SageMaker Model Registry. Two IAM personas then move the version through its lifecycle:

| Persona | Responsibilities | Explicitly denied |
|---|---|---|
| Data scientist ([policy](governance/data_scientist_policy.json)) | Register versions, move them to `staging/pending` with the `sagemakerlifecycle-staging-pending` alias, and record provenance. | `UpdateModelPackage` for `production`, `ModelApprovalStatus = Approved`, or resources tagged `frozen=true`. |
| Governance officer ([policy](governance/governance_officer_policy.json)) | Review metrics, Model Card, and lineage; promote to `production/active`; set `ModelApprovalStatus = Approved`; optionally freeze the group with the [freeze policy](governance/freeze_policy.json). | — |

Lifecycle stage is driven from MLflow: setting the alias `sagemakerlifecycle-<stage>-<status>` updates the synced Model Package version's `ModelLifeCycle`. The enforcement path matters:

- Alias-driven lifecycle updates are performed by the MLflow application's service role, not by the notebook caller. The notebooks therefore demonstrate the data-scientist lifecycle deny with `iam:SimulateCustomPolicy`, following the blog's [reference notebook](https://github.com/aws-samples/genai-ml-platform-examples/blob/main/operations/sagemaker-mlflow-model-registry-goverance/01_single_account_governance.ipynb). In production, attach the deny to both the data-scientist roles and the MLflow application service role.
- `ModelApprovalStatus` is set by a direct `UpdateModelPackage` call, so the calling role's policy applies. Notebooks 01 and 03 demonstrate that denial against the assumed data-scientist role.

SageMaker supports tag operations on Model Package Groups rather than individual versions. Group-wide facts such as project, use case, and team are group resource tags, which IAM `aws:ResourceTag` conditions can evaluate. Per-version facts such as who registered or approved the model, evaluation metrics, and timestamps are stored in `CustomerMetadataProperties`.

The register logic in notebooks 01 and 03 stops at `staging/pending`; it never promotes a version to production. Notebook 04 separately verifies that the latest registry version is both `production` and `Approved` before deploying it. Notebook 05's deployment project selects an `Approved` package for staging, runs an automatic inference test, and requires a separate guarded CodePipeline approval before production.

The workshop provisioning creates both persona roles; the notebooks look them up rather than creating them. Notebook 00 resolves the roles and displays their policies, notebook 01 exercises the gate for a manually registered version, and notebook 03 repeats it for a pipeline-registered version. Shared helpers live in [`governance/approval_gate.py`](governance/approval_gate.py); the JSON policy files are reference copies of the deployed policies.

## Repository structure

```text
├── 0x-*.ipynb             # Workshop notebooks 00–05
├── byoc/                  # Custom XGBoost training/serving image and ECR build script
├── cfn/                   # Deployment-only SageMaker Project template used by notebook 05
├── deployment-seed-code/  # Editable CodePipeline seed: endpoint IaC, stage configs,
│                          # build logic, buildspecs, and inference tests
├── explainability/        # Additional dependencies for the managed SHAP/fairness job
├── governance/            # Approval-gate helpers and reference IAM policies
├── img/                   # Notebook diagrams and screenshots
├── model_card_automation/ # EventBridge-triggered Model Card Lambda used by notebook 02
├── monitoring/            # Reference Evidently drift/quality Processing-job script with SNS
│                          # alerts; not run by the notebooks
├── pipeline_steps/        # Reusable preprocess, train, evaluate, explain, register,
│                          # inference, and runtime functions used by notebooks 01 and 03
├── processing/            # Legacy native-ProcessingJob script retained for reference
├── training/              # Legacy ModelTrainer script retained for reference
├── deployment_automation/ # Legacy/reference endpoint-deployment Lambda; not used by the
│                          # current notebook 00–05 workflow
└── setup-py312.sh         # Creates the Python 3.12 Jupyter kernel
```

Notebook execution also creates local runtime artifacts such as downloaded data, temporary model files, monitoring reports, and deployment-template build output. These are not workshop source files and should not be committed.

## Prerequisites

- An AWS account with a SageMaker AI domain and a JupyterLab space in SageMaker Studio
- A workshop environment with permission to run SageMaker jobs and endpoints and to create the IAM, ECR, Lambda, EventBridge, SNS, CloudFormation, CodeCommit, CodePipeline, CodeBuild, S3, and SageMaker Project resources used by the notebooks
- **SageMaker Distribution Image 4.3.2** and **SageMaker Python SDK 3.20.0**, the tested versions
- **Python 3.12** for every notebook

If the JupyterLab space does not have a Python 3.12 kernel, open a terminal from the repository root and run:

```bash
./setup-py312.sh
```

Refresh the browser, then select the **Python 3.12** kernel in each notebook.

> **Cost:** managed jobs, real-time endpoints, monitoring jobs, and the notebook 05 Project resources incur AWS charges. Follow the guarded cleanup sections in the notebooks when you finish.

## Getting started

1. Clone this repository into the JupyterLab space.
2. Open [00-start-here.ipynb](00-start-here.ipynb) and complete the environment setup.
3. Continue through the notebooks in numeric order.

## License

See the upstream [amazon-sagemaker-from-idea-to-production](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) repository for the MIT-0 license terms.
