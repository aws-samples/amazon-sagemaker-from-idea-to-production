# Deployment-only SageMaker Project template

[`deployment_only_project_template.yml`](deployment_only_project_template.yml) is the organization-owned S3 custom Project template used by [`05-templatize-deployment.ipynb`](../05-templatize-deployment.ipynb). It preserves the AWS deployment-toolchain pattern but reads the editable repository seed from your own S3 object instead of a hard-coded AWS bucket.

## Parameters

SageMaker injects `SageMakerProjectName` and `SageMakerProjectId`. The notebook supplies:

- `SourceModelPackageGroupName` — the registry group whose approved versions trigger deployment;
- `DeploymentSeedCodeBucket` — the Region-local bucket containing the repository ZIP; and
- `DeploymentSeedCodeKey` — the ZIP's object key.

## What it creates

- a CodeCommit repository initialized from [`deployment-seed-code/`](../deployment-seed-code/);
- EventBridge rules for Model Package state changes and commits to `main`;
- CodeBuild projects for rendering endpoint CloudFormation and running a real staging inference test;
- a CodePipeline with `Source → Build → DeployStaging (deploy, test, manual approval) → DeployProd`;
- an artifact bucket named `sagemaker-project-${SageMakerProjectId}`, retained when the Project stack is deleted.

The seed `build.py` selects the newest `Approved` Model Package in the configured group. The endpoint template creates a SageMaker Model, EndpointConfig with input/output capture, and an endpoint named `${SageMakerProjectName}-${StageName}`. Stage settings are in `staging-config.json` and `prod-config.json`.

## IAM roles

The Project is launched with `AmazonSageMakerProjectsLaunchRole`; resources use the verified per-service role family:

- `AmazonSageMakerProjectsEventsRole`
- `AmazonSageMakerProjectsCodeBuildRole`
- `AmazonSageMakerProjectsCodePipelineRole`
- `AmazonSageMakerProjectsCloudformationRole`
- `AmazonSageMakerProjectsExecutionRole`

The launch role must also allow `codecommit:CreateRepository` for `sagemaker-*` repositories and `iam:PassRole` for those service roles.

## Support limitation

The template retains CodeCommit to match this workshop account and the original SageMaker deployment pattern. AWS labels the original CodeCommit Project product **not supported**, and CodeCommit is unavailable to new customers. Treat this implementation as a reference for an organization-owned successor using CodeConnections and your supported Git provider.
