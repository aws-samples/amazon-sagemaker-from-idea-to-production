# SageMaker model-deployment project template

[`endpoint_deployment_toolchain_template.yml`](endpoint_deployment_toolchain_template.yml) is the exact CloudFormation template behind the SageMaker Service Catalog product **`[Not supported] MLOps template for model deployment`** in `us-west-2`, retrieved on 2026-09-23 from:

- Product ID: `prod-w345pvwt5qufy`
- Product view ID: `prodview-iooyougb7gvx4`
- Provisioning artifact: `pa-vivdxpf35bcuw` (`v1.5`, “Updating code build docker image to latest version”, 2024-12-10)
- Launch path: `lpv3-4emgcsk733ccw`
- AWS-owned source: `s3://ciclo-us-west-2-prod-product-templates/endpoint_deployment_toolchain_template.yml`
- SHA-256: `1a6349150e4b984641202718cbae55e1497f7185d1c14afeebf67fb825f5b960`

The AWS-owned S3 object is private. The template was retrieved from the original CloudFormation stack of a temporary SageMaker Project provisioned through the product's normal launch path, then the project was deleted. No endpoint was created during retrieval. The Project's retained bucket, `sagemaker-project-p-k22obfchpxot`, was emptied; this Studio role lacks `s3:DeleteBucket`, so an administrator may remove the empty bucket.


## Organization-owned S3 derivative

[`deployment_only_project_template.yml`](deployment_only_project_template.yml) is a deployment-only derivative used by [`06-publish-deployment-project-template.ipynb`](../06-publish-deployment-project-template.ipynb). The original snapshot above remains unchanged. The derivative adds two parameters—`DeploymentSeedCodeBucket` and `DeploymentSeedCodeKey`—and uses them in `ModelDeployCodeCommitRepository.Code.S3` instead of the hard-coded AWS-owned seed object.

The notebook packages [`deployment-seed-code/`](../deployment-seed-code/) at the ZIP root, uploads both artifacts to your S3 bucket, applies the `sagemaker:studio-visibility=true` object tag, and can launch the template directly through SageMaker `TemplateProviders/CfnTemplateProvider`. No model-build pipeline or custom Service Catalog product is involved.

## What it creates

The template accepts `SourceModelPackageGroupName`, `SageMakerProjectName`, and `SageMakerProjectId`, and creates:

- a CodeCommit repository seeded from `sagemaker-servicecatalog-seedcode-us-west-2/toolchain/mpg-deployment-config-v1.1.zip`;
- an EventBridge rule for Model Package state changes in the configured group;
- an EventBridge rule for commits to the repository's `main` branch;
- CodeBuild projects for generating endpoint CloudFormation artifacts and testing staging;
- a CodePipeline with `Source → Build → DeployStaging → TestStaging → ApproveDeployment → DeployProd`;
- an S3 artifact bucket retained when the toolchain stack is deleted.

The seed repository's `build.py` resolves the newest `Approved` Model Package version in the group. Staging and production endpoint definitions are generated from `endpoint-config-template.yml`, `staging-config.json`, and `prod-config.json` in that repository.

## Important limitations

AWS documentation says the CodeCommit-based SageMaker project templates were removed for new projects on 2024-10-28. This product remains visible and launchable in the workshop account, but AWS labels it **not supported**. It may be absent or unusable in other accounts, particularly accounts without existing CodeCommit access.

The template is region-specific as retrieved: its CodeCommit seed archive points to a hard-coded `us-west-2` bucket. Treat this file as a versioned reference snapshot, not as a portable or supported AWS template.

Notebook [`05-automated-model-deployment.ipynb`](../05-automated-model-deployment.ipynb) uses the normal SageMaker Project and Service Catalog launch path. It accepts a Model Package Group **ARN**, validates it, extracts the group name required by the product, discovers the product and latest artifact at runtime, and provisions the deployment toolchain only after an explicit opt-in flag is enabled.
