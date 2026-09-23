# Origin and adaptation notes

The files in this directory were copied byte-for-byte from the deployment seed archive referenced by [`cfn/endpoint_deployment_toolchain_template.yml`](../cfn/endpoint_deployment_toolchain_template.yml):

- Source: `s3://sagemaker-servicecatalog-seedcode-us-west-2/toolchain/mpg-deployment-config-v1.1.zip`
- Retrieved: 2026-09-23
- Archive size: 6,995 bytes
- Archive SHA-256: `02da5a1e4b110290595a0328744fb3047a93a2136e80608d492b5e0612146ccc`
- AWS object version: `KH_y4AQ8OMGVtHFxo5tEN692YzMQfNsU`

The archive contains only deployment code. It has no model-build or training pipeline.

## How it is used

The custom project template creates a CodeCommit repository from a ZIP in S3. CodePipeline then:

1. runs `build.py` through `buildspec.yml`;
2. finds the newest SageMaker Model Package with `ModelApprovalStatus=Approved`;
3. renders staging and production CloudFormation parameter files;
4. deploys `endpoint-config-template.yml` to staging;
5. runs `test/test.py`;
6. waits for manual approval; and
7. deploys production.

Notebook [`06-publish-deployment-project-template.ipynb`](../06-publish-deployment-project-template.ipynb) packages this directory at the ZIP root, uploads it and the deployment-only project template to your S3 bucket, marks the template as visible to Studio, and can optionally create a SageMaker Project from it.

## Common adaptations

- `staging-config.json` and `prod-config.json`: instance type/count and data-capture settings.
- `endpoint-config-template.yml`: endpoint resources, autoscaling, networking, alarms, or tags.
- `test/test.py`: replace the placeholder invocation with a real request and assertions before treating the staging gate as production quality.
- `build.py`: selection rules for Model Package versions. The original chooses by `Approved` status only; it does not check SageMaker Model Registry lifecycle stage.
