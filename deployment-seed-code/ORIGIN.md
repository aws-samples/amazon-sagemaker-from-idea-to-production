# Origin and adaptation notes

The files in this directory originated from the AWS deployment seed archive used by the SageMaker-provided MLOps project templates for model deployment:

- Source: `s3://sagemaker-servicecatalog-seedcode-us-west-2/toolchain/mpg-deployment-config-v1.1.zip`
- Archive SHA-256: `02da5a1e4b110290595a0328744fb3047a93a2136e80608d492b5e0612146ccc`

The archive contains only deployment code. It has no model-build or training pipeline.

Changes compared to the original archive:

- `test/test.py` sends a real CSV inference request to the staging endpoint instead of only checking the endpoint status.
- `test/payload.csv` contains one held-out, 64-feature record of the workshop's bank marketing dataset.
- Both buildspec files use the Python 3.12 CodeBuild runtime.

## How it is used

The project template [`sm-custom-projects/deployment-only-project-template.yaml`](../sm-custom-projects/deployment-only-project-template.yaml) creates an AWS CodeCommit repository from a ZIP of this directory in S3. CodePipeline then:

1. runs `build.py` through `buildspec.yml`;
2. finds the newest SageMaker model package with `ModelApprovalStatus=Approved`;
3. renders staging and production CloudFormation parameter files;
4. deploys `endpoint-config-template.yml` to staging;
5. runs `test/test.py`;
6. waits for manual approval; and
7. deploys production.

The notebook [`05-templatize-deployment.ipynb`](../05-templatize-deployment.ipynb) packages this directory at the ZIP root, uploads it with the project template to your S3 bucket, and creates a SageMaker project from it.

## Common adaptations

- `staging-config.json` and `prod-config.json`: instance type and count, and data capture settings.
- `endpoint-config-template.yml`: endpoint resources, autoscaling, networking, alarms, or tags.
- `test/test.py` and `test/payload.csv`: the staging test requires an `InService` endpoint with data capture, HTTP 200, a `text/csv` response, and one finite probability in `[0, 1]`. Replace the payload and `--expected-feature-count` in `test/buildspec.yml` when you adapt the seed to another model.
- `build.py`: the rule for selecting a model package version. It selects by `Approved` status only.
