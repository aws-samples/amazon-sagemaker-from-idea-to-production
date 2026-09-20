"""EventBridge -> Lambda: deploy a Model Package version the moment it is approved.

Triggered by the ``SageMaker Model Package State Change`` event that SageMaker emits
whenever a Model Package version is created or updated. The EventBridge rule already
narrows it down to ``ModelApprovalStatus = Approved`` on the workshop's model package
groups; this function re-reads the registry and does the rest.

Design:

* **The registry state is the contract.** Nothing is deployed unless the version is
  ``Approved`` *and* in the required lifecycle stage (``production`` by default) *and*
  carries an inference specification. The event is only a trigger; every decision is
  taken from ``DescribeModelPackage``, so a stale or replayed event cannot deploy the
  wrong thing.
* **One stable endpoint, updated in place.** The first approval creates the endpoint;
  every later approval creates a new ``Model`` and ``EndpointConfig`` and switches the
  endpoint over with ``UpdateEndpoint`` (SageMaker performs a blue/green deployment:
  the new fleet is provisioned and health-checked before traffic moves and the old
  fleet is released, so the endpoint name and its callers never change). Re-approving
  an earlier version is therefore a rollback.
* **Data capture stays on.** The endpoint configuration carries the same
  ``DataCaptureConfig`` notebook 04 uses, so the monitoring built there keeps working
  across deployments.
* **Idempotent and loop-safe.** SageMaker may deliver several events per state change,
  and the model card automation (notebook 02) updates the package right after the
  approval, which emits one more ``Approved`` event. Every invocation first checks
  which Model Package the endpoint serves, or is switching to, and exits without a
  write when it is already this one. A busy endpoint (``Creating``/``Updating``) is
  reported as ``deferred`` instead of failing the invocation.
* **Provenance on the endpoint, not on the package.** The version ARN, group and
  version number are recorded as tags on the ``Model``, ``EndpointConfig`` and
  ``Endpoint``. Nothing is written back to the Model Package, so the function never
  triggers itself.

Configuration (environment variables):

  ENDPOINT_NAME              (required) the endpoint to create or update
  EXECUTION_ROLE_ARN         (required) role the endpoint assumes to pull the image and
                             read the model artifacts (the SageMaker execution role)
  INSTANCE_TYPE              default ml.m5.xlarge; must be in the package's
                             SupportedRealtimeInferenceInstanceTypes, otherwise the
                             first supported type is used
  INSTANCE_COUNT             default 1
  DATA_CAPTURE_S3_URI        enable data capture (100 %, Input+Output) to this prefix;
                             leave empty to deploy without capture
  REQUIRED_LIFECYCLE_STAGE   default production; empty string disables the check
"""
import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AUTOMATION_TAG = "deployment-automation"
AUTOMATION_VERSION = "1"

# Endpoint states in which UpdateEndpoint is rejected. The invocation is reported as
# deferred; the next Approved event (or a manual re-invoke) picks the work up again.
BUSY_ENDPOINT_STATES = {"Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"}

_sm_client = None


def _sm():
    global _sm_client
    if _sm_client is None:
        _sm_client = boto3.client("sagemaker")
    return _sm_client


def _env(name, default=None, required=False):
    value = os.environ.get(name, "").strip()
    if not value and required:
        raise RuntimeError(f"Environment variable {name} is required")
    return value or default


# --------------------------------------------------------------------------- registry
def _deployability(described, required_stage):
    """Return None when the version may be deployed, otherwise the reason it may not."""
    if described.get("ModelPackageStatus") not in (None, "Completed"):
        return f"ModelPackageStatus={described.get('ModelPackageStatus')}"
    if described.get("ModelApprovalStatus") != "Approved":
        return f"ModelApprovalStatus={described.get('ModelApprovalStatus', '(not set)')}"
    if required_stage:
        stage = (described.get("ModelLifeCycle") or {}).get("Stage", "(none)")
        if stage != required_stage:
            return f"ModelLifeCycle.Stage={stage}, required {required_stage}"
    containers = (described.get("InferenceSpecification") or {}).get("Containers") or []
    if not containers:
        return "no InferenceSpecification on the model package"
    return None


def _instance_type(described, preferred):
    supported = (described.get("InferenceSpecification") or {}).get(
        "SupportedRealtimeInferenceInstanceTypes"
    ) or []
    if not supported or preferred in supported:
        return preferred
    logger.warning("%s is not in %s; using %s", preferred, supported, supported[0])
    return supported[0]


# --------------------------------------------------------------------------- endpoint state
def _describe_endpoint(endpoint_name):
    try:
        return _sm().describe_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException" and "Could not find" in e.response["Error"]["Message"]:
            return None
        raise


def _model_packages_of_config(endpoint_config_name):
    """Model Package ARNs served by the models of an endpoint configuration."""
    arns = set()
    try:
        config = _sm().describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    except ClientError:
        return arns
    for variant in config.get("ProductionVariants", []):
        try:
            model = _sm().describe_model(ModelName=variant["ModelName"])
        except ClientError:
            continue
        containers = model.get("Containers") or [model.get("PrimaryContainer") or {}]
        for container in containers:
            if container.get("ModelPackageName"):
                arns.add(container["ModelPackageName"])
    return arns


def _served_model_packages(endpoint):
    """Model Packages the endpoint serves now, plus the one it is switching to (if any)."""
    config_names = {endpoint.get("EndpointConfigName")}
    pending = endpoint.get("PendingDeploymentSummary") or {}
    config_names.add(pending.get("EndpointConfigName"))
    arns = set()
    for name in filter(None, config_names):
        arns |= _model_packages_of_config(name)
    return arns


# --------------------------------------------------------------------------- deployment
def _resource_name(endpoint_name, version):
    # Model and EndpointConfig names are immutable and must be unique, so every
    # deployment of a version gets its own pair; the endpoint name stays stable.
    return f"{endpoint_name}-v{version}-{time.strftime('%y%m%d%H%M%S', time.gmtime())}"[:63]


def _tags(described, extra=None):
    tags = {
        AUTOMATION_TAG: AUTOMATION_VERSION,
        "model-package-arn": described["ModelPackageArn"],
        "model-package-group": described["ModelPackageGroupName"],
        "model-package-version": str(described["ModelPackageVersion"]),
        "approved-by": (described.get("CustomerMetadataProperties") or {}).get("ApprovedBy", ""),
    }
    tags.update(extra or {})
    return [{"Key": k, "Value": str(v)[:256]} for k, v in tags.items() if v]


def deploy(described, settings):
    """Create Model + EndpointConfig for the version, then create or update the endpoint.

    Returns an action string. ``settings`` holds the resolved configuration."""
    endpoint_name = settings["endpoint_name"]
    model_package_arn = described["ModelPackageArn"]

    endpoint = _describe_endpoint(endpoint_name)
    if endpoint:
        if model_package_arn in _served_model_packages(endpoint):
            return f"no-op: {endpoint_name} already serves v{described['ModelPackageVersion']}"
        status = endpoint["EndpointStatus"]
        if status in BUSY_ENDPOINT_STATES:
            return f"deferred: {endpoint_name} is {status}"
        if status == "Failed":
            # A failed endpoint cannot be updated, only replaced. It serves nothing.
            logger.warning("Deleting failed endpoint %s: %s", endpoint_name, endpoint.get("FailureReason"))
            _sm().delete_endpoint(EndpointName=endpoint_name)
            _sm().get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
            endpoint = None

    name = _resource_name(endpoint_name, described["ModelPackageVersion"])
    tags = _tags(described)

    # What to serve: the Model Package carries image, model data source and
    # environment, so the container definition is just its ARN.
    _sm().create_model(
        ModelName=name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=settings["execution_role_arn"],
        Tags=tags,
    )

    config_kwargs = {
        "EndpointConfigName": name,
        "ProductionVariants": [{
            "VariantName": "AllTraffic",
            "ModelName": name,
            "InitialInstanceCount": settings["instance_count"],
            "InstanceType": _instance_type(described, settings["instance_type"]),
        }],
        "Tags": tags,
    }
    if settings["data_capture_s3_uri"]:
        config_kwargs["DataCaptureConfig"] = {
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": settings["data_capture_s3_uri"],
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"]},
        }
    _sm().create_endpoint_config(**config_kwargs)

    if endpoint is None:
        response = _sm().create_endpoint(EndpointName=endpoint_name, EndpointConfigName=name, Tags=tags)
        action = f"created: {endpoint_name} <- v{described['ModelPackageVersion']} ({name})"
    else:
        previous = sorted(_served_model_packages(endpoint))
        response = _sm().update_endpoint(EndpointName=endpoint_name, EndpointConfigName=name)
        # AddTags overwrites existing values for the same keys, so the endpoint's tags
        # always describe the version it is moving to.
        _sm().add_tags(ResourceArn=response["EndpointArn"], Tags=tags)
        action = (f"updated: {endpoint_name} {[p.rsplit('/', 1)[-1] for p in previous]} -> "
                  f"v{described['ModelPackageVersion']} ({name})")
    logger.info("%s -> %s", action, response["EndpointArn"])
    return action


# --------------------------------------------------------------------------- handler
def settings_from_env():
    return {
        "endpoint_name": _env("ENDPOINT_NAME", required=True),
        "execution_role_arn": _env("EXECUTION_ROLE_ARN", required=True),
        "instance_type": _env("INSTANCE_TYPE", "ml.m5.xlarge"),
        "instance_count": int(_env("INSTANCE_COUNT", "1")),
        "data_capture_s3_uri": _env("DATA_CAPTURE_S3_URI", ""),
        "required_lifecycle_stage": os.environ.get("REQUIRED_LIFECYCLE_STAGE", "production").strip(),
    }


def handle_model_package(model_package_arn, settings=None):
    """Decide from the registry whether to deploy ``model_package_arn``, and do it."""
    settings = settings or settings_from_env()
    described = _sm().describe_model_package(ModelPackageName=model_package_arn)
    reason = _deployability(described, settings["required_lifecycle_stage"])
    if reason:
        return f"skipped: {reason}"
    return deploy(described, settings)


def lambda_handler(event, context):
    detail = event.get("detail") or {}
    model_package_arn = detail.get("ModelPackageArn") or (event.get("resources") or [None])[0]
    if not model_package_arn:
        logger.error("Event has no ModelPackageArn: %s", json.dumps(event))
        return {"status": "ignored", "reason": "no ModelPackageArn"}

    logger.info(
        "Model package %s (group=%s, version=%s, approval=%s)",
        model_package_arn,
        detail.get("ModelPackageGroupName"),
        detail.get("ModelPackageVersion"),
        detail.get("ModelApprovalStatus"),
    )
    action = handle_model_package(model_package_arn)
    logger.info("%s: %s", model_package_arn, action)
    return {"status": "ok", "model_package_arn": model_package_arn, "action": action}
