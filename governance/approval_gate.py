"""Two-persona approval gate for the SageMaker Model Registry.

The workshop separates two personas with IAM guardrails, following
https://aws.amazon.com/blogs/machine-learning/govern-models-with-mlflow-and-amazon-sagemaker-ai-model-registry-sync-part-1/

* **Data scientist** - registers models from MLflow, moves them to the ``staging``
  lifecycle stage and tags them. An IAM ``Deny`` with the
  ``sagemaker:ModelLifeCycle/stage`` and ``sagemaker:ModelApprovalStatus``
  condition keys stops this role from promoting a version to ``production`` or
  setting its approval status to ``Approved`` (see ``data_scientist_policy.json``).
* **Governance officer** - reviews the synced metrics, model card and lineage, then
  promotes the version to ``production``/``active`` and approves it for deployment
  (see ``governance_officer_policy.json``). This role carries no such deny.

Both roles trust the Studio execution role, so a single workshop user can act as
either persona by assuming the corresponding role with STS. In a real organisation
the two roles are attached to different people (or different accounts, see Part 2
of the blog series).

Lifecycle stages follow the predefined template that the MLflow sync understands:
stage ``staging`` | ``production``, status ``pending`` | ``active``. Every lifecycle
change in the workshop is an ``UpdateModelPackage`` call made with a persona's
credentials, so the IAM policy of that persona is what allows or denies it. The
MLflow alias ``sagemakerlifecycle-<stage>-<status>`` is set afterwards to mirror the
stage in MLflow; it is not the gate, because the MLflow app applies aliases with its
own service role.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

POLICY_DIR = Path(__file__).resolve().parent

DATA_SCIENTIST = "data-scientist"
GOVERNANCE_OFFICER = "governance-officer"

# The two persona roles are pre-provisioned in the workshop account.
PERSONA_ROLE_NAMES = {
    DATA_SCIENTIST: "from-idea-to-prod-data-scientist-role",
    GOVERNANCE_OFFICER: "from-idea-to-prod-governance-officer-role",
}

STAGING_PENDING_ALIAS = "sagemakerlifecycle-staging-pending"
PRODUCTION_ACTIVE_ALIAS = "sagemakerlifecycle-production-active"


# --------------------------------------------------------------------------- IAM setup
def load_persona_policy(persona, region, account_id, model_package_group_prefix):
    """Read the reference copy of a persona policy from ``<persona>_policy.json``.

    The deployed policies are pre-provisioned; these JSON files mirror them so the
    policy can be read."""
    file_name = f"{persona.replace('-', '_')}_policy.json"
    template = (POLICY_DIR / file_name).read_text()
    rendered = (
        template.replace("${REGION}", region)
        .replace("${ACCOUNT_ID}", account_id)
        .replace("${MODEL_PACKAGE_GROUP_PREFIX}", model_package_group_prefix)
    )
    return json.loads(rendered)


def resolve_persona_role(iam_client, role_name):
    """Return the ARN and inline policy of a persona role."""
    try:
        role = iam_client.get_role(RoleName=role_name)["Role"]
    except iam_client.exceptions.NoSuchEntityException:
        raise LookupError(
            f"IAM role '{role_name}' was not found in this account. Ask your workshop "
            "administrator for the approval-gate persona roles."
        ) from None
    # Reading the policy is for display only, so not being allowed to read it must not
    # stop the workshop: the guardrail is enforced by IAM when the persona calls
    # SageMaker, not by anything shown here.
    policy = None
    try:
        policy_names = iam_client.list_role_policies(RoleName=role_name).get("PolicyNames", [])
        if policy_names:
            policy = iam_client.get_role_policy(
                RoleName=role_name, PolicyName=policy_names[0]
            )["PolicyDocument"]
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("AccessDenied", "AccessDeniedException"):
            raise
        print("  (not allowed to read this role's policy, so it is not shown here)")
    return role["Arn"], policy


def describe_guardrail(policy_document):
    """One line per statement of a persona policy: what it allows, what it denies."""
    if not policy_document:
        print("  (no inline policy readable on the role)")
        return
    for statement in policy_document.get("Statement", []):
        condition = statement.get("Condition", {})
        keys = [f"{k} = {v}" for c in condition.values() for k, v in c.items()]
        actions = statement.get("Action")
        actions = [actions] if isinstance(actions, str) else actions
        print(f"  {statement['Effect']:<6} {', '.join(actions)}")
        if keys:
            print(f"         when {'; '.join(keys)}")


# --------------------------------------------------------------------------- personas
def assume_persona(role_arn, persona, region=None, duration_seconds=900):
    """Assume ``role_arn`` and return a SageMaker client whose requests are signed
    as that persona. IAM evaluates the persona's policies on every call."""
    sts = boto3.client("sts")
    credentials = sts.assume_role(
        RoleArn=role_arn, RoleSessionName=persona, DurationSeconds=duration_seconds
    )["Credentials"]
    session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=region,
    )
    identity = session.client("sts").get_caller_identity()["Arn"]
    print(f"Acting as {persona}: {identity}")
    return session.client("sagemaker")


def try_update_model_package(sm_client, **update_kwargs):
    """Call ``UpdateModelPackage`` and report the IAM decision instead of raising.

    Returns ``(True, "allowed")`` or ``(False, <AccessDenied message>)``. Any other
    error is re-raised - only the authorization outcome is part of the demo."""
    try:
        sm_client.update_model_package(**update_kwargs)
        return True, "allowed"
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            return False, e.response["Error"]["Message"]
        raise


# --------------------------------------------------------------------------- registry state
def lifecycle_summary(sm_client, model_package_arn):
    """Stage, stage status and approval status of a Model Package version."""
    described = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    lifecycle = described.get("ModelLifeCycle") or {}
    return {
        "model_package_group": described["ModelPackageGroupName"],
        "version": described["ModelPackageVersion"],
        "stage": lifecycle.get("Stage", "(none)"),
        "stage_status": lifecycle.get("StageStatus", "(none)"),
        "stage_description": lifecycle.get("StageDescription", ""),
        "approval_status": described.get("ModelApprovalStatus", "(not set)"),
        "approval_description": described.get("ApprovalDescription", ""),
        "customer_metadata": described.get("CustomerMetadataProperties", {}),
    }


def print_lifecycle(sm_client, model_package_arn, title=None):
    summary = lifecycle_summary(sm_client, model_package_arn)
    if title:
        print(title)
    print(f"  Model package:    {summary['model_package_group']} v{summary['version']}")
    print(f"  Lifecycle stage:  {summary['stage']} / {summary['stage_status']}")
    print(f"  Approval status:  {summary['approval_status']}")
    if summary["approval_description"]:
        print(f"  Approval note:    {summary['approval_description']}")
    return summary


def model_package_tags(sm_client, model_package_arn):
    """Resource tags of a Model Package version as a plain dict."""
    tags = sm_client.list_tags(ResourceArn=model_package_arn)["Tags"]
    return {t["Key"]: t["Value"] for t in tags}


def tag_model_package(sm_client, model_package_arn, tags):
    """Attach resource tags to a Model Package version (or group). Resource tags -
    unlike MLflow tags - are what IAM ``aws:ResourceTag`` conditions, Resource
    Groups and cost allocation see."""
    sm_client.add_tags(
        ResourceArn=model_package_arn,
        Tags=[{"Key": k, "Value": str(v)[:256]} for k, v in tags.items() if v not in (None, "")],
    )
    return model_package_tags(sm_client, model_package_arn)


# --------------------------------------------------------------------------- transitions
def set_lifecycle_stage(persona_sm_client, model_package_arn, stage, status, description=None,
                        mlflow_client=None, registered_model_name=None, version=None):
    """Set a Model Package version's lifecycle stage **as a persona**, letting IAM decide.

    ``UpdateModelPackage`` is called with the persona's SageMaker client, so the
    condition keys on that role are evaluated: the data-scientist role gets
    ``staging`` through and is denied ``production`` (``AccessDeniedException``).
    If ``mlflow_client`` (plus name and version) is given, the matching
    ``sagemakerlifecycle-<stage>-<status>`` alias is set afterwards so MLflow shows
    the same stage. The alias alone is not a gate: the MLflow app syncs aliases
    under its own service role, so IAM is only evaluated on the direct call."""
    persona_sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelLifeCycle={
            "Stage": stage,
            "StageStatus": status,
            "StageDescription": description or f"Set to {stage}/{status}",
        },
    )
    if mlflow_client is not None:
        alias = f"sagemakerlifecycle-{stage}-{status}"
        mlflow_client.set_registered_model_alias(registered_model_name, alias, version)
        print(f"MLflow alias '{alias}' set on {registered_model_name} v{version}")
    return lifecycle_summary(persona_sm_client, model_package_arn)


def promote_to_production(sm_client, model_package_arn, approved_by, note=None, lock=True):
    """Governance officer path: one ``UpdateModelPackage`` call moves the version to
    ``production``/``active`` **and** sets ``ModelApprovalStatus=Approved``. Lifecycle
    stage and approval status are separate attributes: the first says where the
    model is in its life, the second says it may be deployed.

    Afterwards the version is tagged with the reviewer and time, and (optionally)
    ``locked=true`` - the data-scientist policy denies any further update to a
    locked version. Tag-based conditions can take a few minutes to propagate."""
    approved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = note or f"Promoted to production and approved for deployment by {approved_by}"
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelLifeCycle={
            "Stage": "production",
            "StageStatus": "active",
            "StageDescription": note,
        },
        ModelApprovalStatus="Approved",
        ApprovalDescription=note[:1024],
    )
    tags = {"approved-by": approved_by, "approved-at": approved_at}
    if lock:
        tags["locked"] = "true"
    tag_model_package(sm_client, model_package_arn, tags)
    return print_lifecycle(sm_client, model_package_arn, "After governance officer action:")
