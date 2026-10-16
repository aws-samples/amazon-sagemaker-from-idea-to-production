"""Model approval gate for the MLflow / SageMaker AI Model Registry sync.

Implements the single-account governance topology from
https://aws.amazon.com/blogs/machine-learning/govern-models-with-mlflow-and-amazon-sagemaker-ai-model-registry-sync-part-1/

Two personas share one account and one registry:

* **Data scientist** - trains in MLflow, logs an inference specification and
  evaluation metrics, registers the model (the sync creates the Model Package Group
  and version), then moves the version to the ``staging`` lifecycle stage.
* **Governance officer** - reviews the synced metrics, model card and lineage,
  promotes the version to ``production``/``active`` and sets
  ``ModelApprovalStatus=Approved`` so it can be deployed.

Two mechanics are worth keeping straight, because they decide where a guardrail can
be enforced:

1. **Lifecycle stage is driven from MLflow, by alias.** Setting the alias
   ``sagemakerlifecycle-<stage>-<status>`` on a model version updates the
   ``ModelLifeCycle`` of the synced Model Package version. A version holds one
   lifecycle alias at a time, so the previous one is removed first.
   The sync applies that update with the **MLflow app's own service role**, not the
   caller's role - so an IAM deny on the *notebook* role is not what stops a
   production promotion made through an alias. The guardrail is therefore expressed
   as a policy and verified with ``iam:SimulateCustomPolicy``
   (see ``simulate_lifecycle_decision``), which is how the blog's reference notebook
   demonstrates it. Attach the same policy to the roles your data scientists assume,
   and to the MLflow app service role, to enforce it in production.
2. **Approval status is set by the caller.** ``UpdateModelPackage`` with
   ``ModelApprovalStatus`` is an ordinary API call, so an IAM deny on the calling
   role does take effect - that one can be demonstrated live.

Lifecycle stages use the predefined template the sync understands: stage
``staging`` | ``production``, status ``pending`` | ``active``, all lowercase
(MLflow aliases only allow lowercase).
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
    DATA_SCIENTIST: "aim408-from-idea-to-prod-data-scientist-role",
    GOVERNANCE_OFFICER: "aim408-from-idea-to-prod-governance-officer-role",
}

LIFECYCLE_STAGES = ("staging", "production")
LIFECYCLE_STATUSES = ("pending", "active")

STAGING_PENDING_ALIAS = "sagemakerlifecycle-staging-pending"
PRODUCTION_ACTIVE_ALIAS = "sagemakerlifecycle-production-active"

LIFECYCLE_STAGE_CONDITION_KEY = "sagemaker:ModelLifeCycle/stage"
LIFECYCLE_STATUS_CONDITION_KEY = "sagemaker:ModelLifeCycle/stageStatus"


def lifecycle_alias(stage, status):
    """``sagemakerlifecycle-<stage>-<status>``, validated against the template."""
    if stage not in LIFECYCLE_STAGES or status not in LIFECYCLE_STATUSES:
        raise ValueError(
            f"Unsupported lifecycle {stage}/{status}. Stage must be one of "
            f"{LIFECYCLE_STAGES} and status one of {LIFECYCLE_STATUSES}, lowercase."
        )
    return f"sagemakerlifecycle-{stage}-{status}"


# --------------------------------------------------------------------------- policies
def load_policy(file_name, region=None, account_id=None, model_package_group_prefix=None):
    """Read a policy document from this folder, filling in the ``${...}`` placeholders."""
    text = (POLICY_DIR / file_name).read_text()
    for placeholder, value in (
        ("${REGION}", region),
        ("${ACCOUNT_ID}", account_id),
        ("${MODEL_PACKAGE_GROUP_PREFIX}", model_package_group_prefix),
    ):
        if value:
            text = text.replace(placeholder, value)
    return json.loads(text)


def load_persona_policy(persona, region=None, account_id=None, model_package_group_prefix=None):
    """Reference copy of a persona's policy, for reading alongside the notebooks."""
    return load_policy(
        f"{persona.replace('-', '_')}_policy.json", region, account_id, model_package_group_prefix
    )


# IAM's policy grammar accepts only these keys. The reference policies in this folder
# carry a top-level "_comment" for readers, which iam:SimulateCustomPolicy rejects with
# "invalid content", so anything sent to IAM is filtered down to the grammar first.
_POLICY_KEYS = ("Version", "Id", "Statement")
_STATEMENT_KEYS = (
    "Sid", "Effect", "Action", "NotAction", "Resource", "NotResource", "Condition",
)


def as_iam_policy_document(policy_document):
    """``policy_document`` reduced to the keys IAM's policy grammar accepts.

    Drops documentation keys such as ``_comment`` at both the document and the
    statement level, so a reference policy can be fed to IAM APIs unchanged."""
    document = {k: v for k, v in policy_document.items() if k in _POLICY_KEYS}
    document.setdefault("Version", "2012-10-17")
    document["Statement"] = [
        {k: v for k, v in statement.items() if k in _STATEMENT_KEYS}
        for statement in policy_document.get("Statement", [])
    ]
    return document


def deny_statements(policy_document):
    """The ``Deny`` statements of a policy - the part that makes it a guardrail."""
    return [s for s in policy_document.get("Statement", []) if s.get("Effect") == "Deny"]


def describe_guardrail(policy_document):
    """One line per statement: what it allows, what it denies, under which condition."""
    if not policy_document:
        print("  (no policy to show)")
        return
    for statement in policy_document.get("Statement", []):
        condition = statement.get("Condition", {})
        keys = [f"{k} = {v}" for c in condition.values() for k, v in c.items()]
        actions = statement.get("Action")
        actions = [actions] if isinstance(actions, str) else actions
        print(f"  {statement['Effect']:<6} {', '.join(actions)}")
        if keys:
            print(f"         when {'; '.join(keys)}")


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
    # stop the workshop: the guardrail is enforced by IAM, not by anything shown here.
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


# --------------------------------------------------------------------------- guardrail check
def _local_decision(policy_document, action, context):
    """Evaluate ``policy_document`` against one request locally.

    Fallback for environments where ``iam:SimulateCustomPolicy`` is not permitted.
    Only the subset the guardrails use is implemented: ``StringEquals`` conditions on
    an action with ``Resource: "*"``. Returns ``"explicitDeny"``, ``"allowed"`` or
    ``"implicitDeny"``, matching the vocabulary of the IAM simulator."""
    decision = "implicitDeny"
    for statement in policy_document.get("Statement", []):
        actions = statement.get("Action")
        actions = [actions] if isinstance(actions, str) else actions
        if not any(a == action or a == "*" or (a.endswith("*") and action.startswith(a[:-1]))
                   for a in actions):
            continue
        conditions = statement.get("Condition", {}).get("StringEquals", {})
        if any(context.get(key) != value for key, value in conditions.items()):
            continue  # condition does not match this request
        if statement.get("Effect") == "Deny":
            return "explicitDeny"
        decision = "allowed"
    return decision


def simulate_lifecycle_decision(iam_client, policy_document, stage, status=None,
                                action="sagemaker:UpdateModelPackage"):
    """Ask IAM what ``policy_document`` decides for a lifecycle transition.

    Returns ``(decision, source)`` where decision is ``allowed`` / ``explicitDeny`` /
    ``implicitDeny`` and source is ``"iam"`` or ``"local"``. This is how the guardrail
    is verified for alias-driven transitions: the sync performs them with the MLflow
    app's service role, so calling the API from here would not exercise the policy.

    The request is deliberately simulated **without** ``ResourceArns``. The lifecycle
    condition keys are SageMaker-specific and not yet in the simulator's condition-key
    catalog; once a concrete resource ARN is supplied the simulator ignores keys it
    does not recognise, the ``StringEquals`` condition can never be satisfied and the
    Deny statement silently stops matching - reporting a production promotion as
    ``allowed``. Evaluated against ``*`` the supplied context is honoured, so an
    allowed transition comes back as ``implicitDeny`` (nothing denies it) and a denied
    one as ``explicitDeny``."""
    context = {LIFECYCLE_STAGE_CONDITION_KEY: stage}
    if status:
        context[LIFECYCLE_STATUS_CONDITION_KEY] = status
    try:
        result = iam_client.simulate_custom_policy(
            PolicyInputList=[json.dumps(as_iam_policy_document(policy_document))],
            ActionNames=[action],
            ContextEntries=[
                {"ContextKeyName": key, "ContextKeyValues": [value], "ContextKeyType": "string"}
                for key, value in context.items()
            ],
        )
        return result["EvaluationResults"][0]["EvalDecision"], "iam"
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("AccessDenied", "AccessDeniedException"):
            raise
        return _local_decision(policy_document, action, context), "local"


# --------------------------------------------------------------------------- personas
def assume_persona(role_arn, persona, region=None, duration_seconds=900):
    """Assume ``role_arn`` and return a SageMaker client signed as that persona."""
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
    print(f"Acting as {persona}: {session.client('sts').get_caller_identity()['Arn']}")
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
    """Stage, stage status, approval status and metadata of a Model Package version."""
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
    approved_by = summary["customer_metadata"].get("ApprovedBy")
    if approved_by:
        print(f"  Approved by:      {approved_by} at {summary['customer_metadata'].get('ApprovedAt', '')}")
    return summary


def resolve_model_package_group(sm_client, registered_model_name, prefix=None):
    """Name of the SageMaker Model Package Group that holds a registered model.

    The sync derives the group name from the MLflow registered model name but appends a
    short hash suffix (``my-model`` -> ``my-model-1a2b3c``), so the two names are not
    interchangeable: ``ListModelPackages`` on the MLflow name returns nothing, without
    raising. Searches narrowest first - the MLflow name as-is, then groups whose name
    contains it, then any group under ``prefix`` - and returns the first that holds at
    least one version."""
    def versions(group_name):
        try:
            return sm_client.list_model_packages(
                ModelPackageGroupName=group_name, SortBy="CreationTime",
                SortOrder="Descending", MaxResults=1,
            )["ModelPackageSummaryList"]
        except ClientError:
            return []

    def groups_containing(fragment):
        # NameContains must match ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$, so a trailing
        # hyphen (which a prefix usually has) has to go.
        names = []
        paginator = sm_client.get_paginator("list_model_package_groups")
        for page in paginator.paginate(
            NameContains=fragment.rstrip("-"), SortBy="CreationTime", SortOrder="Descending"
        ):
            names.extend(g["ModelPackageGroupName"] for g in page["ModelPackageGroupSummaryList"])
        return names

    candidates = [registered_model_name]
    for name in groups_containing(registered_model_name):
        if name not in candidates:
            candidates.append(name)
    if prefix:
        for name in groups_containing(prefix):
            if name.startswith(prefix) and name not in candidates:
                candidates.append(name)

    for candidate in candidates:
        if versions(candidate):
            return candidate
    raise LookupError(
        f"No Model Package Group with a registered version was found for "
        f"'{registered_model_name}'. Groups checked: {candidates}"
    )


def mlflow_version_for_package(mlflow_client, registered_model_name, model_package_arn):
    """MLflow model version whose sync tag points at ``model_package_arn``.

    MLflow version numbers and SageMaker Model Package version numbers usually run in
    step, but nothing guarantees it, and a lifecycle alias has to be set on the right
    MLflow version. The sync records the package ARN as the
    ``sagemaker.model_package_arn`` tag on the model version, so match on that."""
    for mv in mlflow_client.search_model_versions(f"name='{registered_model_name}'"):
        if mv.tags.get("sagemaker.model_package_arn") == model_package_arn:
            return mv.version
    raise LookupError(
        f"No version of MLflow model '{registered_model_name}' is tagged with "
        f"sagemaker.model_package_arn={model_package_arn}"
    )


def latest_model_package(sm_client, registered_model_name, prefix=None):
    """Summary of the most recently created version of a registered model."""
    group_name = resolve_model_package_group(sm_client, registered_model_name, prefix)
    return sm_client.list_model_packages(
        ModelPackageGroupName=group_name, SortBy="CreationTime",
        SortOrder="Descending", MaxResults=1,
    )["ModelPackageSummaryList"][0]


def group_arn_of(model_package_arn):
    """Model Package Group ARN for a version ARN.

    ``arn:...:model-package/<group>/<version>`` -> ``arn:...:model-package-group/<group>``"""
    prefix, _, path = model_package_arn.partition(":model-package/")
    if not path:
        raise ValueError(f"Not a Model Package version ARN: {model_package_arn}")
    return f"{prefix}:model-package-group/{path.split('/')[0]}"


def model_package_group_tags(sm_client, model_package_arn):
    """Resource tags of the version's Model Package **Group**, as a plain dict."""
    tags = sm_client.list_tags(ResourceArn=group_arn_of(model_package_arn))["Tags"]
    return {t["Key"]: t["Value"] for t in tags}


def tag_model_package_group(sm_client, model_package_arn, tags):
    """Attach resource tags to the version's Model Package **Group**.

    Tags attach to the group, not to individual versions: ``AddTags`` against a
    version ARN fails with ``ValidationException``. So resource tags carry facts about
    the model as a whole, and they are what IAM ``aws:ResourceTag`` conditions see.
    Per-version facts belong in ``set_version_metadata``."""
    sm_client.add_tags(
        ResourceArn=group_arn_of(model_package_arn),
        Tags=[{"Key": k, "Value": str(v)[:256]} for k, v in tags.items() if v not in (None, "")],
    )
    return model_package_group_tags(sm_client, model_package_arn)


# CustomerMetadataProperties limits (UpdateModelPackage): 50 entries, key <= 128 and
# value <= 256 characters, both restricted to letters, digits, whitespace and _ . : / = + - @
_METADATA_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.:/=+-@"
)


def _clean_metadata_value(value, limit):
    return "".join(c if c in _METADATA_ALLOWED else "-" for c in str(value))[:limit]


def set_version_metadata(sm_client, model_package_arn, metadata, merge=True):
    """Write per-version facts into the version's ``CustomerMetadataProperties``.

    This is the per-version counterpart of group tags: who registered this version,
    from where, out of which MLflow run. New values are sanitised to the character set
    and lengths the API accepts. With ``merge=True`` (the default) metadata already on
    the version is read and sent back unchanged alongside the new keys, so this never
    drops entries written earlier (the model card automation reads some of them)."""
    properties = {}
    if merge:
        # Sanitise carried-over values too: a version registered before this encoding
        # was in place may hold a value the API would now reject (for example metrics
        # stored as JSON), and re-sending it unchanged would fail the whole call.
        properties.update({
            _clean_metadata_value(k, 128): _clean_metadata_value(v, 256)
            for k, v in sm_client.describe_model_package(ModelPackageName=model_package_arn)
            .get("CustomerMetadataProperties", {}).items()
        })
    properties.update({
        _clean_metadata_value(k, 128): _clean_metadata_value(v, 256)
        for k, v in metadata.items()
        if v not in (None, "")
    })
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn, CustomerMetadataProperties=properties
    )
    return sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    ).get("CustomerMetadataProperties", {})


# --------------------------------------------------------------------------- transitions
def set_lifecycle(mlflow_client, registered_model_name, version, stage, status):
    """Move a model version to ``stage``/``status`` by setting its MLflow alias.

    A version can hold only one lifecycle alias at a time, so any existing one is
    removed first. The sync then updates the Model Package's ``ModelLifeCycle``;
    use ``wait_for_lifecycle`` to observe it landing."""
    alias = lifecycle_alias(stage, status)
    for existing_stage in LIFECYCLE_STAGES:
        for existing_status in LIFECYCLE_STATUSES:
            existing = f"sagemakerlifecycle-{existing_stage}-{existing_status}"
            if existing == alias:
                continue
            try:
                mlflow_client.delete_registered_model_alias(registered_model_name, existing)
            except Exception:
                pass  # alias not set on this model - nothing to remove
    mlflow_client.set_registered_model_alias(registered_model_name, alias, version)
    print(f"MLflow alias '{alias}' set on {registered_model_name} v{version}")
    return alias


def wait_for_lifecycle(sm_client, model_package_arn, stage, status, timeout=90, interval=3):
    """Wait until the synced Model Package version shows ``stage``/``status``."""
    import time

    deadline = time.time() + timeout
    while True:
        lifecycle = (
            sm_client.describe_model_package(ModelPackageName=model_package_arn)
            .get("ModelLifeCycle") or {}
        )
        if lifecycle.get("Stage") == stage and lifecycle.get("StageStatus") == status:
            print(f"Model Registry shows ModelLifeCycle: {lifecycle}")
            return lifecycle
        if time.time() > deadline:
            raise TimeoutError(
                f"ModelLifeCycle did not reach {stage}/{status} within {timeout}s: {lifecycle}"
            )
        time.sleep(interval)


def approve_for_deployment(sm_client, model_package_arn, approved_by, note=None):
    """Governance officer: set ``ModelApprovalStatus=Approved`` on the version.

    Approval status is a separate attribute from the lifecycle stage - the stage says
    where the model is in its life, the approval status says it may be deployed (and
    is what enables one-click deployment in Studio, and what the model card
    automation reacts to). Unlike the alias route this is a direct API call, so the
    calling role's policy is what allows or denies it.

    The reviewer and the time are recorded on the version too, so the decision is
    auditable next to the model and not only in CloudTrail."""
    approved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = note or f"Approved for deployment by {approved_by}"
    # Carry existing metadata through, so recording the approval cannot drop entries.
    metadata = dict(
        sm_client.describe_model_package(ModelPackageName=model_package_arn)
        .get("CustomerMetadataProperties", {})
    )
    metadata["ApprovedBy"] = _clean_metadata_value(approved_by, 256)
    metadata["ApprovedAt"] = approved_at
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=note[:1024],
        CustomerMetadataProperties=metadata,
    )
    return lifecycle_summary(sm_client, model_package_arn)


def freeze_model_package_group(sm_client, model_package_arn):
    """Tag the Model Package **Group** ``frozen=true``.

    Paired with ``freeze_policy.json`` on the MLflow app service role (and on the
    data-scientist role), this locks an approved model against further changes
    through MLflow. Tag conditions can take a few minutes to propagate."""
    tags = tag_model_package_group(sm_client, model_package_arn, {"frozen": "true"})
    print(f"Tagged frozen=true on {group_arn_of(model_package_arn)}")
    return tags
