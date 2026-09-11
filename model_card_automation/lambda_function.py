"""EventBridge -> Lambda: complete the Model Card of a newly registered Model Package version.

Triggered by the ``SageMaker Model Package State Change`` event that SageMaker emits
whenever a Model Package version is created or updated (approval status, metadata...).

Design, following the Model Cards / Model Registry unification guidance
(https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-details.html):

* **One card per model package version.** The card is the ``ModelCard`` block *on the
  Model Package* (``UpdateModelPackage(ModelCard=...)``), not a standalone
  ``CreateModelCard`` resource. Every version registered in the group therefore gets
  its own card, and the card travels with the version in the Studio Model Registry UI.
* **Content is written once, status follows the approval workflow.** A model card
  version is immutable except for its status; every content change creates a new card
  version. So this function enriches the content only when it has not done so yet
  (marker in ``additional_information.custom_details``) and afterwards only reconciles
  ``ModelCardStatus`` with ``ModelApprovalStatus``:

      (no approval status)     -> Draft
      PendingManualApproval    -> PendingReview
      Approved                 -> Approved
      Rejected                 -> Archived

* **Idempotent and loop-safe.** SageMaker may deliver several events per state change,
  register.py updates the package right after the MLflow auto-sync created it, and this
  function's own ``UpdateModelPackage`` emits yet another event. Every invocation
  re-derives the desired card from ``DescribeModelPackage`` and exits without a write
  when nothing needs to change.

Configuration (environment variables, all optional):

  MODEL_CARD_CREATOR           -> model_overview.model_creator
  MODEL_CARD_PURPOSE           -> intended_uses.purpose_of_model
  MODEL_CARD_INTENDED_USES     -> intended_uses.intended_uses
  MODEL_CARD_RISK_RATING       -> intended_uses.risk_rating (High|Medium|Low|Unknown)
  MODEL_CARD_BUSINESS_PROBLEM  -> business_details.business_problem
  MODEL_CARD_STAKEHOLDERS      -> business_details.business_stakeholders
  MODEL_CARD_LINE_OF_BUSINESS  -> business_details.line_of_business
"""
import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AUTOMATION_MARKER = "ModelCardAutomation"
AUTOMATION_VERSION = "1"

APPROVAL_TO_CARD_STATUS = {
    None: "Draft",
    "": "Draft",
    "PendingManualApproval": "PendingReview",
    "Approved": "Approved",
    "Rejected": "Archived",
}

# Card statuses whose content must not be modified any more.
FROZEN_CARD_STATUSES = {"Approved", "Archived"}

_sm_client = None


def _sm():
    global _sm_client
    if _sm_client is None:
        _sm_client = boto3.client("sagemaker")
    return _sm_client


def _env(name, default):
    value = os.environ.get(name, "").strip()
    return value or default


# --------------------------------------------------------------------------- content
def _evaluation_details(customer_metadata):
    """Turn the ``EvaluationMetrics`` JSON written by pipeline_steps/register.py into
    a model card ``evaluation_details`` entry. Returns [] when nothing usable exists."""
    raw = customer_metadata.get("EvaluationMetrics")
    if not raw:
        return []
    try:
        metrics = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("EvaluationMetrics is not valid JSON: %s", raw)
        return []

    metric_data = []
    for name, value in sorted(metrics.items()):
        try:
            metric_data.append({"name": name, "type": "number", "value": float(value)})
        except (TypeError, ValueError):
            metric_data.append({"name": name, "type": "string", "value": str(value)[:63]})
    if not metric_data:
        return []

    return [{
        "name": "test-set-evaluation",
        "evaluation_observation": "Metrics computed by the pipeline evaluate step on the held-out test set.",
        "metadata": {"MlflowRunId": customer_metadata.get("MlflowRunId", "")} if customer_metadata.get("MlflowRunId") else {},
        "metric_groups": [{"name": "classification_metrics", "metric_data": metric_data}],
    }]


def build_card_content(described, existing_content):
    """Merge governance details into the (possibly auto-generated) card content.

    Only fields allowed by the *model package* model card schema are used — note that
    ``model_overview`` there is limited to ``model_creator`` and ``model_artifact``.
    Training details already filled in by the MLflow auto-sync are kept as they are.
    """
    content = json.loads(json.dumps(existing_content))  # deep copy
    customer_metadata = described.get("CustomerMetadataProperties", {}) or {}
    containers = (described.get("InferenceSpecification") or {}).get("Containers") or []

    artifacts = []
    for container in containers:
        s3_source = (container.get("ModelDataSource") or {}).get("S3DataSource") or {}
        uri = s3_source.get("S3Uri") or container.get("ModelDataUrl")
        if uri and uri not in artifacts:
            artifacts.append(uri)
    if customer_metadata.get("MlflowArtifactLocation") and customer_metadata["MlflowArtifactLocation"] not in artifacts:
        artifacts.append(customer_metadata["MlflowArtifactLocation"])

    overview = content.setdefault("model_overview", {})
    overview.setdefault("model_creator", _env("MODEL_CARD_CREATOR", "from-idea-to-prod workshop"))
    if artifacts:
        overview["model_artifact"] = artifacts[:15]

    intended = content.setdefault("intended_uses", {})
    intended.setdefault("purpose_of_model", _env(
        "MODEL_CARD_PURPOSE",
        "Predict whether a bank customer will subscribe to a term deposit after a direct marketing call.",
    ))
    intended.setdefault("intended_uses", _env(
        "MODEL_CARD_INTENDED_USES",
        "Rank customers for outbound marketing campaigns. Not intended for credit or eligibility decisions.",
    ))
    intended.setdefault("factors_affecting_model_efficiency",
                        "Data drift in customer demographics or campaign timing; see the Evidently monitoring pipeline.")
    intended.setdefault("risk_rating", _env("MODEL_CARD_RISK_RATING", "Low"))
    intended.setdefault("explanations_for_risk_rating",
                        "Marketing prioritization only; a wrong prediction costs one call, not a customer outcome.")

    business = content.setdefault("business_details", {})
    business.setdefault("business_problem", _env(
        "MODEL_CARD_BUSINESS_PROBLEM",
        "Increase the conversion rate of term-deposit marketing campaigns while reducing call volume.",
    ))
    business.setdefault("business_stakeholders", _env("MODEL_CARD_STAKEHOLDERS", "Retail banking marketing team"))
    business.setdefault("line_of_business", _env("MODEL_CARD_LINE_OF_BUSINESS", "Retail banking"))

    training = content.setdefault("training_details", {})
    job_details = training.setdefault("training_job_details", {})
    if customer_metadata.get("ContainerImage") and customer_metadata["ContainerImage"] != "n/a":
        job_details.setdefault("training_environment", {}).setdefault(
            "container_image", [customer_metadata["ContainerImage"]]
        )
    if customer_metadata.get("TrainingJobName") and customer_metadata["TrainingJobName"] != "local":
        arn = described["ModelPackageArn"]
        # arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version>
        prefix = arn.rsplit(":", 1)[0]
        job_details.setdefault("training_arn", f"{prefix}:training-job/{customer_metadata['TrainingJobName']}")
    if not training.get("objective_function"):
        training["objective_function"] = {
            "function": {"function": "Maximize", "facet": "AUC", "condition": "validation set"},
            "notes": "XGBoost binary:logistic, early stopping on validation AUC.",
        }

    if not content.get("evaluation_details"):
        evaluation = _evaluation_details(customer_metadata)
        if evaluation:
            content["evaluation_details"] = evaluation

    additional = content.setdefault("additional_information", {})
    additional.setdefault("caveats_and_recommendations",
                          "Re-train when the monitoring pipeline reports drift on critical features.")
    custom = additional.setdefault("custom_details", {})
    custom[AUTOMATION_MARKER] = AUTOMATION_VERSION
    custom["ModelPackageGroupName"] = described["ModelPackageGroupName"]
    custom["ModelPackageVersion"] = str(described["ModelPackageVersion"])
    custom["ModelPackageArn"] = described["ModelPackageArn"]
    for key in ("MlflowRunId", "MlflowModelId", "TrainingJobName", "ContainerImage"):
        if customer_metadata.get(key):
            custom[key] = customer_metadata[key][:1024]

    return content


# --------------------------------------------------------------------------- reconcile
def _desired_status(described):
    return APPROVAL_TO_CARD_STATUS.get(described.get("ModelApprovalStatus"), "Draft")


def _update(model_package_arn, content, status):
    _sm().update_model_package(
        ModelPackageName=model_package_arn,
        ModelCard={"ModelCardContent": json.dumps(content), "ModelCardStatus": status},
    )


def reconcile_model_card(model_package_arn):
    """Bring the model package's card in line with the package. Returns an action string."""
    described = _sm().describe_model_package(ModelPackageName=model_package_arn)

    if described.get("ModelPackageStatus") not in (None, "Completed"):
        return f"skipped: ModelPackageStatus={described.get('ModelPackageStatus')}"

    card = described.get("ModelCard") or {}
    current_status = card.get("ModelCardStatus")
    current_content = json.loads(card["ModelCardContent"]) if card.get("ModelCardContent") else {}
    desired_status = _desired_status(described)

    already_enriched = (
        current_content.get("additional_information", {})
        .get("custom_details", {})
        .get(AUTOMATION_MARKER) == AUTOMATION_VERSION
    )

    if already_enriched or current_status in FROZEN_CARD_STATUSES:
        if current_status == desired_status:
            return "no-op: card already up to date"
        if current_status == "Archived":
            return "no-op: card is archived"
        _update(model_package_arn, current_content, desired_status)
        return f"status: {current_status} -> {desired_status}"

    content = build_card_content(described, current_content)
    if desired_status == "Approved":
        # Content and approval cannot be combined: approved cards are frozen.
        _update(model_package_arn, content, "PendingReview")
        _update(model_package_arn, content, "Approved")
        return "content written, status PendingReview -> Approved"
    _update(model_package_arn, content, desired_status)
    return f"content written, status {current_status or '(none)'} -> {desired_status}"


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
    action = reconcile_model_card(model_package_arn)
    logger.info("%s: %s", model_package_arn, action)
    return {"status": "ok", "model_package_arn": model_package_arn, "action": action}
