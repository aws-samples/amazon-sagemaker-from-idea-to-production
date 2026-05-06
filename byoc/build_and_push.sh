#!/usr/bin/env bash
#
# Build and push the BYOC XGBoost image to the caller's Amazon ECR.
#
# Usage:
#   ./build_and_push.sh [<region>] [<repo_name>] [<tag>]
#
# Defaults:
#   region    = $AWS_REGION (or us-east-1 if unset)
#   repo_name = sagemaker-xgboost-byoc
#   tag       = 1.0
#
# Requirements:
#   - Docker engine reachable (in SageMaker Studio, requires DOCKER_BUILDKIT=0
#     and --network sagemaker, both already handled below).
#   - AWS CLI credentials with ECR create/push permissions.
set -euo pipefail

REGION="${1:-${AWS_REGION:-us-east-1}}"
REPO_NAME="${2:-sagemaker-xgboost-byoc}"
TAG="${3:-1.0}"

# Resolve account ID so the full image URI can be built.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"

echo "Region:   ${REGION}"
echo "Account:  ${ACCOUNT_ID}"
echo "Image:    ${IMAGE_URI}"
echo

# Create ECR repository if it does not already exist.
if ! aws ecr describe-repositories \
        --repository-names "${REPO_NAME}" \
        --region "${REGION}" >/dev/null 2>&1; then
    echo "Creating ECR repository ${REPO_NAME} in ${REGION}..."
    aws ecr create-repository \
        --repository-name "${REPO_NAME}" \
        --region "${REGION}" >/dev/null
fi

# Authenticate Docker to this account's ECR.
echo "Logging Docker in to ${REGISTRY}..."
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"

# SageMaker Studio requires the legacy builder and the sagemaker network.
export DOCKER_BUILDKIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${IMAGE_URI}..."
docker build \
    --network sagemaker \
    -t "${IMAGE_URI}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo "Pushing ${IMAGE_URI}..."
docker push "${IMAGE_URI}"

echo
echo "Done: ${IMAGE_URI}"
