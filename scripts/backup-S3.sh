#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=()
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=(--dryrun); echo "[backup-S3] Dry-run mode." ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 1 ;;
esac
[[ $# -le 1 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${S3_BACKUP_SOURCE:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUCKET="${S3_BACKUP_BUCKET:-epic-ace-leis-jobs}"
PREFIX="${S3_BACKUP_PREFIX:-$(basename "${SRC}")}"
DST="s3://${BUCKET}/${PREFIX}"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"
ENV_FILE="${S3_BACKUP_ENV_FILE:-${SRC}/.env}"
DIRS=(results data)

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[backup-S3] Credential file not found: ${ENV_FILE}" >&2
    exit 1
fi
command -v aws >/dev/null || { echo "[backup-S3] AWS CLI is required." >&2; exit 1; }

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
unset aws_access_key_id aws_secret_access_key aws_session_token
set -a
source "${ENV_FILE}"
set +a
export AWS_ACCESS_KEY_ID="${aws_access_key_id:?Set aws_access_key_id in the credential file}"
export AWS_SECRET_ACCESS_KEY="${aws_secret_access_key:?Set aws_secret_access_key in the credential file}"
export AWS_SESSION_TOKEN="${aws_session_token:?Set aws_session_token in the credential file}"
export AWS_DEFAULT_REGION="${REGION}"

echo "[backup-S3] Source : ${SRC}"
echo "[backup-S3] Dest   : ${DST}"

for dir in "${DIRS[@]}"; do
    if [[ -d "${SRC}/${dir}" ]]; then
        echo "[backup-S3] Syncing ${dir}/ ..."
        aws s3 sync "${DRY_RUN[@]}" "${SRC}/${dir}/" "${DST}/${dir}/" --region "${REGION}"
        echo "[backup-S3] Done: ${dir}/"
    else
        echo "[backup-S3] Skipping ${dir}/ (not found locally)"
    fi
done

echo "[backup-S3] Backup complete."
