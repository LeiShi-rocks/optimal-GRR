# Pluto Job Backup & Restore Guide (S3 version)

A reusable setup for persisting experiment outputs across ephemeral Pluto jobs using the `epic-ace-leis-jobs` S3 bucket as permanent storage and GitHub for code.

---

## Problem

Pluto job workspaces (`/mnt/localssd/`) are **ephemeral** — everything is lost when the job ends. Large outputs (results, datasets, model weights) cannot go on GitHub. This guide sets up a two-command backup/restore pipeline so you can pick up exactly where you left off in any new job.

---

## Storage architecture

| Layer | Location | Purpose |
|-------|----------|---------|
| **Code** | GitHub repo | Versioned source code |
| **Scratch** | `/mnt/localssd/<project>/` | Active working directory (ephemeral) |
| **Persistent outputs** | `s3://epic-ace-leis-jobs/<project>/` | Results, datasets, large files |

---

## One-time setup (per project)

### 1. Clone the repo onto localssd

```bash
cd /mnt/localssd
git clone <your-repo-url> <project>
cd <project>
```

### 2. Save AWS credentials in `.env`

Unlike sensei-fs (filesystem permissions), S3 needs explicit AWS credentials.
Save the temporary credentials from klam in the project-root `.env` file using
these lowercase variable names:

```bash
aws_access_key_id=...
aws_secret_access_key=...
aws_session_token=...
```

Credentials come from klam.corp.adobe.com — see
[`S3-bucket/create-bucket.md`](../S3-bucket/create-bucket.md) for how to get
them and [`S3-bucket/use-bucket.md`](../S3-bucket/use-bucket.md) for basic
manual usage. The S3 scripts clear inherited AWS credentials, read this file,
and export the uppercase AWS variables each time they run. They're temporary —
see "Notes on AWS credential expiry" below.

### 3. Add a `scripts/` directory with backup and restore scripts

Create `scripts/backup-S3.sh`:

```bash
#!/usr/bin/env bash
# Usage: bash scripts/backup-S3.sh [--dry-run]
set -euo pipefail

DRY_RUN=()
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=(--dryrun) && echo "[backup-S3] Dry-run mode."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${S3_BACKUP_SOURCE:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUCKET="${S3_BACKUP_BUCKET:-epic-ace-leis-jobs}"
PREFIX="${S3_BACKUP_PREFIX:-$(basename "${SRC}")}"
DST="s3://${BUCKET}/${PREFIX}"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"
ENV_FILE="${S3_BACKUP_ENV_FILE:-${SRC}/.env}"

# Add/remove directories here as needed
DIRS=(results datasets data)

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[backup-S3] Credential file not found: ${ENV_FILE}" >&2
    exit 1
fi

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
set -a
source "${ENV_FILE}"
set +a
export AWS_ACCESS_KEY_ID="${aws_access_key_id}"
export AWS_SECRET_ACCESS_KEY="${aws_secret_access_key}"
export AWS_SESSION_TOKEN="${aws_session_token}"
export AWS_DEFAULT_REGION="us-west-2"
REGION="${AWS_DEFAULT_REGION}"

echo "[backup-S3] Source : ${SRC}"
echo "[backup-S3] Dest   : ${DST}"
echo ""

for dir in "${DIRS[@]}"; do
    if [[ -d "${SRC}/${dir}" ]]; then
        echo "[backup-S3] Syncing ${dir}/ ..."
        aws s3 sync "${DRY_RUN[@]}" "${SRC}/${dir}/" "${DST}/${dir}/" --region "${REGION}"
        echo "[backup-S3] Done: ${dir}/"
    else
        echo "[backup-S3] Skipping ${dir}/ (not found locally)"
    fi
done

echo ""
echo "[backup-S3] Backup complete."
```

Create `scripts/restore-S3.sh`:

```bash
#!/usr/bin/env bash
# Usage: bash scripts/restore-S3.sh [--dry-run]
set -euo pipefail

DRY_RUN=()
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=(--dryrun) && echo "[restore-S3] Dry-run mode."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${S3_BACKUP_SOURCE:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUCKET="${S3_BACKUP_BUCKET:-epic-ace-leis-jobs}"
PREFIX="${S3_BACKUP_PREFIX:-$(basename "${DST}")}"
SRC="s3://${BUCKET}/${PREFIX}"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"
ENV_FILE="${S3_BACKUP_ENV_FILE:-${DST}/.env}"

DIRS=(results datasets data)

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[restore-S3] Credential file not found: ${ENV_FILE}" >&2
    exit 1
fi

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
set -a
source "${ENV_FILE}"
set +a
export AWS_ACCESS_KEY_ID="${aws_access_key_id}"
export AWS_SECRET_ACCESS_KEY="${aws_secret_access_key}"
export AWS_SESSION_TOKEN="${aws_session_token}"
export AWS_DEFAULT_REGION="us-west-2"
REGION="${AWS_DEFAULT_REGION}"

echo "[restore-S3] Source : ${SRC}"
echo "[restore-S3] Dest   : ${DST}"
echo ""

for dir in "${DIRS[@]}"; do
    echo "[restore-S3] Restoring ${dir}/ ..."
    mkdir -p "${DST}/${dir}"
    aws s3 sync "${DRY_RUN[@]}" "${SRC}/${dir}/" "${DST}/${dir}/" --region "${REGION}"
    echo "[restore-S3] Done: ${dir}/"
done

echo ""
echo "[restore-S3] Restore complete."
```

Unlike sensei-fs, there's no need for a "does the backup path exist" check —
`aws s3 sync` from a prefix with no objects is a safe no-op, it just copies
nothing.

Make them executable:

```bash
chmod +x scripts/backup-S3.sh scripts/restore-S3.sh
```

### 4. Update `.gitignore`

Add the following to prevent large outputs (and credentials) from being committed:

```gitignore
# Virtual environments
venv/
terra-env/
long-cate/

# Ephemeral outputs (backed up to S3 via scripts/backup-S3.sh)
results/
datasets/
data/
runs/

# Local workspace snapshots
old-version/
submitted-version/

# AWS credentials — never commit
.env
```

---

## Job workflow

### Start of every job

```bash
cd /mnt/localssd/<project>
git pull                  # get latest code from GitHub
bash scripts/restore-S3.sh   # pull results/datasets/data from S3
```

### During the job

Work normally. Code changes are committed to GitHub. Outputs go into `results/`, `datasets/`, or `data/` — all excluded from git, all backed up to S3.

### End of job (or any checkpoint)

```bash
bash scripts/backup-S3.sh
git add -p && git commit -m "..." && git push   # code only
```

---

## Overrides

Both scripts respect environment variables so you can redirect targets without editing the scripts:

| Variable | Purpose | Example |
|----------|---------|---------|
| `S3_BACKUP_BUCKET` | Override the destination bucket | `export S3_BACKUP_BUCKET=my-other-bucket` |
| `S3_BACKUP_PREFIX` | Override the key prefix (defaults to the source directory name) | `export S3_BACKUP_PREFIX=my-other-path` |
| `S3_BACKUP_ENV_FILE` | Override the credential-file location | `export S3_BACKUP_ENV_FILE=/path/to/.env` |
| `AWS_DEFAULT_REGION` | Must be `us-west-2` for this bucket, not the account default | `export AWS_DEFAULT_REGION=us-west-2` |

Use `--dry-run` to preview what would be synced without writing anything:

```bash
bash scripts/backup-S3.sh --dry-run
bash scripts/restore-S3.sh --dry-run
```

---

## Notes on AWS credential expiry

Unlike sensei-fs (filesystem uid/gid permissions, never expires), S3 access
here depends on temporary klam-vended STS credentials that typically last
around an hour. If `backup-S3.sh` or `restore-S3.sh` start failing with
`ExpiredToken` or similar auth errors:

1. Get fresh credentials from klam.corp.adobe.com (GET CLI).
2. Update the `.env` file referenced in step 2 of setup.
3. Retry `backup-S3.sh`; it reloads `.env` on every invocation.

There's currently no auto-refreshing IAM role wired up for
`epic-ace-leis-jobs` — that would require a Pluto Resource Role attachment,
which isn't set up yet for this bucket (see
[`../S3-bucket/create-bucket.md`](../S3-bucket/create-bucket.md) and
[`aip-cli-usage.md`](aip-cli-usage.md) for status). For long-running jobs,
budget for a mid-job credential refresh until that's in place.

Also note: this bucket lives in `us-west-2`, while the AWS account's default
region is `us-east-2` — always pass `--region us-west-2` (or export
`AWS_DEFAULT_REGION=us-west-2`), or you'll hit `PermanentRedirect` errors.

---

## Reference

- [`../S3-bucket/create-bucket.md`](../S3-bucket/create-bucket.md) — how the bucket was created, and how to get klam credentials
- [`../S3-bucket/use-bucket.md`](../S3-bucket/use-bucket.md) — basic manual `aws s3 cp`/`sync` examples against this bucket
