# S3 backup and restore

These scripts implement the [S3 backup guide](../support/backup-S3.md) for this
repository. They sync `results/`, `datasets/`, `data/`, `runs/`, and
`dgp-2/results/` to matching paths under `s3://epic-ace-leis-jobs/optimal-GRR/`
(the prefix defaults to the local project directory name). Neither direction
uses `--delete`; files absent at the source remain at the destination. Existing
files may be overwritten by AWS sync when the source differs.

## Credentials

The AWS CLI must be installed. Copy the [credential template](../.env.example)
and fill in all three lowercase fields with temporary klam credentials:

```bash
cp .env.example .env
chmod 600 .env
```

The scripts source this file as Bash, so use only a trusted credential file.
They clear inherited credentials and reload it on each invocation. Refresh it
when AWS reports `ExpiredToken`. Never commit `.env`.

## Workflow

Run from the repository root, or invoke the scripts by absolute path:

```bash
bash scripts/backup-S3.sh --dry-run
bash scripts/backup-S3.sh
bash scripts/restore-S3.sh --dry-run
bash scripts/restore-S3.sh
```

Dry runs still require valid credentials and S3 read access, but upload/download
nothing and do not create local output directories. Restore at job start and
back up at checkpoints or before the job ends; no automatic schedule is installed.

Both scripts accept `S3_BACKUP_SOURCE` (local project root), `S3_BACKUP_BUCKET`,
`S3_BACKUP_PREFIX`, `S3_BACKUP_ENV_FILE`, and `AWS_DEFAULT_REGION`. The default
region is `us-west-2`; keep that region for the supplied bucket.

Output directories are now ignored by Git, including nested `results/`
directories. Previously tracked outputs remain tracked; this setup does not
remove them from Git or rewrite history.
