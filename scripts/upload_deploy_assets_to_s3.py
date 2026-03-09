#!/usr/bin/env python3
"""Upload config, model artifacts, and optional feature store data to S3 for EC2 deployment.

Uses the same credentials as AWS CLI (env vars or ~/.aws/credentials).
Run from the project root after training so artifacts/ and optionally feature_repo/data/ exist.

Usage:
  pip install -e ".[deploy]"
  python scripts/upload_deploy_assets_to_s3.py --bucket YOUR_BUCKET [--prefix investor-ml] [--feature-store]

  Or set env and run:
  set AWS_S3_BUCKET=your-bucket
  set AWS_S3_PREFIX=investor-ml
  python scripts/upload_deploy_assets_to_s3.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload config, artifacts (models), and optional feature store data to S3 for deployment."
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name (or set AWS_S3_BUCKET)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="S3 key prefix, e.g. investor-ml (default: investor-ml; or set AWS_S3_PREFIX)",
    )
    parser.add_argument(
        "--feature-store",
        action="store_true",
        help="Also upload feature_repo/data/ (for /predict with deal_ids)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    args = parser.parse_args()

    bucket = args.bucket or __import__("os").environ.get("AWS_S3_BUCKET")
    prefix = (args.prefix or __import__("os").environ.get("AWS_S3_PREFIX", "investor-ml")).strip("/")

    if not bucket:
        print("Error: provide --bucket or set AWS_S3_BUCKET", file=sys.stderr)
        return 1

    try:
        import boto3
    except ImportError:
        print("Error: install with  pip install -e \".[deploy]\"  (adds boto3)", file=sys.stderr)
        return 1

    s3 = boto3.client("s3")
    base = f"s3://{bucket}/{prefix}"

    def upload_file(local_path: Path, s3_key: str) -> None:
        if not local_path.exists():
            print(f"  Skip (missing): {local_path}")
            return
        key = f"{prefix}/{s3_key}".lstrip("/")
        if args.dry_run:
            print(f"  Would upload: {local_path} -> s3://{bucket}/{key}")
        else:
            s3.upload_file(str(local_path), bucket, key)
            print(f"  Uploaded: {local_path.name} -> s3://{bucket}/{key}")

    def upload_dir(local_dir: Path, s3_subdir: str, exclude: tuple[str, ...] = ()) -> None:
        if not local_dir.is_dir():
            print(f"  Skip (not a dir): {local_dir}")
            return
        for f in local_dir.iterdir():
            if f.name.startswith(".") or f.suffix.lower() in exclude:
                continue
            key = f"{s3_subdir}/{f.name}".lstrip("/")
            if f.is_file():
                if args.dry_run:
                    print(f"  Would upload: {f} -> s3://{bucket}/{prefix}/{key}")
                else:
                    s3.upload_file(str(f), bucket, f"{prefix}/{key}")
                    print(f"  Uploaded: {f.name} -> {key}")
            else:
                upload_dir(f, key, exclude)

    print(f"Bucket: {bucket}, prefix: {prefix}")
    if args.dry_run:
        print("(dry run)\n")

    # 1. Config
    config_file = PROJECT_ROOT / "config" / "config.yaml"
    print("\nConfig:")
    upload_file(config_file, "config/config.yaml")

    # 2. Artifacts (model and related)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    print("\nArtifacts:")
    if artifacts_dir.is_dir():
        for f in artifacts_dir.iterdir():
            if f.is_file() and f.suffix == ".joblib":
                upload_file(f, f"artifacts/{f.name}")
        if args.dry_run and not list(artifacts_dir.glob("*.joblib")):
            print("  No .joblib files in artifacts/")
    else:
        print("  Skip (no artifacts/ dir)")

    # 3. Optional: feature store data
    if args.feature_store:
        feat_data = PROJECT_ROOT / "feature_repo" / "data"
        print("\nFeature store data (feature_repo/data/):")
        if feat_data.is_dir():
            for f in feat_data.iterdir():
                if f.is_file():
                    upload_file(f, f"feature_repo_data/{f.name}")
        else:
            print("  Skip (no feature_repo/data/ dir)")

    if not args.dry_run:
        print("\n--- GitHub Actions secrets (use these values) ---")
        print(f"  CONFIG_S3_URI       = {base}/config/")
        print(f"  ARTIFACTS_S3_URI   = {base}/artifacts/")
        if args.feature_store:
            print(f"  FEATURE_REPO_DATA_S3_URI = {base}/feature_repo_data/")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
