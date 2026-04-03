from __future__ import annotations

from pathlib import Path

try:
    import boto3
    import urllib3
    from urllib3.exceptions import InsecureRequestWarning
except ModuleNotFoundError:
    boto3 = None
    urllib3 = None  # type: ignore[assignment]
    InsecureRequestWarning = None  # type: ignore[assignment,misc]


def upload_model_to_s3(
    run_id: str,
    output_root: str,
    bucket: str,
    s3_prefix: str = "models",
    s3_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    verify_ssl: bool = True,
) -> str:
    """Upload model artifacts to S3 and return s3:// URI."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 model publishing.")

    model_dir = Path(output_root) / run_id / "models"
    if not verify_ssl and urllib3 is not None:
        urllib3.disable_warnings(InsecureRequestWarning)
    client = boto3.client(
        "s3",
        region_name=s3_region,
        endpoint_url=s3_endpoint,
        verify=verify_ssl,
    )

    s3_model_prefix = f"{s3_prefix.strip('/')}/{run_id}"
    files = ["stage1.pkl", "stage2.pkl", "stage3.pkl", "metrics_snapshot.json"]

    for fname in files:
        src = model_dir / fname
        if not src.exists():
            if fname == "metrics_snapshot.json":
                continue
            raise FileNotFoundError(f"Model artifact not found: {src}")
        key = f"{s3_model_prefix}/{fname}"
        print(f"[pipeline] Upload model s3://{bucket}/{key}", flush=True)
        client.upload_file(str(src), bucket, key)

    s3_uri = f"s3://{bucket}/{s3_model_prefix}"
    print(f"[pipeline] Model published to {s3_uri}", flush=True)
    return s3_uri
