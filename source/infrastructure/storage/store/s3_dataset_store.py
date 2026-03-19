from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from source.application.ports import DatasetStorePort
from source.domain.entities import DatasetArtifacts, DatasetVersion

try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError:
    boto3 = None
    ClientError = Exception


# Сохраняет артефакты датасета в S3-совместимое хранилище.
class S3DatasetStore(DatasetStorePort):

    def __init__(
        self,
        bucket: str = "",
        region: str = "us-east-1",
        endpoint_url: str = "",
    ) -> None:
        self._bucket = bucket.strip()
        self._region = region.strip() or "us-east-1"
        self._endpoint_url = endpoint_url.strip() or None
        self._client = None

    def save(
        self, dataset_version: DatasetVersion, artifacts: DatasetArtifacts
    ) -> DatasetArtifacts:

        bucket, base_prefix = self._resolve_prefix(dataset_version.s3_prefix)
        version_prefix = self._join_key(base_prefix, dataset_version.version_id)
        print(
            f"[prepare] Публикация в S3: bucket={bucket}, prefix={version_prefix}",
            flush=True,
        )

        books_uri = self._upload(
            artifacts.books_uri, bucket, self._join_key(version_prefix, "books.parquet")
        )
        train_uri = self._upload(
            artifacts.train_uri, bucket, self._join_key(version_prefix, "train.parquet")
        )
        test_uri = self._upload(
            artifacts.test_uri, bucket, self._join_key(version_prefix, "test.parquet")
        )
        local_train_uri = self._upload(
            artifacts.local_train_uri,
            bucket,
            self._join_key(version_prefix, "local_train.parquet"),
        )
        local_val_uri = self._upload(
            artifacts.local_val_uri,
            bucket,
            self._join_key(version_prefix, "local_val.parquet"),
        )
        local_val_warm_uri = self._upload(
            artifacts.local_val_warm_uri,
            bucket,
            self._join_key(version_prefix, "local_val_warm.parquet"),
        )
        local_val_cold_uri = self._upload(
            artifacts.local_val_cold_uri,
            bucket,
            self._join_key(version_prefix, "local_val_cold.parquet"),
        )
        summary_uri = self._upload(
            artifacts.summary_uri,
            bucket,
            self._join_key(version_prefix, "summary.json"),
        )
        manifest_uri = self._upload(
            artifacts.manifest_uri,
            bucket,
            self._join_key(version_prefix, "manifest.json"),
        )

        return DatasetArtifacts(
            books_uri=books_uri,
            train_uri=train_uri,
            test_uri=test_uri,
            local_train_uri=local_train_uri,
            local_val_uri=local_val_uri,
            local_val_warm_uri=local_val_warm_uri,
            local_val_cold_uri=local_val_cold_uri,
            summary_uri=summary_uri,
            manifest_uri=manifest_uri,
        )

    def exists(self, dataset_version: DatasetVersion) -> bool:

        bucket, base_prefix = self._resolve_prefix(dataset_version.s3_prefix)
        manifest_key = self._join_key(
            base_prefix, dataset_version.version_id, "manifest.json"
        )

        client = self._s3()
        try:
            client.head_object(Bucket=bucket, Key=manifest_key)
            return True
        except ClientError:
            return False

    def _s3(self):

        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for S3 backend. "
                "Install dependency: pip install boto3"
            )
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
            )
        return self._client

    def _upload(self, local_path: str, bucket: str, key: str) -> str:

        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {src}")
        print(f"[prepare] Upload -> s3://{bucket}/{key}", flush=True)
        self._s3().upload_file(str(src), bucket, key)
        print(f"[prepare] Upload завершен: s3://{bucket}/{key}", flush=True)
        return f"s3://{bucket}/{key}"

    def _resolve_prefix(self, s3_prefix: str) -> tuple[str, str]:

        raw = s3_prefix.strip()
        if raw.startswith("s3://"):
            parsed = urlparse(raw)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            if not bucket:
                raise ValueError(f"Invalid s3 prefix: {s3_prefix}")
            return bucket, prefix
        if not self._bucket:
            raise ValueError(
                "S3 bucket is required (either in s3_prefix or --s3-bucket)."
            )
        return self._bucket, raw.lstrip("/")

    @staticmethod
    def _join_key(*parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part and part.strip("/"))
