"""Save complete year-boundary evidence before attempting optional S3 publication."""
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from utopia.utils.data_utils import file_sha256, write_json_atomic


def preserve_snapshot(checkpoint, cache, docs, logs, source_root, metadata, *, final_manifest=None):
    checkpoint, docs, source_root = map(Path, (checkpoint, docs, source_root))
    year = int(checkpoint.name.removeprefix("portable_checkpoint_year_").removesuffix(".json.gz"))
    backup = docs / "backup" / f"year_{year}" / f"snapshot_{time.time_ns()}"
    backup.mkdir(parents=True)
    cache.backup(backup / "responses.sqlite")
    shutil.copy2(checkpoint, backup / checkpoint.name)
    shutil.copytree(docs, backup / "results",
                    ignore=shutil.ignore_patterns("responses", "backup", "*.sqlite*"))
    shutil.copytree(logs, backup / "logs")
    shutil.copytree(checkpoint.parent, backup / "outputs",
                    ignore=shutil.ignore_patterns("portable_checkpoint_*"))
    if final_manifest is not None:
        write_json_atomic(backup / "results/experiment.json", final_manifest, indent=2)
    sources = sorted((source_root / "utopia").rglob("*.py")) + [source_root / "requirements.txt"]
    source_manifest = {str(p.relative_to(source_root)): file_sha256(p) for p in sources}
    write_json_atomic(backup / "source_manifest.json", source_manifest, indent=2)
    with tarfile.open(backup / "source.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(source_root)), recursive=False)
    metadata = dict(
        metadata, year_completed=year, checkpoint=checkpoint.name,
        checkpoint_sha256=file_sha256(checkpoint),
        response_snapshot_sha256=file_sha256(backup / "responses.sqlite"),
        completed_responses=cache.committed_count(), server_directory=str(backup),
        snapshot_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        includes_final_report=final_manifest is not None,
        storage_status="server_preserved", is_offhost=False,
        independent_readback_verified=False,
        preserved_files={str(p.relative_to(backup)): {
            "bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in sorted(backup.rglob("*")) if p.is_file()},
    )
    write_json_atomic(backup / "snapshot.json", metadata, indent=2)
    write_json_atomic(docs / "latest_preserved_checkpoint.json", metadata, indent=2)
    return metadata


def upload_snapshot(metadata, prefix):
    """One upload attempt; failure leaves the complete server snapshot available."""
    metadata = dict(metadata)
    backup = Path(metadata["server_directory"])
    destination = prefix.rstrip("/") + f"/checkpoints/year_{metadata['year_completed']}/{backup.name}"
    metadata.update(destination=destination, storage_status="server_preserved_s3_pending")
    try:
        subprocess.run(["aws", "s3", "sync", str(backup) + "/", destination + "/",
                        "--exclude", "snapshot.json", "--only-show-errors"],
                       capture_output=True, text=True, check=True)
        metadata.update(storage_status="s3_uploaded", is_offhost=True)
        write_json_atomic(backup / "snapshot.json", metadata, indent=2)
        subprocess.run(["aws", "s3", "cp", str(backup / "snapshot.json"),
                        destination + "/snapshot.json", "--only-show-errors"],
                       capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        metadata.update(storage_status="server_preserved_s3_pending", is_offhost=False,
                        s3_error=getattr(error, "stderr", None) or str(error))
    write_json_atomic(backup / "snapshot.json", metadata, indent=2)
    return metadata
