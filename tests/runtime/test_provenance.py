import json
from types import SimpleNamespace

import pytest

from utopia.runtime.historical import trusted_audit_sources
from utopia.runtime.provenance import write_run_manifest
from utopia.utils.data_utils import json_sha256, without_hash_metadata
from utopia.utils.paths import project_root
from pathlib import Path
import io
import threading


def test_run_manifest_preserves_start_fields_and_accumulates_resume_statistics(
    tmp_path,
):
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "command": "frozen original command",
                "timestamp_running": "original-start",
                "git_commit": "original-commit",
                "llm_call_stats": {"calls": 4, "tokens": 10},
            }
        )
    )
    args = SimpleNamespace(seed=42)
    config = {"levels": (1, 2), "strategy": "explorer"}
    write_run_manifest(
        str(tmp_path),
        "complete",
        args,
        config,
        extra={"llm_call_stats": {"calls": 2, "failures": 1}},
    )
    record = json.loads(path.read_text())
    assert record["command"] == "frozen original command"
    assert record["git_commit"] == "original-commit"
    assert record["timestamp_running"] == "original-start"
    assert record["status"] == "complete"
    assert record["args"] == {"seed": 42}
    assert record["llm_call_stats"] == {"calls": 6, "tokens": 10, "failures": 1}
    assert record["scientific_config_hash"] == json_sha256(
        config, sort_keys=True, default=str
    )
    assert not path.read_bytes().endswith(b"\n")


def test_historical_source_hashes_do_not_block_saved_results():
    golden = json.loads(
        (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
    )
    pins = golden["previous_package_sources"]
    assert trusted_audit_sources(pins, pins) == pins
    altered = dict(pins)
    altered["utopia/utils/general_utils.py"] = "0" * 64
    assert trusted_audit_sources(altered, pins) == altered
    missing = dict(pins)
    missing.pop("utopia/utils/general_utils.py")
    assert trusted_audit_sources(missing, pins) == missing
    assert trusted_audit_sources({}, pins) == {}


def test_historical_hash_metadata_is_optional_but_results_and_seed_still_match():
    evidence = {
        "request_id": 17, "score": 0.5, "seed": 42,
        "env": {"PYTHONHASHSEED": "42"},
        "steps": [{"applicant_id": "A", "request_sha256": "old"}],
        "source_binding": {"old.py": "old"},
    }
    resumed = {
        "request_id": 17, "score": 0.5, "seed": 42,
        "env": {"PYTHONHASHSEED": "42"}, "steps": [{"applicant_id": "A"}],
    }
    assert without_hash_metadata(evidence) == resumed
    assert evidence["steps"][0]["request_sha256"] == "old"
    for key, value in (("request_id", 18), ("score", 0.7), ("seed", 43)):
        assert without_hash_metadata(evidence) != {**resumed, key: value}


def test_scientific_input_comparison_keeps_settings_and_ignores_provenance():
    from utopia.runtime.switching_evidence import scientific_inputs

    original = {
        "seed": 42, "resolved_config": {"reviewers": 3},
        "source_files_sha256": {"old.py": "old"},
        "embedding_model": {"model": "encoder", "revision": "v1", "files_sha256": {"weights": "old"}},
    }
    resumed = {
        **original, "source_files_sha256": {},
        "embedding_model": {"model": "encoder", "revision": "v1"},
    }
    assert scientific_inputs(original) == scientific_inputs(resumed)
    assert scientific_inputs(original) != scientific_inputs({**resumed, "seed": 43})
    assert scientific_inputs(original) != scientific_inputs({
        **resumed, "resolved_config": {"reviewers": 2},
    })


class SnapshotS3:
    """In-memory S3 transport; snapshot ordering and file handling remain native."""
    def __init__(self):
        self.objects = {}
        self.fail_upload = False

    def upload_file(self, path, bucket, key, **kwargs):
        if self.fail_upload:
            raise OSError('Interrupted upload')
        self.objects[bucket, key] = Path(path).read_bytes()

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Bucket, Key])}

    def download_fileobj(self, bucket, key, stream):
        # boto3's managed download seeks from the object's offset, not the
        # caller's append position. Reply segments must use streaming reads.
        stream.seek(0)
        stream.write(self.objects[bucket, key])
        # Multipart writes can finish at an earlier offset than the file's end.
        stream.seek(0)


def test_stage_snapshot_reply_segments_and_failed_upload_preserve_latest(tmp_path, monkeypatch):
    import boto3
    from utopia.runtime.provenance import publish_native_snapshot, restore_native_snapshot
    s3 = SnapshotS3()
    monkeypatch.setattr(boto3, 'client', lambda *a, **k: s3)
    output = tmp_path / 'original'
    output.mkdir()
    checkpoint = output / 'checkpoint_year_3_stage_4_running.json'
    checkpoint.write_text(json.dumps({'year': 3, 'phase': 4, 'next_step': 6}))
    replies = output / 'model_replies.jsonl'
    replies.write_text('{"request_id":"paper:A"}\n')
    fulltext = tmp_path / 'fulltext'
    fulltext.mkdir()
    body = fulltext / 'original_body.utf8'
    body.write_text('Introduction\nComplete paper\nReferences\nAppendix END')
    index = fulltext / 'index.json'
    index.write_text(json.dumps({'text_file': str(body), 'papers': {
        'A': {'offset': 0, 'nbytes': body.stat().st_size}}}))
    sim = SimpleNamespace(
        output_dir=str(output), experiment_name='fixture',
        latest_checkpoint_path=str(checkpoint),
        llm=SimpleNamespace(response_write_lock=threading.Lock()),
        args=SimpleNamespace(checkpoint_s3_prefix='s3://aaron0929/population_influx/test',
                             log_dir=str(output), data_cache_dir=str(output / 'cache'),
                             fulltext_index=str(index), experiment_id='fixture'))
    publish_native_snapshot(sim)
    replies.write_text(replies.read_text() + '{"request_id":"paper:B"}\n')
    publish_native_snapshot(sim)
    latest_key = ('aaron0929', 'population_influx/test/latest.json')
    latest = s3.objects[latest_key]
    s3.fail_upload = True
    replies.write_text(replies.read_text() + '{"request_id":"paper:C"}\n')
    with pytest.raises(OSError, match='Interrupted upload'):
        publish_native_snapshot(sim)
    assert s3.objects[latest_key] == latest
    migrated = tmp_path / 'another-machine'
    restored = restore_native_snapshot('s3://aaron0929/population_influx/test', migrated)
    assert json.loads(Path(restored).read_text())['next_step'] == 6
    assert (migrated / 'model_replies.jsonl').read_text().splitlines() == [
        '{"request_id":"paper:A"}', '{"request_id":"paper:B"}']
    assert (migrated / 'code/restore_code.tar.gz').is_file()
    assert json.loads((migrated / 'fulltext/index.json').read_text())['text_file'] == 'texts.utf8'
    assert (migrated / 'fulltext/texts.utf8').read_bytes() == body.read_bytes()


def test_snapshot_parallel_files_finish_readback_before_latest(tmp_path, monkeypatch):
    import boto3
    from utopia.runtime import provenance

    s3 = SnapshotS3()
    monkeypatch.setattr(boto3, 'client', lambda *a, **k: s3)
    output = tmp_path / 'original'
    cache = output / 'cache'
    cache.mkdir(parents=True)
    checkpoint = output / 'checkpoint_year_3_stage_4_running.json'
    checkpoint.write_text('{"year":3,"phase":4,"next_step":6}')
    (output / 'model_replies.jsonl').write_text('{"request_id":"paper:A"}\n')
    assets = {'first.txt', 'second.txt'}
    for name in assets:
        (cache / name).write_text(name)
    sim = SimpleNamespace(
        output_dir=str(output), experiment_name='fixture',
        latest_checkpoint_path=str(checkpoint),
        llm=SimpleNamespace(response_write_lock=threading.Lock()),
        args=SimpleNamespace(checkpoint_s3_prefix='s3://aaron0929/population_influx/parallel',
                             log_dir=str(output), data_cache_dir=str(cache),
                             fulltext_index=None, experiment_id='fixture'))
    barrier = threading.Barrier(2)
    verified = set()
    upload = provenance.upload_verified

    def verify_file(client, path, bucket, key):
        asset = '/cache/' in key
        if asset:
            barrier.wait(timeout=3)
        if key.endswith('/latest.json'):
            assert verified == assets
        result = upload(client, path, bucket, key)
        if asset:
            verified.add(Path(path).name)
        return result

    monkeypatch.setattr(provenance, 'upload_verified', verify_file)
    provenance.publish_native_snapshot(sim)
    latest_key = ('aaron0929', 'population_influx/parallel/latest.json')
    latest = s3.objects[latest_key]
    for name in assets:
        (cache / name).write_text('changed ' + name)
    get_object = s3.get_object

    def corrupt_one_readback(Bucket, Key):
        if Key.endswith('/cache/second.txt'):
            return {'Body': io.BytesIO(b'corrupt')}
        return get_object(Bucket=Bucket, Key=Key)

    monkeypatch.setattr(s3, 'get_object', corrupt_one_readback)
    verified.clear()
    with pytest.raises(IOError, match='S3 readback differs'):
        provenance.publish_native_snapshot(sim)
    assert s3.objects[latest_key] == latest
    monkeypatch.setattr(s3, 'get_object', get_object)
    destination = tmp_path / 'another-machine'
    provenance.restore_native_snapshot('s3://aaron0929/population_influx/parallel', destination)
    assert all((destination / 'cache' / name).read_text() == name for name in assets)


def test_resume_arguments_restore_population_and_allow_runtime_changes(tmp_path):
    from utopia.arguments import parse_arguments
    source = tmp_path / 'old'
    source.mkdir()
    checkpoint = source / 'checkpoint.json'
    checkpoint.write_text(json.dumps({
        'year': 3, 'phase': 4,
        'checkpoint_args': dict(model='Qwen/Qwen3.5-397B-A17B-FP8', seed=42,
                                annual_population_influx=0.2, num_years=10,
                                max_model_len=262144, always_rerun=True,
                                experiment_name='influx_fixture', log_dir=str(source)),
    }))
    args = parse_arguments(['--resume-from', str(checkpoint), '--output-dir', str(tmp_path / 'new'),
                            '--batch_size', '1024', '--vllm_url', 'http://another-host:8000/v1'])
    assert args.annual_population_influx == 0.2
    assert args.model == 'Qwen/Qwen3.5-397B-A17B-FP8'
    assert args.batch_size == 1024 and args.num_years == 10
    assert args.max_model_len == 262144 and not args.always_rerun
    with pytest.raises(ValueError, match='scientific setting annual_population_influx'):
        parse_arguments(['--resume-from', str(checkpoint), '--annual-population-influx', '0.1'])
    data = json.loads(checkpoint.read_text())
    data['checkpoint_args']['annual_influx_rate'] = data['checkpoint_args'].pop('annual_population_influx')
    checkpoint.write_text(json.dumps(data))
    legacy = parse_arguments(['--resume-from', str(checkpoint), '--output-dir', str(tmp_path / 'legacy')])
    assert legacy.population_influx_mode == 'legacy_fixed'


def test_latest_checkpoint_ignores_unfinished_write(tmp_path):
    from utopia.runtime.provenance import latest_native_checkpoint
    annual = tmp_path / 'checkpoint_year_2.json'
    annual.write_text('{}')
    (tmp_path / 'checkpoint_year_3_stage_4_running.json.tmp').write_text('{')
    assert Path(latest_native_checkpoint(tmp_path)) == annual
    stage = tmp_path / 'checkpoint_year_3_stage_4_running.json'
    stage.write_text('{}')
    (tmp_path / 'latest_checkpoint.json').write_text(json.dumps({'path': stage.name}))
    assert Path(latest_native_checkpoint(tmp_path)) == stage
