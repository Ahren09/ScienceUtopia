"""Funding feedback protocols and the shared full-text runner.

Fresh public configurations use common.main_for. Full-text worlds use an explicit
portable protocol; historical frozen worlds continue with their original source.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback
from utopia.funding.feedback import (CELLS, Mechanisms, effective_args, fallback_audit,
                                    simulation_class)
from utopia.runtime.provenance import ROOT, digest, write_json
from utopia.utils.data_utils import file_sha256 as file_hash

LARGE_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
LARGE_REVISION = "ea5b4f81096f3901c91dea97f81324302495781d"


def prospective_source_fingerprint(root=None):
    root = Path(root or ROOT)
    paths = sorted((root / 'utopia').rglob('*.py')) + [root / 'requirements.txt']
    return digest({str(p.relative_to(root)): file_hash(p) for p in paths})


def prospective_arguments(command):
    """Shared explicit options for new full-text, resumable experiments only."""
    command.add_argument("--cell", required=True)
    command.add_argument("--seed", type=int, choices=[42], default=42)
    command.add_argument("--founders", type=int, default=10000)
    command.add_argument("--years", type=int, default=6)
    command.add_argument("--run-id", required=True)
    command.add_argument("--vllm-url", required=True)
    command.add_argument("--server-manifest", required=True)
    command.add_argument("--fulltext-index", required=True)
    command.add_argument("--deadline-epoch", type=float, required=True)
    command.add_argument("--s3-prefix", help="Optional private S3 backup prefix; server snapshots are always saved")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--request-timeout", type=float, default=1800.0,
                         help="HTTP timeout in seconds, at most 1800 within the original shutdown grace")
    command.add_argument("--model", default=LARGE_MODEL)
    command.add_argument("--model-revision", default=LARGE_REVISION)
    command.add_argument("--context-tokens", type=int, default=131072)
    command.add_argument("--max-tokens", type=int, default=8192)
    command.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                         help="Explicit thinking mode for a new prospective study; default preserves legacy requests")
    command.add_argument("--resume", action="store_true")



def configure_prospective(options):
    from utopia.arguments import parse_arguments
    from utopia.config import SIMULATION_CONFIG
    if os.environ.get("PYTHONHASHSEED") != str(options.seed):
        raise ValueError(f"Start Python with PYTHONHASHSEED={options.seed}")
    if not 0 < getattr(options, "request_timeout", 600.0) <= 1800:
        raise ValueError("request-timeout must be positive and at most 1800 seconds")
    if options.founders <= 0 or options.founders % 5:
        raise ValueError("Founder count must be a positive multiple of five")
    if options.years <= 0 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in options.run_id + options.cell):
        raise ValueError("Invalid horizon or run/cell identifier")
    exp_id = f"{options.run_id}_{options.cell}"
    with tempfile.TemporaryDirectory(prefix="scienceutopia-args-") as temporary:
        args = parse_arguments([
            "--experiment_name", "default_prospective", "--experiment_stage", "mechanism",
            "--output-dir", temporary, "--seed", str(options.seed),
            "--population_mode", "university_only", "--num_institutions", str(options.founders // 5),
            "--researchers_per_institution", "5", "--strategy_mix", "balanced",
            "--num_years", str(options.years), "--num_conferences", "6",
            "--model", options.model, "--vllm_url", options.vllm_url,
            "--batch_size", str(options.batch_size), "--funding_panel_max_apps", "25",
            "--funding_budget_mode", "track", "--log_funding_applications",
        ])
    args.experiment_id = exp_id
    args.expected_founders = options.founders
    args.embedding_model_revision = "17e1f347d17fe144873b1201da91788898c639cd"
    args.portable_checkpoint = True
    args.always_rerun = False
    args.deadline_epoch = options.deadline_epoch
    funding_cell = getattr(options, "funding_cell", options.cell if options.cell in CELLS else "P1F1")
    args.funding_feedback = asdict(Mechanisms.from_cell(funding_cell))
    docs_root = ROOT / "outputs/docs" / exp_id
    args.docs_dir = str(docs_root / "metrics")
    args.log_dir = str(ROOT / "outputs/logs" / exp_id)
    args.checkpoint_dir = str(ROOT / "outputs/checkpoints" / exp_id)
    args.output_dir = args.checkpoint_dir
    args.rag_device = "cpu"
    args.data_cache_dir = str(ROOT / "data/cache" / exp_id)
    args.checkpoint_replay_paths = {}  # Native simulator declares exact replay outputs.
    checkpoint_root = Path(args.checkpoint_dir)
    if not options.resume and ((docs_root.exists() and any(docs_root.iterdir()))
                               or (checkpoint_root.exists() and any(checkpoint_root.iterdir()))):
        raise FileExistsError("Existing run requires explicit --resume")
    if options.resume and not any(checkpoint_root.glob("portable_checkpoint_year_*.json.gz")):
        raise FileNotFoundError("Resume requires a portable checkpoint from this new protocol")
    for path in (args.docs_dir, args.log_dir, args.checkpoint_dir, args.data_cache_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    # Disabled W&B transport still leaves the project logger enabled. Portable
    # replay requires disabling that logger before the scientific binding is built.
    SIMULATION_CONFIG["wandb"]["enabled"] = False
    return args



def run_prospective(options, simulation_factory=None, configure=None, run_context=None):
    """One cell through the native simulator, with pinned data and portable replies."""
    import time
    from contextlib import nullcontext
    from utopia.data.fulltext import FullTextCorpus
    from utopia.config import SIMULATION_CONFIG
    from utopia.runtime.checkpointing import PortableStop
    args = configure_prospective(options)
    extra_binding = configure(args) if configure else {}
    corpus = FullTextCorpus(options.fulltext_index)
    os.environ["UTOPIA_FULLTEXT_INDEX"] = str(corpus.path)
    os.environ["UTOPIA_FULLTEXT_SHA256"] = corpus.sha256
    os.environ["UTOPIA_EMBEDDING_REVISION"] = args.embedding_model_revision
    os.environ["WANDB_MODE"] = "disabled"
    protocol = getattr(options, "protocol", "pf-fulltext-portable-v1")
    scientific_args = effective_args(args)
    for name in ("deadline_epoch", "checkpoint_replay_paths", "data_cache_dir", "rag_device", "batch_size"):
        scientific_args.pop(name, None)
    binding = {
        "protocol": protocol, "cell": options.cell, "seed": options.seed,
        "model": options.model, "model_revision": options.model_revision,
        "corpus_sha256": corpus.sha256,
        "science_config": {"args": scientific_args, "config": SIMULATION_CONFIG,
                           "source_hash": prospective_source_fingerprint(),
                           "context_tokens": options.context_tokens, "max_tokens": options.max_tokens,
                           "funding_selection": "one_complete_permutation_v1", "extra": extra_binding or {}},
    }
    thinking_override = getattr(options, "thinking", None)
    if thinking_override is not None:
        if type(thinking_override) is not bool:
            raise ValueError("thinking must be None or a bool")
        binding["science_config"]["thinking_override"] = thinking_override
    binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    docs_root = Path(args.docs_dir).parent
    manifest_path = docs_root / "experiment.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None:
        if previous.get("binding") != binding:
            raise ValueError("Scientific configuration or source differs from the saved experiment")
        if previous.get("status") == "complete" and previous.get("final_preserved_snapshot"):
            return previous
    source_info = {"source_hash": binding["science_config"]["source_hash"]}
    args.checkpoint_binding = binding
    sessions = docs_root / "sessions"
    sessions.mkdir(exist_ok=True)
    session = sessions / str(time.time_ns())
    session.mkdir()
    server = json.loads(Path(options.server_manifest).read_text())
    for key, expected in {"model": options.model, "model_revision": options.model_revision,
                          "endpoint": options.vllm_url}.items():
        if server.get(key) != expected:
            raise ValueError(f"Server deployment differs from requested {key}")
    if server.get('context_tokens', 0) < options.context_tokens:
        raise ValueError('Request context budget exceeds the declared server capacity')
    manifest = {"binding": binding, "status": "starting", "server": server,
                "args": dict(vars(args)), **source_info,
                "transport": {"batch_size": options.batch_size,
                              "request_timeout": getattr(options, "request_timeout", 600.0)},
                "corpus": corpus.reference(), "session": str(session),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sampling_scope": "One seed and one world per condition; no across-world uncertainty."}
    write_json(manifest_path, manifest)
    from utopia.simulation import Simulation, VLLMServerModel, FundingAgency, set_seed
    from utopia.models.request_audit import request_audit_scope
    from utopia.funding.validation import install_funding_validation
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(Path(args.log_dir) / "simulation.log"), logging.StreamHandler()])
    set_seed(args.seed)
    handle = install_funding_validation(FundingAgency, session / "funding_validation.jsonl", compact=True)
    llm = sim = audit = upload_checkpoint = None
    backup_failed = False
    s3_available = True
    try:
        with request_audit_scope(session / "llm_requests.jsonl", model_name=options.model,
                                 model_revision=options.model_revision,
                                 context_tokens=options.context_tokens, sdk_retries=0, allow_empty=options.resume) as audit:
            llm = VLLMServerModel(
                model_name=options.model, base_url=options.vllm_url, run_seed=options.seed,
                max_concurrent_requests=options.batch_size,
                response_cache_dir=str(docs_root / "responses"), checkpoint_binding=binding,
                model_revision=options.model_revision, max_tokens_override=options.max_tokens,
                sdk_max_retries=0, request_timeout=getattr(options, "request_timeout", 600.0),
                thinking_override=thinking_override)
            if options.model not in {m.id for m in llm.client.models.list().data}:
                raise RuntimeError("Endpoint does not advertise the pinned model")
            def upload_checkpoint(checkpoint_path, final=False):
                nonlocal backup_failed, s3_available
                from utopia.runtime.snapshots import preserve_snapshot, upload_snapshot
                try:
                    metadata = preserve_snapshot(
                        checkpoint_path, llm.response_cache, docs_root, args.log_dir, ROOT,
                        {"cell": options.cell, "binding": binding, **source_info},
                        final_manifest=manifest if final else None)
                except Exception as error:
                    backup_failed = True
                    raise PortableStop(f"Server snapshot incomplete: {error}") from error
                if options.s3_prefix:
                    if s3_available:
                        metadata = upload_snapshot(metadata, options.s3_prefix.rstrip('/') + '/' + options.cell)
                        s3_available = metadata['storage_status'] == 's3_uploaded'
                    else:
                        metadata.update(storage_status='server_preserved_s3_pending')
                        write_json(Path(metadata['server_directory']) / 'snapshot.json', metadata)
                write_json(docs_root / 'latest_preserved_checkpoint.json', metadata)
                if metadata['is_offhost']:
                    write_json(docs_root / 'latest_uploaded_checkpoint.json', metadata)
                return metadata
            args.checkpoint_callback = upload_checkpoint
            with (run_context() if run_context else nullcontext()):
                cls = (simulation_factory or simulation_class)(Simulation)
                sim = cls(llm=llm, args=args, mechanisms=Mechanisms(**args.funding_feedback),
                          num_years=args.num_years, output_dir=args.output_dir,
                          **{key: getattr(args, key) for key in (
                              "verbose", "debug", "industry_funding_mode", "funding_allocation_mode",
                              "always_rerun", "use_langchain", "max_retries", "weighted_funding_assignment",
                              "experiment_name", "collaboration_mode", "collaboration_network_growth_mode", "start_year")})
                manifest["status"] = "running"
                write_json(manifest_path, manifest)
                sim.run()
            if len(sim.founder_ids) != options.founders or set(sim.funding_feedback_years) != set(range(1, options.years + 1)):
                raise RuntimeError("The declared founder count/full horizon was not completed")
            if not handle.summary()["completion_allowed"]:
                raise RuntimeError("Funding validation did not permit completion")
        manifest.update(status="complete", years_completed=options.years,
                        initial_world_hash=sim.initial_world_hash, corpus_hash=sim.corpus_hash,
                        fallback_audit=fallback_audit(sim))
        manifest["request_audit"] = audit.summary()
        manifest["funding_validation"] = handle.summary()
        pending_upload = dict(manifest, status="completed_pending_upload")
        write_json(session / "result.json", pending_upload)
        write_json(manifest_path, pending_upload)
        # The final native report is produced after the last annual checkpoint.
        receipt = upload_checkpoint(Path(args.checkpoint_dir) / f"portable_checkpoint_year_{options.years}.json.gz", final=True)
        manifest['final_preserved_snapshot'] = receipt['server_directory']
        manifest['storage_status'] = receipt['storage_status']
        if receipt['is_offhost']:
            manifest['final_uploaded_snapshot'] = receipt['destination']
    except PortableStop as error:
        manifest.update(status="paused", reason=str(error),
                        years_completed=getattr(sim, "_portable_last_year", 0))
        raise
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc(),
                        years_completed=getattr(sim, "_portable_last_year", 0))
        raise
    finally:
        handle.restore()
        manifest["funding_validation"] = handle.summary()
        if audit is not None:
            manifest["request_audit"] = audit.summary()
        write_json(session / "result.json", manifest)
        write_json(manifest_path, manifest)
        if llm is not None:
            try:
                if manifest.get("status") != "complete" and upload_checkpoint is not None and not backup_failed:
                    checkpoints = list(Path(args.checkpoint_dir).glob("portable_checkpoint_year_*.json.gz"))
                    if checkpoints:
                        latest = max(checkpoints, key=lambda q: int(q.name.removeprefix("portable_checkpoint_year_").removesuffix(".json.gz")))
                        try:
                            upload_checkpoint(latest)
                        except PortableStop as error:
                            manifest["backup_error"] = str(error)
            finally:
                llm.response_cache.close()
        write_json(session / "result.json", manifest)
        write_json(manifest_path, manifest)
    return manifest



def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['run-fulltext']:
        parser = argparse.ArgumentParser(description=__doc__)
        prospective_arguments(parser)
        options = parser.parse_args(argv[1:])
        if options.cell not in CELLS:
            parser.error('P/F cell must be P0F0, P0F1, P1F0 or P1F1')
        return run_prospective(options)
    from utopia.experiments.common import main_for
    return main_for('funding_feedback', argv)


if __name__ == '__main__':
    main()
