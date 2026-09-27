"""Fresh-checkout behavior and input binding for the public release."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from langchain_core.documents import Document

from utopia.data.rag import RAG, RAGConfig
from utopia.experiments.common import load_config, prepare_args, simulation_flags
from utopia.models.request_audit import RequestAuditFailure, request_audit_scope
from utopia.utils.paths import project_root

ROOT = project_root()
FAMILIES = ('exploration', 'scale_expansion', 'influx_factorial', 'funding_feedback',
            'switching_propensity', 'resource_size', 'funding_cutoff', 'project_cost')


@pytest.mark.parametrize('family', FAMILIES)
def test_public_plans_resolve_every_case_without_creating_outputs(family, tmp_path):
    config = load_config(family)
    config['model'] = 'Qwen/Qwen3-8B'
    outputs = tmp_path / 'outputs'
    identities = []
    for cell in config['cases']:
        args = prepare_args(config, cell, 'http://localhost:19000/v1', outputs, tmp_path / 'cache')
        assert args.model == config['model'] and args.seed == config['seed']
        assert args.vllm_url == 'http://localhost:19000/v1'
        assert not hasattr(args, 'visual_dir')
        identities.append(args.experiment_id)
    assert len(identities) == len(set(identities))
    assert not outputs.exists()
    assert not (tmp_path / 'cache').exists()


def test_unknown_case_and_mismatched_config_fail_before_output_creation(tmp_path):
    config = load_config('funding_feedback')
    with pytest.raises(ValueError, match='Unknown case'):
        simulation_flags(config, 'unknown')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Expected a scale_expansion'):
        load_config('scale_expansion', path)


def test_public_source_has_no_visualization_dependencies_or_assets():
    assert not (ROOT / 'utopia/visual').exists()
    forbidden = {'matplotlib', 'seaborn', 'plotly', 'pacmap'}
    for path in (ROOT / 'utopia').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = []
            assert not any(name.split('.')[0] in forbidden or name.startswith('utopia.visual') for name in names), path
    for directory in ('utopia', 'configs', 'tests', 'docs'):
        assert not any(p.suffix.lower() in {'.png', '.svg', '.pdf', '.jpg', '.ipynb'}
                       for p in (ROOT / directory).rglob('*'))


class Rows(list):
    def filter(self, function, **kwargs):
        return Rows(row for row in self if function(row))


def source_rows():
    return Rows({'id': f'https://arxiv.org/abs/{year}.{index}', 'title': f'Paper {year} {index}',
                 'summary': f'An abstract about learning and algorithms {year} {index}.',
                 'published': f'{year}-01-01', 'tags': ['cs.AI']}
                for year in (2016, 2017) for index in range(2))


def test_document_cache_separates_year_ranges_and_dataset_revisions(tmp_path):
    with patch('utopia.data.rag.load_dataset', side_effect=lambda *a, **kw: source_rows()) as loader:
        config = RAGConfig(cache_dir=str(tmp_path), n_workers=1, device='cpu')
        first = RAG(config, start_year=2016)
        first.load_documents(num_years=1)
        again = RAG(config, start_year=2016)
        again.load_documents(num_years=1)
        assert loader.call_count == 1
        assert first.document_identity == again.document_identity
        second = RAG(config, start_year=2017)
        second.load_documents(num_years=1)
        assert loader.call_count == 2
        assert {doc.metadata['publication_year'] for doc in second.documents} == {2017}
        assert first.document_identity != second.document_identity
        other = RAG(RAGConfig(cache_dir=str(tmp_path), n_workers=1, device='cpu', dataset_revision='a' * 40))
        other.load_documents(num_years=1)
        assert loader.call_count == 3
        assert loader.call_args.kwargs['revision'] == 'a' * 40


class Encoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return torch.tensor([[float(sum(text.encode()) % 97), 1.0] for text in texts])


def test_embedding_cache_binds_order_content_and_model_revision(tmp_path):
    encoder = Encoder()
    docs = [Document(page_content=text, metadata={'id': text}) for text in ('first', 'second')]
    config = RAGConfig(cache_dir=str(tmp_path), device='cpu')
    def build(documents, current_config=config):
        rag = RAG(current_config, model=encoder)
        rag.documents = documents
        rag.build_knowledge_index()
        return rag.document_embeddings
    original = build(docs)
    assert torch.equal(original, build(docs)) and encoder.calls == 1
    reversed_embeddings = build(list(reversed(docs)))
    assert torch.equal(reversed_embeddings, original.flip(0)) and encoder.calls == 2
    build([Document(page_content='changed', metadata={'id': 'first'}), docs[1]])
    assert encoder.calls == 3
    build(docs, RAGConfig(cache_dir=str(tmp_path), device='cpu', embedding_revision='b' * 40))
    assert encoder.calls == 4


def test_qwen8_request_audit_uses_its_actual_tokenizer_revision(tmp_path):
    tokenizer = SimpleNamespace(chat_template='fixture')
    from utopia.models.request_audit import RequestAudit
    audit = RequestAudit(tmp_path / 'audit.jsonl', model_name='Qwen/Qwen3-8B', _tokenizer=tokenizer)
    try:
        assert audit.identity['model'] == 'Qwen/Qwen3-8B'
        assert audit.identity['model_revision'] == 'b968826d9c46dd6066d109eabc6255188de91218'
    finally:
        audit.close(False)
    with pytest.raises(RequestAuditFailure, match='immutable'):
        with request_audit_scope(tmp_path / 'unknown.jsonl', model_name='unknown/model', _tokenizer=tokenizer):
            pass


def test_all_cells_launch_pristine_processes_without_parent_inference(monkeypatch, tmp_path):
    from utopia.experiments import common
    calls = []
    monkeypatch.setattr(common.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(common, 'execute', lambda *a, **kw: pytest.fail('parent must not run inference'))
    common.main_for('exploration', ['--all-cells', '--model', 'Qwen/Qwen3-8B',
                                  '--output-root', str(tmp_path / 'outputs')])
    assert len(calls) == len(load_config('exploration')['cases'])
    assert {cmd[cmd.index('--cell') + 1] for cmd in calls} == set(load_config('exploration')['cases'])
    assert all('--all-cells' not in cmd and cmd[3] == 'utopia.experiments.exploration' for cmd in calls)
    assert not (tmp_path / 'outputs').exists()
