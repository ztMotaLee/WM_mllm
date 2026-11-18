"""Synthetic prepare/compose/export/evaluator handoff; no model inference or downloads.

Set COMPBENCH_EVAL_ALL to the pinned upstream eval_all.py to also check its result layout.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from eval.compbench import prepare_index as prepare, run_eval
from eval.compbench.compose import main as compose_main
from eval.compbench.export import export_results
from test_compbench_index import make_dataset


def upstream_task_configs(evaluator, data_root, model_root, model):
    # Execute the actual pure path-mapping function without heavyweight imports.
    tree = ast.parse(evaluator.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_all_task_configs')
    namespace = {'os': __import__('os')}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(evaluator), 'exec'), namespace)
    return namespace['get_all_task_configs'](str(data_root), str(model_root), model)


def test_prepare_compose_export_evaluation_preflight_preserves_eval_key(tmp_path, monkeypatch, capsys):
    dataset = make_dataset(tmp_path / 'dataset', key_local='nested/scene.png')
    for path in dataset.rglob('*.png'):
        Image.fromarray(np.full((96, 96, 3), 20, np.uint8)).save(path)
    monkeypatch.setattr(prepare, 'METADATA_SHA256', {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
                                                     for name in prepare.METADATA_SHA256})
    monkeypatch.setattr(prepare, 'EXPECTED_COUNTS', {task: 1 for task in prepare.METADATA})
    index = tmp_path / 'index.jsonl'
    assert prepare.main(['--root', str(dataset), '--output', str(index)]) == 0
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    proposals = tmp_path / 'proposals'
    proposals.mkdir()
    for number, row in enumerate(rows):
        proposal = np.full((96, 96, 3), 20, np.uint8)
        proposal[35:55, 35:55] = (80 + number, 150, 220)
        Image.fromarray(proposal).save(proposals / f"{row['row_id']}.png")
    composed = tmp_path / 'composed'
    assert compose_main(['--inputs', str(index), '--input-root', str(dataset),
                         '--proposal-root', str(proposals), '--out', str(composed)]) == 0
    results = tmp_path / 'results'
    receipt = export_results(composed, results, 'sample')
    assert receipt['rows'] == 6
    for item, original in zip(receipt['items'], rows):
        assert item['eval_key'] == original['eval_key']
        assert (results / item['result']).read_bytes() == (composed / item['final']).read_bytes()
    if os.environ.get('COMPBENCH_EVAL_ALL'):
        configs = {item['subtask']: item for item in upstream_task_configs(
            Path(os.environ['COMPBENCH_EVAL_ALL']), dataset / 'tasks', results / 'sample', 'sample')}
        for item, original in zip(receipt['items'], rows):
            assert results / item['result'] == Path(configs[original['subtask']]['res_dir']) / original['eval_key']
    stub = tmp_path / 'eval_all.py'
    stub.write_text('raise RuntimeError("must not execute")\n')
    monkeypatch.setattr(run_eval, 'EVALUATOR_SHA256', hashlib.sha256(stub.read_bytes()).hexdigest())
    capsys.readouterr()
    arguments = ['sample', '--evaluator', str(stub), '--check-inputs', '--index', str(index),
                 '--data-root', str(dataset / 'tasks'), '--results-root', str(results),
                 '--output-dir', str(tmp_path / 'scores')]
    assert run_eval.main(arguments + ['--tasks', 'local', 'multi_object', 'implicit']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['items'] == 6 and result['metric_inference'] is False
    assert {item['row_id'] for item in result['mapping']} == {row['row_id'] for row in rows}
    assert not (tmp_path / 'scores').exists()
    with pytest.raises(FileExistsError):
        export_results(composed, results, 'sample')
    (results / receipt['items'][0]['result']).unlink()
    with pytest.raises(ValueError, match='unavailable'):
        run_eval.main(arguments)


def test_evaluation_export_rejects_duplicate_mapping_and_corrupt_pixels(tmp_path):
    root = tmp_path / 'composed'
    (root / 'final').mkdir(parents=True)
    image = root / 'final/row.png'
    Image.new('RGB', (2, 2)).save(image)
    item = {'row_id': 'row', 'subtask': 'add', 'eval_key': 'original.png', 'final': 'final/row.png',
            'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    manifest = root / 'RUN_MANIFEST.json'
    manifest.write_text(json.dumps({'schema': 'wmmllm.compbench.composition.v1', 'rows': 2, 'items': [item, item]}))
    with pytest.raises(ValueError, match='duplicate'):
        export_results(root, tmp_path / 'results', 'sample')
    manifest.write_text(json.dumps({'schema': 'wmmllm.compbench.composition.v1', 'rows': 1, 'items': [item]}))
    Image.new('RGB', (2, 2), 'red').save(image)
    with pytest.raises(ValueError, match='hash mismatch'):
        export_results(root, tmp_path / 'results', 'sample')
    assert not (tmp_path / 'results').exists()
