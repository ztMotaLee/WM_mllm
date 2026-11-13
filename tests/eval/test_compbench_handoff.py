"""Synthetic prepare/compose/evaluator handoff; no model inference or downloads."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wmmllm_datasets import prepare_compbench as prepare
from reproduce.compbench.__main__ import main as compose_main
from eval.compbench.prepare_results import export_results, result_relative
from eval.compbench.run_eval import main as evaluate_main, PORTABLE_EVALUATOR


def make_dataset(root: Path):
    for subtask, metadata in prepare.METADATA.items():
        local = subtask in ('add', 'remove', 'replace')
        key = 'nested/scene.png' if local else 'scene.png'
        if local:
            record = {'image_path': key, 'instruction': 'Add a colored square.'}
            prefix = f'tasks/{subtask}'
            directories = ('input_image', 'edited_image', 'mask')
        else:
            field = 'instruction' if subtask == 'implicit' else 'add_ins' if subtask == 'multi_object_add' else 'remove_ins'
            record = {'image_name': key, field: 'Add a colored square.'}
            prefix = 'tasks/implicit_reasoning' if subtask == 'implicit' else f'tasks/multi_turn_editing/{subtask}'
            directories = ('ori_images', 'edited_images', 'masks')
        meta = root / metadata
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps([record]) if meta.suffix == '.json' else json.dumps(record) + '\n')
        for directory in directories:
            path = root / prefix / directory / key
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((96, 96, 3), 20, np.uint8)).save(path)


def upstream_task_configs(data_root, model_root, model):
    # Execute the actual pure path-mapping function without heavyweight imports.
    tree = ast.parse(PORTABLE_EVALUATOR.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_all_task_configs')
    namespace = {'os': __import__('os')}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(PORTABLE_EVALUATOR), 'exec'), namespace)
    return namespace['get_all_task_configs'](str(data_root), str(model_root), model)


def test_prepare_compose_export_evaluation_preflight_preserves_eval_key(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / 'dataset'
    make_dataset(dataset)
    checksums = [(path.relative_to(dataset).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in dataset.rglob('*') if path.is_file()]
    monkeypatch.setattr(prepare, 'read_manifest', lambda _: checksums)
    monkeypatch.setattr(prepare, 'EXPECTED_COUNTS', {task: 1 for task in prepare.METADATA})
    index = tmp_path / 'index.jsonl'
    assert prepare.main(['--root', str(dataset), '--output', str(index), '--verify']) == 0
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
    configs = {item['subtask']: item for item in upstream_task_configs(dataset / 'tasks', results / 'sample', 'sample')}
    for item, original in zip(receipt['items'], rows):
        expected = Path(configs[original['subtask']]['res_dir']) / original['eval_key']
        assert results / item['result'] == expected
        assert item['eval_key'] == original['eval_key']
        assert expected.read_bytes() == (composed / item['final']).read_bytes()
    capsys.readouterr()
    assert evaluate_main(['sample', '--check-inputs', '--index', str(index), '--data-root', str(dataset / 'tasks'),
                          '--results-root', str(results), '--output-dir', str(tmp_path / 'scores'),
                          '--tasks', 'local', 'multi_object', 'implicit']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['items'] == 6 and result['metric_inference'] is False
    assert {item['row_id'] for item in result['mapping']} == {row['row_id'] for row in rows}
    assert not (tmp_path / 'scores').exists()
    with pytest.raises(FileExistsError):
        export_results(composed, results, 'sample')
    (results / receipt['items'][0]['result']).unlink()
    with pytest.raises(ValueError, match='unavailable'):
        evaluate_main(['sample', '--check-inputs', '--index', str(index), '--data-root', str(dataset / 'tasks'),
                       '--results-root', str(results), '--output-dir', str(tmp_path / 'scores')])


@pytest.mark.parametrize('key', ['../bad.png', '/bad.png', 'a//b.png', 'a\\b.png', 'a?b.png'])
def test_evaluation_export_rejects_escaping_keys(key):
    with pytest.raises(ValueError):
        result_relative('sample', 'add', key)


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
