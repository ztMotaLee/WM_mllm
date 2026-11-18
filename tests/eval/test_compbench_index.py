"""CompBench index construction from isolated synthetic metadata; no downloads."""
from __future__ import annotations

import hashlib
import json

import pytest

from eval.compbench import prepare_index as prepare
from eval.compbench.export import contained_file, relative_reference


def make_dataset(root, key_local='scene/0/test.png'):
    for subtask, metadata in prepare.METADATA.items():
        path = root / metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        if subtask in {'add', 'remove', 'replace'}:
            key = key_local
            path.write_text(json.dumps([{'image_path': key, 'instruction': 'synthetic test instruction'}]))
            dirs = [f'tasks/{subtask}/{kind}' for kind in ('input_image', 'edited_image', 'mask')]
        else:
            key = 'test.png'
            field = 'instruction' if subtask == 'implicit' else 'add_ins' if subtask == 'multi_object_add' else 'remove_ins'
            path.write_text(json.dumps({'image_name': key, field: 'synthetic test instruction'}) + '\n')
            prefix = 'tasks/implicit_reasoning' if subtask == 'implicit' else f'tasks/multi_turn_editing/{subtask}'
            dirs = [f'{prefix}/{kind}' for kind in ('ori_images', 'edited_images', 'masks')]
        for directory in dirs:
            image = root / directory / key
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'synthetic test bytes')
    (root / 'tasks/implicit_reasoning/implicit_data.jsonl').write_text('{}\n')
    return root


@pytest.mark.parametrize('value', ['/absolute', '../escape', 'a/../b', './a', 'a//b', 'a\\b', 'C:/a',
                                   'https://example.org/a', '', 'a\x00b', 'a?b', 'a#b', 'a\nb'])
def test_relative_paths_reject_invalid_values(value):
    with pytest.raises(ValueError):
        relative_reference(value)


def test_index_maps_all_six_task_layouts(tmp_path):
    root = make_dataset(tmp_path)
    rows = prepare.build_index(root, expected_counts={key: 1 for key in prepare.METADATA})
    assert len({row['row_id'] for row in rows}) == 6
    for row in rows:
        assert row['instruction'] == 'synthetic test instruction' and 'direct' not in row
        assert row['item_id'] == f"{row['subtask']}/{row['eval_key']}"
        assert row['row_id'] == hashlib.sha256(row['item_id'].encode()).hexdigest()[:24]
        for key in ('source', 'target', 'mask'):
            assert contained_file(root, row[key]).is_file()


def test_index_rejects_wrong_population_missing_reference_duplicates_and_symlinks(tmp_path):
    root = make_dataset(tmp_path / 'data')
    counts = {key: 1 for key in prepare.METADATA}
    with pytest.raises(ValueError, match='population'):
        prepare.build_index(root)
    path = root / 'tasks/add/metadata.json'
    record = json.loads(path.read_text())[0]
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match='duplicate item'):
        prepare.build_index(root, expected_counts=counts)
    path.write_text(json.dumps([record]))
    (root / 'tasks/add/mask/scene/0/test.png').unlink()
    with pytest.raises(ValueError, match='unavailable'):
        prepare.build_index(root, expected_counts=counts)
    (tmp_path / 'linked').symlink_to(root / 'tasks', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        contained_file(tmp_path, 'linked/add/metadata.json')


def test_main_checks_metadata_hashes_before_indexing(tmp_path, monkeypatch, capsys):
    root = make_dataset(tmp_path / 'data')
    monkeypatch.setattr(prepare, 'EXPECTED_COUNTS', {key: 1 for key in prepare.METADATA})
    assert prepare.main(['--root', str(root), '--output', str(tmp_path / 'index.jsonl')]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'FAIL'
    monkeypatch.setattr(prepare, 'METADATA_SHA256', {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                                     for name in prepare.METADATA_SHA256})
    assert prepare.main(['--root', str(root), '--output', str(tmp_path / 'index.jsonl')]) == 0
    assert json.loads(capsys.readouterr().out)['items'] == 6
    assert len((tmp_path / 'index.jsonl').read_text().splitlines()) == 6
