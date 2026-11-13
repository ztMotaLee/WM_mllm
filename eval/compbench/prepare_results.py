"""Export composed pixels under the original CompBench evaluation identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from reproduce.compbench.inputs import read_rows


def relative_reference(value: str) -> Path:
    if (not isinstance(value, str) or not value or any(c in value for c in '\\:?#')
            or any(ord(c) < 32 for c in value)):
        raise ValueError('invalid relative image reference')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('image reference must stay inside its root')
    return Path(*path.parts)


def contained_file(root: Path, reference: str) -> Path:
    relative = relative_reference(reference)
    root = root.resolve(strict=True)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('image references cannot use symlinks')
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('referenced image is unavailable')
    return path


def result_relative(model: str, subtask: str, eval_key: str) -> Path:
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', model):
        raise ValueError('model must be a simple result-directory name')
    key = relative_reference(eval_key)
    if subtask in ('add', 'remove', 'replace'):
        task = Path(f'{model}_local') / subtask
    elif subtask in ('multi_object_add', 'multi_object_remove'):
        task = Path(f'{model}_multi') / subtask
    elif subtask == 'implicit':
        task = Path(f'{model}_implicit')
    else:
        raise ValueError('unsupported CompBench subtask')
    return Path(model) / task / key


def export_results(composed: Path, results_root: Path, model: str) -> dict:
    manifest = json.loads(contained_file(composed, 'RUN_MANIFEST.json').read_text())
    items = manifest.get('items')
    if (manifest.get('schema') != 'wmmllm.compbench.composition.v1'
            or not isinstance(items, list) or not items or manifest.get('rows') != len(items)):
        raise ValueError('composition manifest has no complete evaluation mapping')
    plan = []
    seen_ids, seen_paths = set(), set()
    for item in items:
        row_id = item['row_id']
        if not isinstance(row_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', row_id):
            raise ValueError('invalid row_id')
        if item['final'] != f'final/{row_id}.png':
            raise ValueError('composition final does not match row_id')
        source = contained_file(composed, item['final'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('composition final hash mismatch')
        relative = result_relative(model, item['subtask'], item['eval_key'])
        if row_id in seen_ids or relative in seen_paths:
            raise ValueError('duplicate row_id or evaluation destination')
        seen_ids.add(row_id)
        seen_paths.add(relative)
        plan.append((source, relative, item))
    results_root.mkdir(parents=True, exist_ok=True)
    results_root = results_root.resolve(strict=True)
    model_root = results_root / model
    model_root.mkdir(exist_ok=False)
    try:
        exported = []
        for source, relative, item in plan:
            destination = results_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
            if hashlib.sha256(destination.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('exported result hash mismatch')
            exported.append({**item, 'result': relative.as_posix()})
        receipt = {'schema': 'wmmllm.compbench.results.v1', 'model': model,
                   'rows': len(exported), 'items': exported}
        (model_root / 'RESULTS_MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(model_root)
        raise
    return receipt


def check_inputs(index: Path, data_root: Path, results_root: Path, model: str, tasks: list[str]) -> dict:
    """Validate the prepared population's paths before loading metric models."""
    rows = read_rows(index)
    checked = []
    destinations = set()
    for row in rows:
        subtask = row['subtask']
        family = 'local' if subtask in ('add', 'remove', 'replace') else 'multi' if subtask.startswith('multi_object_') else 'implicit'
        if 'all' not in tasks and not any(token.lower() in subtask or token.lower() == family for token in tasks):
            continue
        destination = result_relative(model, subtask, row['eval_key'])
        if destination in destinations:
            raise ValueError('duplicate evaluation destination')
        destinations.add(destination)
        for field in ('source', 'target', 'mask'):
            reference = row[field]
            if not isinstance(reference, str) or not reference.startswith('tasks/'):
                raise ValueError('dataset index paths must start with tasks/')
            contained_file(data_root, reference.removeprefix('tasks/'))
        contained_file(results_root, destination.as_posix())
        checked.append({'row_id': row['row_id'], 'subtask': subtask,
                        'eval_key': row['eval_key'], 'result': destination.as_posix()})
    if not checked:
        raise ValueError('no indexed items match the selected tasks')
    return {'status': 'PASS', 'items': len(checked), 'metric_inference': False, 'mapping': checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composed', required=True, type=Path)
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args(argv)
    receipt = export_results(args.composed, args.results_root, args.model)
    print(json.dumps({'status': 'PASS', 'items': receipt['rows'], 'model': args.model}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
