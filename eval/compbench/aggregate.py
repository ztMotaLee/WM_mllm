"""Aggregate saved per-image CompBench metrics using the paper's arithmetic.

No metric model is imported. Archive CSV members are read in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

ARMS = {'direct': 'direct_w48', 'paste': 'thresh_paste_all3', 'ours': 'aiwm_v14_w48'}
GROUPS = {'Local': ['add', 'remove', 'replace'],
          'Multi-object': ['multi_object_add', 'multi_object_remove'],
          'Implicit': ['implicit']}
TAG = {'add': 'add', 'remove': 'remove', 'replace': 'replace',
       'multi_object_add': 'multi_object', 'multi_object_remove': 'multi_object',
       'implicit': 'implicit'}
METRICS = ['LC-T', 'LC-I', 'PSNR', 'SSIM', 'LPIPS']
RULES = ('Independent recompute of the three-arm CompBench installation ablation from the per-image evaluator CSVs (in-memory tar read).\n'
         'Rules: non-PSNR metrics use all rows (NaN skipped as pandas mean); PSNR uses rows finite in all three arms;\n'
         'subtask means rounded to 3 with NumPy scalar semantics; group = equal-weight mean of subtask means, rounded to 3.')


def read_table(text: str) -> dict[str, dict[str, float]]:
    """Read detailed evaluator CSVs, coercing invalid metric fields to NaN."""
    reader = csv.DictReader(io.StringIO(text))
    if not {'Image_Name', *METRICS}.issubset(reader.fieldnames or []):
        raise ValueError('detailed CSV is missing required columns')
    table = {}
    for row in reader:
        name = row['Image_Name']
        if name in table:
            raise ValueError('duplicate Image_Name in detailed CSV')
        values = {}
        for metric in METRICS:
            try:
                values[metric] = float(row[metric])
            except (ValueError, TypeError):
                values[metric] = float('nan')
        table[name] = values
    return table


def aggregate_tables(tables, *, source_sha256: str | None = None):
    """Average subtasks equally after three-decimal subtask rounding.

    tables[(arm, subtask)] maps Image_Name to the five metric values.
    PSNR uses each subtask's common finite population across all three arms.
    Other metrics use every available row for their own arm, skipping NaN.
    """
    required = {(arm, subtask) for arm in ARMS for subtask in TAG}
    if not required.issubset(tables):
        raise ValueError('all three arms and six subtasks are required')
    out = {'rules': RULES, 'source_sha256': source_sha256, 'groups': {}, 'n': {}}
    for group, subtasks in GROUPS.items():
        out['groups'][group] = {}
        for arm in ARMS:
            out['groups'][group][arm] = {}
            for metric in METRICS:
                rounded = []
                for subtask in subtasks:
                    table = tables[(arm, subtask)]
                    if metric == 'PSNR':
                        common = [name for name in tables[('direct', subtask)]
                                  if all(name in tables[(a, subtask)] and
                                         np.isfinite(tables[(a, subtask)][name][metric])
                                         for a in ARMS)]
                        out['n'][f'{subtask}_PSNR_common_finite'] = len(common)
                        values = np.array([table[name][metric] for name in common], dtype=np.float64)
                    else:
                        values = np.array([row[metric] for row in table.values()], dtype=np.float64)
                    # pandas' float64 nanops mean replaces NaN by zero, sums in
                    # original row order, then divides by the non-NaN count.
                    valid = ~np.isnan(values)
                    count = int(valid.sum())
                    if not count:
                        raise ValueError(f'no valid scores for {arm}/{subtask}/{metric}')
                    mean = np.sum(np.where(valid, values, 0), dtype=np.float64) / count
                    if not np.isfinite(mean):
                        raise ValueError(f'nonfinite aggregate for {arm}/{subtask}/{metric}')
                    rounded.append(round(mean, 3))
                out['groups'][group][arm][metric] = float(round(np.mean(rounded), 3))
    for subtask in TAG:
        out['n'][subtask] = len(tables[('ours', subtask)])
    return out


def load_archive(path: Path, *, arms=None):
    """Load the paper archive layout without extracting any member to disk."""
    arm_names = dict(ARMS if arms is None else arms)
    if set(arm_names) != set(ARMS):
        raise ValueError('arm names must map direct, paste and ours')
    tables = {}
    with tarfile.open(path, 'r:*') as archive:
        members = archive.getmembers()
        for arm, model in arm_names.items():
            if not model or '/' in model or model in ('.', '..'):
                raise ValueError('arm directory must be a simple name')
            for subtask, tag in TAG.items():
                suffix = f'{model}/{tag}/{model}/{subtask}_detailed.csv'
                matches = [member for member in members
                           if member.name == suffix or member.name.endswith('/' + suffix)]
                if len(matches) != 1:
                    raise ValueError(f'expected exactly one detailed CSV for {arm}/{subtask}')
                member = matches[0]
                if not member.isfile() or member.size > 20_000_000:
                    raise ValueError('detailed CSV must be a regular bounded archive member')
                with archive.extractfile(member) as handle:
                    tables[(arm, subtask)] = read_table(handle.read().decode('utf-8-sig'))
    return tables


def aggregate_archive(path: Path, *, arms=None):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return aggregate_tables(load_archive(path, arms=arms), source_sha256=hasher.hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct-model', default=ARMS['direct'])
    parser.add_argument('--paste-model', default=ARMS['paste'])
    parser.add_argument('--ours-model', default=ARMS['ours'])
    args = parser.parse_args(argv)
    out = aggregate_archive(args.scores, arms={'direct': args.direct_model,
                                             'paste': args.paste_model, 'ours': args.ours_model})
    serialized = json.dumps(out, indent=1, allow_nan=False) + '\n'
    with args.output.open('x') as handle:
        handle.write(serialized)
    print(json.dumps({'groups': out['groups'], 'n': out['n']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
