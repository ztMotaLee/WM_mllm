"""Valid-only recovery-table aggregation, independent of model inference.

Native item scores are already canonical. Invalid or missing values are counted,
never imputed. Item-condition population membership depends on video presence.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
MODELS = {
    "hy_worldplay": "HY-WorldPlay", "matrix_game_3": "Matrix-Game 3.0",
    "yume": "Yume-1.5", "infinite_world": "Infinite-World",
    "lingbot_world": "Lingbot-World", "matrix_game_2": "Matrix-Game 2.0",
    "aiwm": "AI-WM",
}
METRICS = {
    "mbencha.environment.spatial_epipolar": "Epipolar",
    "mbencha.environment.spatial_reprojection": "Reprojection",
    "mbencha.environment.rendering_lighting": "Lighting",
    "mbencha.environment.rendering_style": "Style",
}
ITERATIONS = 20000

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)



def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text())
        rows = value if isinstance(value, list) else next(
            (value[key] for key in ('rows', 'files', 'videos', 'items', 'exports')
             if isinstance(value.get(key), list)), None)
    if rows is None or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Expected a list of manifest objects: ' + str(path))
    return rows



def model_id(value):
    return 'aiwm' if value in {'aiwm', 'ours', 'AI-WM', 'AI-WM (ours)', 'ai-wm', 'AIWM'} else value



def identity(row, default_model=None):
    return (model_id(row.get('model_id') or row.get('model') or default_model or ''),
            row.get('item_id') or '', row.get('condition_id') or row.get('condition') or '')



def video_sha(row):
    video = row.get('video')
    value = (row.get('lfs_sha256') or row.get('expected_sha256') or row.get('sha256') or
             (video.get('sha256') if isinstance(video, dict) else None))
    return value.removeprefix('sha256:') if isinstance(value, str) else value



def video_path(row):
    video = row.get('video')
    return (row.get('video_path') or row.get('final_path') or
            (video.get('path') if isinstance(video, dict) else video) or row.get('path'))



def declared_present(row):
    status = str(row.get('status', '')).upper()
    return (row.get('missing') is not True and not status.startswith('MISSING') and
            isinstance(video_path(row), str) and bool(video_path(row)))



def percentile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction



def bootstrap(values, iterations, seed):
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iterations))
    return [percentile(means, 0.025), percentile(means, 0.975)]



def pair_list(pairs):
    return [{'item_id': item, 'condition_id': condition} for item, condition in sorted(pairs)]



def populations(expected):
    present = {model: {(k[1], k[2]) for k, row in expected.items()
                       if k[0] == model and declared_present(row)} for model in MODELS}
    common = set.intersection(*(present[model] for model in MODELS))
    return common, present



def cell(model, metric, pairs, expected, records, population, seed=45, iterations=ITERATIONS):
    valid, invalid, missing = [], [], []
    for item_id, condition in sorted(pairs):
        key = (model, item_id, condition)
        base = {'item_id': item_id, 'condition_id': condition}
        record = records.get(key)
        if record is None:
            reason = 'score_receipt_missing' if declared_present(expected[key]) else 'manifest_video_missing'
            missing.append({**base, 'reason': reason})
            continue
        result = record.get('metrics', {}).get(metric)
        if result is None:
            invalid.append({**base, 'reason': 'metric_absent_from_receipt',
                            'receipt_status': record.get('status'), 'processing_errors': record.get('processing_errors', [])})
            continue
        item = result.get('item', {})
        score = item.get('scores', {}).get('score')
        if item.get('valid') is True and finite(score):
            valid.append({**base, 'score': float(score)})
        else:
            reasons = sorted({str(unit['error']) for unit in result.get('units', []) if unit.get('error')})
            invalid.append({**base, 'reason': item.get('error') or '; '.join(reasons) or
                            ('official_item_invalid' if item.get('valid') is not True else 'canonical_score_absent_or_nonfinite'),
                            'official_item_valid': item.get('valid'),
                            'finite_canonical_score_if_invalid': float(score) if finite(score) else None,
                            'receipt_status': record.get('status')})
    values = [row['score'] for row in valid]
    stable_seed = int(hashlib.sha256(f'{seed}\0{population}\0{model}\0{metric}'.encode()).hexdigest()[:16], 16)
    mean = math.fsum(values) / len(values) if values else None
    ci = bootstrap(values, iterations, stable_seed)
    return {'model_id': model, 'model': MODELS[model], 'metric': metric, 'column': METRICS[metric],
            'population': population, 'population_count': len(pairs),
            'valid_count': len(valid), 'invalid_count': len(invalid), 'missing_count': len(missing),
            'coverage_complete': not missing, 'mean': mean, 'bootstrap_95_ci': ci,
            'mean_display100': mean * 100 if mean is not None else None,
            'bootstrap_95_ci_display100': [v * 100 for v in ci] if ci is not None else None,
            'bootstrap_seed': stable_seed, 'valid_items': valid, 'invalid_items': invalid, 'missing_items': missing}



def aggregate(expected, records, seed=45, iterations=ITERATIONS):
    common, present = populations(expected)
    return {name: [cell(model, metric, common if name == 'common' else present[model], expected,
                       records, name, seed, iterations) for model in MODELS for metric in METRICS]
            for name in ('common', 'own_full')}



def load_inputs(manifest_paths, record_paths):
    """Load explicit manifests and exact record files/directories, without media."""
    expected = {}
    for path in manifest_paths:
        for row in read_rows(path):
            key = identity(row)
            if key[0] not in MODELS or any(not isinstance(part, str) or not part for part in key):
                raise ValueError("Manifest has an unknown model or incomplete identity")
            if key in expected:
                raise ValueError("Duplicate manifest identity")
            expected[key] = row
    primary = {(key[1], key[2]) for key in expected}
    for model in MODELS:
        if {(key[1], key[2]) for key in expected if key[0] == model} != primary:
            raise ValueError("Every model needs every item-condition row; declare missing videos explicitly")
    records = {}
    cohort = None
    for path in record_paths:
        path = Path(path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            row = json.loads(file.read_text())
            if row.get("schema") != "mem6.score.v1" or row.get("family") != "environment":
                raise ValueError("Expected an environment mem6.score.v1 receipt")
            key = identity(row)
            if key not in expected or not declared_present(expected[key]):
                raise ValueError("Receipt conflicts with manifest population")
            if key in records:
                raise ValueError("Duplicate score identity; select one source explicitly")
            if video_sha(expected[key]) and video_sha(row) != video_sha(expected[key]):
                raise ValueError("Receipt video hash differs from manifest")
            for name, result in row.get("metrics", {}).items():
                item = result.get("item", {})
                if name not in METRICS or item.get("metric") != name or item.get("item_id") != key[1] or model_id(item.get("model_id")) != key[0]:
                    raise ValueError("Metric item identity differs from receipt")
            provenance = row.get("provenance", {})
            binding = (provenance.get("code", {}).get("tree_sha256"),
                       provenance.get("weights", {}).get("lock", {}).get("sha256"),
                       provenance.get("protocol", {}).get("sha256"))
            if any(value is None for value in binding):
                raise ValueError("Receipt lacks code/weights/protocol cohort binding")
            if cohort is not None and cohort != binding:
                raise ValueError("Receipts use different code/weights/protocol cohorts")
            cohort = binding
            records[key] = row
    return expected, records


def aggregate_files(manifest_paths, record_paths, output, seed=45, iterations=ITERATIONS):
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh aggregate output file")
    expected, records = load_inputs(manifest_paths, record_paths)
    cells = aggregate(expected, records, seed, iterations)
    common, present = populations(expected)
    report = {"schema": "wmmllm.mbench.environment_table.v1",
              "common_population_count": len(common),
              "populations": {"common": {"item_count": len(common), "items": pair_list(common), "cells": cells["common"]},
                              "own_full": {"models": {model: {"item_count": len(present[model]), "items": pair_list(present[model])} for model in MODELS},
                                           "cells": cells["own_full"]}},
              "bootstrap": {"iterations": iterations, "base_seed": seed},
              "population_policy": "Exact item-condition video-presence intersection; invalid and missing scores do not shrink populations.",
              "score_policy": "Require item.valid is True and a finite non-boolean canonical score; multiply by 100 once for display."}
    from .mem6_scorer.common import atomic
    atomic(output, report)
    return report
