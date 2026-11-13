# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

from __future__ import annotations
from .result import UnitResult, ItemResult

def _project_unit_scores(metric_name: str, unit: UnitResult) -> UnitResult:
    """Replace ``unit.scores`` with the canonical projection for ``metric_name``.

    Internal compute keys (debug pairs, mean/std/num counts, alternative
    similarity proxies) are preserved on ``unit.metadata['raw_scores']`` so
    debugging stays possible.
    """
    from eval.mbench._upstream.metrics.canonical import canonicalize_scores

    if unit.unit_type == "skipped_contract":
        return unit
    raw = dict(unit.scores or {})
    canonical = canonicalize_scores(metric_name, raw)
    if canonical == raw:
        return unit
    new_metadata = dict(unit.metadata or {})
    # Only stash a copy when there were extra/non-canonical fields.
    extras = {k: v for k, v in raw.items() if k not in canonical}
    if extras:
        new_metadata.setdefault("raw_scores", extras)
    unit.scores = canonical
    unit.metadata = new_metadata
    return unit



def _project_item_scores(metric_name: str, item_result: ItemResult | None) -> ItemResult | None:
    if item_result is None:
        return item_result
    from eval.mbench._upstream.metrics.canonical import canonicalize_scores

    raw = dict(item_result.scores or {})
    item_result.scores = canonicalize_scores(metric_name, raw)
    return item_result

