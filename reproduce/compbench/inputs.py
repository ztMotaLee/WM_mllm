"""Explicit inputs for the historical composer. No checkout discovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image

from .primitives import authorised_region_from_grounding, mean_rgb_abs_diff

FORBIDDEN_PATH_COMPONENTS = frozenset({'edited_image', 'edited_images', 'ground_truth', 'gt'})


def read_rows(path: Path) -> list[dict]:
    """Read either the prepared JSONL index or an explicit JSON array."""
    text = path.read_text(encoding='utf-8')
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('inputs must contain a nonempty JSON array or JSONL records')
    return rows


def input_path(root: Path, reference: str) -> Path:
    """Resolve a manifest-relative input and refuse target-image inputs."""
    if not isinstance(reference, str) or not reference or '\\' in reference or ':' in reference:
        raise ValueError('input references must be relative POSIX paths')
    path = PurePosixPath(reference)
    if path.is_absolute() or any(p in ('', '.', '..') for p in reference.split('/')):
        raise ValueError('input references must stay within input_root')
    if set(path.parts) & FORBIDDEN_PATH_COMPONENTS or path.name.endswith('_detailed.csv'):
        raise PermissionError('target images and metric labels cannot enter composition')
    root = root.resolve(strict=True)
    candidate = (root / reference).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError('input reference must be a contained file')
    return candidate


def load_rgb(path: Path, size=None) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert('RGB')
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


class RowInputs:
    """Load source/proposal and optional diagnostics from an explicit manifest.

    Diagnostic masks are optional. Their absence is labeled in every receipt.
    Only source and direct pixels determine the installed support and image.
    """

    def __init__(self, meta: dict, *, input_root: Path, proposal_root: Path | None = None):
        self.rid = meta['row_id']
        if not isinstance(self.rid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', self.rid):
            raise ValueError('row_id must be a simple file-safe identifier')
        self.meta = dict(meta)
        self.split = meta.get('split', 'external')
        self.operation = meta.get('subtask', meta.get('operation'))
        self.instruction = meta.get('instruction', '')
        self.source_path = meta['source']
        self.direct_path = meta['direct']
        source = input_path(input_root, self.source_path)
        direct = input_path(input_root if proposal_root is None else proposal_root, self.direct_path)
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.direct_sha256 = hashlib.sha256(direct.read_bytes()).hexdigest()
        self.source = load_rgb(source)
        self.size = (self.source.shape[1], self.source.shape[0])
        self.direct = load_rgb(direct, self.size)
        self.diff = mean_rgb_abs_diff(self.source, self.direct)
        self.diagnostics_present = []

        def mask(field):
            reference = meta.get(field)
            if not reference:
                return np.zeros(self.source.shape[:2], dtype=bool)
            with Image.open(input_path(input_root, reference)) as opened:
                if opened.size != self.size:
                    raise ValueError(f'{field} dimensions must match source')
                result = np.asarray(opened.convert('L')) > 127
            self.diagnostics_present.append(field)
            return result

        self.protected = mask('protected')
        self.base_target = mask('base_target')
        self.declared_box = mask('declared_box')
        if meta.get('grounding'):
            if meta.get('declared_box'):
                raise ValueError('provide grounding or declared_box, not both')
            grounding = json.loads(input_path(input_root, meta['grounding']).read_text())
            self.declared_box = authorised_region_from_grounding(grounding, self.size)
            self.diagnostics_present.append('grounding')
        self.v11_committed = bool(meta.get('v11_committed', False))
