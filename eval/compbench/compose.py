"""Historical CompBench composition of saved proposals, independent of the session engine.

The RGB change map is ``mean|source - proposal|`` in float32. Pixels strictly
above 25 form the seed; thin-safe opening and 8-connected reconstruction keep
components of at least 64 pixels. The support is dilated by 7 and feathered
with Pillow's Gaussian radius 3; Pillow composes the final RGB. Pixels where
alpha is 0 stay source-exact. Only source and proposal pixels decide output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy import ndimage

OPENING_STRUCTURE = np.ones((3, 3), dtype=bool)
EIGHT = np.ones((3, 3), dtype=bool)
LINE_LENGTH = 5
STRONG_CHANGE_THRESHOLD = 25.0
SUPPORT_CC_MIN_PX = 64
TARGET_DILATE_PX = 7
FEATHER_PX = 3
SUPPORT_CEILING_S = 0.56
FORBIDDEN_PATH_COMPONENTS = frozenset({'edited_image', 'edited_images', 'ground_truth', 'gt'})


def mean_rgb_abs_diff(source_rgb: np.ndarray, direct_rgb: np.ndarray) -> np.ndarray:
    """mean_RGB_abs(source - Direct) as float32 HxW."""
    return np.abs(
        np.asarray(source_rgb, dtype=np.float32)
        - np.asarray(direct_rgb, dtype=np.float32)
    ).mean(axis=2)


def _line_structures(length: int = LINE_LENGTH) -> list[np.ndarray]:
    horizontal = np.zeros((1, length), dtype=bool)
    horizontal[0, :] = True
    diagonal = np.eye(length, dtype=bool)
    return [horizontal, horizontal.T.copy(), diagonal,
            np.fliplr(diagonal).copy()]


LINE_STRUCTURES = _line_structures()


def thin_safe_noise_floor(strong: np.ndarray) -> np.ndarray:
    """Opening by the union of a 3x3 square and 4 length-5 line elements."""
    kept = ndimage.binary_opening(strong, structure=OPENING_STRUCTURE)
    for structure in LINE_STRUCTURES:
        kept |= ndimage.binary_opening(strong, structure=structure)
    return kept


def necessary_support(diff: np.ndarray):
    """Strong-change connected components of |Direct - source|; returns (support, labels)."""
    strong = diff > STRONG_CHANGE_THRESHOLD
    seed = thin_safe_noise_floor(strong)
    recon = ndimage.binary_propagation(seed, mask=strong, structure=EIGHT)
    labels, count = ndimage.label(recon, structure=EIGHT)
    if not count:
        return np.zeros_like(strong), labels
    sizes = ndimage.sum_labels(recon, labels, range(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = sizes >= SUPPORT_CC_MIN_PX
    return keep[labels], labels


def dilate_mask(mask: np.ndarray, radius: int) -> Image.Image:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    if radius > 0:
        image = image.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return image


def _soft(mask: np.ndarray, dilate_px: int, feather_px: int,
          half: bool) -> np.ndarray:
    core = dilate_mask(mask, dilate_px)
    soft = core
    if feather_px > 0:
        radius = max(1, feather_px // 2) if half else feather_px
        soft = core.filter(ImageFilter.GaussianBlur(radius=radius))
        soft = ImageChops.lighter(soft, core)
    return np.asarray(soft, dtype=np.uint8)


def commit_alpha_v14(support: np.ndarray, *,
                     target_dilate_px: int = TARGET_DILATE_PX,
                     feather_px: int = FEATHER_PX) -> np.ndarray:
    """Build the target-side commit alpha without a protected-region term."""
    return _soft(support, target_dilate_px, feather_px, half=False)


def locality_commit(source: np.ndarray, direct: np.ndarray,
                    alpha_u8: np.ndarray) -> np.ndarray:
    """PIL composite, so rounding is bit-identical to the frozen runs."""
    return np.asarray(Image.composite(
        Image.fromarray(direct), Image.fromarray(source),
        Image.fromarray(alpha_u8.astype(np.uint8), "L")), dtype=np.uint8)


def compose_pixels(source: np.ndarray, direct: np.ndarray):
    """Return support, labels, alpha and RGB composite from equal-sized RGB8 arrays."""
    source = np.asarray(source)
    direct = np.asarray(direct)
    if source.dtype != np.uint8 or direct.dtype != np.uint8:
        raise ValueError('composition requires uint8 RGB inputs')
    if source.ndim != 3 or source.shape[-1] != 3 or source.shape != direct.shape or not source.size:
        raise ValueError('composition requires matching nonempty HxWx3 inputs')
    support, labels = necessary_support(mean_rgb_abs_diff(source, direct))
    alpha = commit_alpha_v14(support)
    return support, labels, alpha, locality_commit(source, direct, alpha)


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


def load_row(meta: dict, *, input_root: Path, proposal_root: Path | None = None):
    """Return (source path, source RGB, proposal RGB resized to the source with Lanczos)."""
    if not isinstance(meta.get('row_id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', meta['row_id']):
        raise ValueError('row_id must be a simple file-safe identifier')
    source_path = input_path(input_root, meta['source'])
    direct_path = input_path(input_root if proposal_root is None else proposal_root, meta['direct'])
    source = load_rgb(source_path)
    return source_path, source, load_rgb(direct_path, (source.shape[1], source.shape[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compose saved CompBench proposals. No model inference.')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--proposal-root', type=Path,
                        help='separate saved-proposal root; missing direct fields use <row_id>.png')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    rows = read_rows(args.inputs)
    if args.proposal_root is not None:
        rows = [dict(row, direct=row.get('direct', f"{row['row_id']}.png")) for row in rows]
    if len({row['row_id'] for row in rows}) != len(rows):
        parser.error('duplicate row_id')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('support', 'alpha', 'final'):
        (args.out / name).mkdir()
    items, checked_total, out_of_scope = [], 0, []
    for meta in rows:
        rid = meta['row_id']
        source_path, source, direct = load_row(meta, input_root=args.input_root, proposal_root=args.proposal_root)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        support, _, alpha, final = compose_pixels(source, direct)
        Image.fromarray(np.where(support, 255, 0).astype(np.uint8)).save(args.out / 'support' / f'{rid}.png')
        Image.fromarray(alpha).save(args.out / 'alpha' / f'{rid}.png')
        final_path = args.out / 'final' / f'{rid}.png'
        Image.fromarray(final).save(final_path)
        # Read the encoded files back: every alpha == 0 pixel must equal the source.
        with Image.open(args.out / 'alpha' / f'{rid}.png') as opened:
            guarded = np.asarray(opened.convert('L')) == 0
        if not np.array_equal(load_rgb(final_path)[guarded], source[guarded]):
            raise RuntimeError(f'encoded artifact guarantee mismatch on {rid}')
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError(f'source input changed before publication on {rid}')
        checked_total += int(guarded.sum())
        # The support ceiling labels global edits for manual review; it never prunes pixels.
        if support.sum() / support.size > SUPPORT_CEILING_S or not guarded.any():
            out_of_scope.append(rid)
        item = {'row_id': rid, 'final': f'final/{rid}.png',
                'sha256': hashlib.sha256(final_path.read_bytes()).hexdigest()}
        item.update({key: meta[key] for key in ('subtask', 'eval_key') if key in meta})
        items.append(item)
    manifest = {'schema': 'wmmllm.compbench.composition.v1', 'rows': len(rows), 'items': items,
                'guarantee_px_checked': checked_total, 'guarantee_px_identical': checked_total,
                'out_of_scope_rows': out_of_scope}
    (args.out / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Composed {len(rows)} saved proposals. No model inference.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
