"""Compose an explicit external saved-proposal manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .inputs import RowInputs, input_path, load_rgb, read_rows
from .select import select_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--proposal-root', type=Path,
                        help='separate saved-proposal root; missing direct fields use <row_id>.png')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    rows = read_rows(args.inputs)
    if args.proposal_root is not None:
        rows = [dict(row, direct=row.get('direct', f"{row['row_id']}.png")) for row in rows]
    identifiers = [row['row_id'] for row in rows]
    if len(set(identifiers)) != len(rows):
        parser.error('duplicate row_id')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('support', 'alpha', 'final', 'receipts'):
        (args.out / name).mkdir()
    receipts = []
    items = []
    for meta in rows:
        row = RowInputs(meta, input_root=args.input_root, proposal_root=args.proposal_root)
        receipt, alpha, final, support = select_row(row)
        Image.fromarray(np.where(support, 255, 0).astype(np.uint8)).save(args.out / 'support' / f'{row.rid}.png')
        Image.fromarray(alpha).save(args.out / 'alpha' / f'{row.rid}.png')
        final_path = args.out / 'final' / f'{row.rid}.png'
        Image.fromarray(final).save(final_path)
        with Image.open(args.out / 'alpha' / f'{row.rid}.png') as opened:
            disk_alpha = np.asarray(opened.convert('L'))
        disk_final = load_rgb(final_path)
        if not np.array_equal(disk_alpha, alpha) or not np.array_equal(disk_final, final):
            raise RuntimeError(f'encoded artifact mismatch on {row.rid}')
        guarded = disk_alpha == 0
        checked = int(guarded.sum())
        identical = int((disk_final[guarded] == row.source[guarded]).all(axis=-1).sum())
        if checked != identical or receipt['ships_with_guarantee'] != bool(checked):
            raise RuntimeError(f'encoded artifact guarantee mismatch on {row.rid}')
        current_source = input_path(args.input_root, row.source_path)
        if hashlib.sha256(current_source.read_bytes()).hexdigest() != receipt['source_sha256']:
            raise RuntimeError(f'source input changed before publication on {row.rid}')
        receipt.update(guarantee_px_checked=checked, guarantee_px_identical=identical,
                       diagnostics_present=row.diagnostics_present,
                       absent_diagnostic_masks_are_not_measured=True)
        (args.out / 'receipts' / f'{row.rid}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        receipts.append(receipt)
        item = {'row_id': row.rid, 'final': f'final/{row.rid}.png',
                'sha256': hashlib.sha256(final_path.read_bytes()).hexdigest()}
        for key in ('subtask', 'eval_key'):
            if key in meta:
                item[key] = meta[key]
        items.append(item)
    manifest = {'schema': 'wmmllm.compbench.composition.v1', 'rows': len(rows),
                'items': items,
                'input_manifest_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob('*.py'))},
                'guarantee_px_checked': sum(r['guarantee_px_checked'] for r in receipts),
                'guarantee_px_identical': sum(r['guarantee_px_identical'] for r in receipts),
                'out_of_scope_rows': [r['row_id'] for r in receipts if r['scope'] != 'in_scope']}
    (args.out / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Composed {len(rows)} saved proposals. No model inference.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
