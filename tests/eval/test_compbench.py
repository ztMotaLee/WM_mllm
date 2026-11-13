"""Synthetic CPU checks; no downloaded models, benchmark media or network."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from reproduce.compbench import RowInputs, compose_pixels, locality_commit, necessary_support, select_row
from reproduce.compbench.__main__ import main as compose_main
from eval.compbench.aggregate import ARMS, TAG, METRICS, aggregate_tables, load_archive, read_table
from eval.compbench.run_eval import verify_evaluator


def synthetic_tables():
    return {(arm, subtask): {'one.png': {metric: 1.0 for metric in METRICS},
                             'two.png': {metric: 3.0 for metric in METRICS}}
            for arm in ARMS for subtask in TAG}


class CompositionTests(unittest.TestCase):
    def test_unchanged_and_subthreshold_pixels_preserve_source(self):
        source = np.zeros((96, 96, 3), dtype=np.uint8)
        for direct in (source.copy(), np.full_like(source, 25)):
            support, _, alpha, final = compose_pixels(source, direct)
            self.assertFalse(support.any())
            self.assertFalse(alpha.any())
            np.testing.assert_array_equal(source, final)

    def test_thin_connected_edit_survives_noise_floor(self):
        diff = np.zeros((120, 120), dtype=np.float32)
        diff[60, 20:100] = 26
        diff[5:7, 5:7] = 100
        support, _ = necessary_support(diff)
        self.assertEqual(int(support.sum()), 80)
        self.assertTrue(support[60, 20:100].all())

    def test_component_floor_is_individual_and_inclusive(self):
        diff = np.zeros((120, 120), dtype=np.float32)
        diff[10:18, 10:18] = 26
        diff[40:47, 40:49] = 26
        support, _ = necessary_support(diff)
        self.assertEqual(int(support.sum()), 64)
        self.assertTrue(support[10:18, 10:18].all())

    def test_dilation_feather_and_exact_zero_alpha(self):
        source = np.full((128, 128, 3), 17, dtype=np.uint8)
        direct = source.copy()
        direct[56:72, 56:72] = 230
        support, _, alpha, final = compose_pixels(source, direct)
        self.assertTrue(np.all(alpha[support] == 255))
        self.assertEqual(int(alpha[64, 49]), 255)  # seven-pixel dilation
        self.assertTrue(np.any((alpha > 0) & (alpha < 255)))
        np.testing.assert_array_equal(final[alpha == 0], source[alpha == 0])
        np.testing.assert_array_equal(final[support], direct[support])

    def test_pil_integer_composite_rounding(self):
        source = np.array([[[0, 10, 255], [0, 10, 255]]], dtype=np.uint8)
        direct = np.array([[[255, 20, 0], [255, 20, 0]]], dtype=np.uint8)
        alpha = np.array([[128, 0]], dtype=np.uint8)
        actual = locality_commit(source, direct, alpha)
        np.testing.assert_array_equal(actual, np.array([[[128, 15, 127], [0, 10, 255]]], dtype=np.uint8))

    def test_input_types_are_explicit(self):
        with self.assertRaises(ValueError):
            compose_pixels(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
        with self.assertRaises(ValueError):
            compose_pixels(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2, 3), np.uint8))

    def make_row(self, root, direct, **extras):
        Image.fromarray(np.zeros_like(direct)).save(root / 'source.png')
        Image.fromarray(direct).save(root / 'proposal.png')
        return RowInputs({'row_id': 'sample', 'source': 'source.png', 'direct': 'proposal.png', **extras}, input_root=root)

    def test_full_frame_is_labeled_without_vacuous_guarantee(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.make_row(Path(directory), np.full((96, 96, 3), 255, np.uint8))
            receipt, _, _, _ = select_row(row)
            self.assertEqual(receipt['scope'], 'out_of_scope_global_edit')
            self.assertFalse(receipt['ships_with_guarantee'])
            self.assertFalse(receipt['counted_in_guarantee_headline'])

    def test_out_of_scope_can_retain_nonempty_guarantee(self):
        with tempfile.TemporaryDirectory() as directory:
            direct = np.zeros((128, 640, 3), np.uint8)
            direct[:, :366] = 255
            row = self.make_row(Path(directory), direct)
            receipt, _, _, _ = select_row(row)
            self.assertEqual(receipt['scope'], 'out_of_scope_global_edit')
            self.assertTrue(receipt['ships_with_guarantee'])
            self.assertFalse(receipt['counted_in_guarantee_headline'])

    def test_diagnostic_masks_never_change_installed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = np.zeros((128, 128, 3), np.uint8)
            direct[50:75, 50:75] = 255
            plain = self.make_row(root, direct)
            Image.fromarray(np.full((128, 128), 255, np.uint8)).save(root / 'diagnostic.png')
            diagnostic = RowInputs({'row_id': 'sample', 'source': 'source.png', 'direct': 'proposal.png',
                                    'protected': 'diagnostic.png', 'base_target': 'diagnostic.png',
                                    'declared_box': 'diagnostic.png'}, input_root=root)
            _, alpha_a, final_a, support_a = select_row(plain)
            _, alpha_b, final_b, support_b = select_row(diagnostic)
            for a, b in [(alpha_a, alpha_b), (final_a, final_b), (support_a, support_b)]:
                np.testing.assert_array_equal(a, b)

    def test_loader_rejects_targets_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ref, error in [('edited_image/a.png', PermissionError), ('../source.png', ValueError),
                               ('/source.png', ValueError)]:
                with self.assertRaises(error):
                    RowInputs({'row_id': 'sample', 'source': ref, 'direct': 'x.png'}, input_root=root)

    def test_cli_writes_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = np.zeros((96, 96, 3), np.uint8)
            direct[35:55, 35:55] = 255
            self.make_row(root, direct)
            manifest = root / 'inputs.json'
            manifest.write_text(json.dumps([{'row_id': 'sample', 'source': 'source.png', 'direct': 'proposal.png'}]))
            out = root / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compose_main(['--inputs', str(manifest), '--input-root', str(root), '--out', str(out)]), 0)
            receipt = json.loads((out / 'receipts/sample.json').read_text())
            self.assertGreater(receipt['guarantee_px_checked'], 0)
            self.assertEqual(receipt['guarantee_px_checked'], receipt['guarantee_px_identical'])
            self.assertEqual(receipt['source_path'], 'source.png')
            self.assertTrue((out / 'RUN_MANIFEST.json').is_file())

    def test_cli_detects_source_change_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_row(root, np.full((96, 96, 3), 255, np.uint8))
            manifest = root / 'inputs.json'
            manifest.write_text(json.dumps([{'row_id': 'sample', 'source': 'source.png', 'direct': 'proposal.png'}]))

            def change_source(row):
                result = select_row(row)
                Image.fromarray(np.full((96, 96, 3), 77, np.uint8)).save(root / 'source.png')
                return result

            out = root / 'out'
            with patch('reproduce.compbench.__main__.select_row', side_effect=change_source):
                with self.assertRaises(RuntimeError):
                    compose_main(['--inputs', str(manifest), '--input-root', str(root), '--out', str(out)])
            self.assertFalse((out / 'RUN_MANIFEST.json').exists())


class AggregationTests(unittest.TestCase):
    def test_subtasks_have_equal_weight_with_rounding(self):
        tables = synthetic_tables()
        for arm in ARMS:
            for subtask, value, count in [('add', 1.2346, 20), ('remove', 3.4566, 2), ('replace', 8.7656, 1)]:
                tables[(arm, subtask)] = {str(i): {m: value for m in METRICS} for i in range(count)}
        out = aggregate_tables(tables)
        expected = float(round(np.mean([np.float64(1.235), np.float64(3.457), np.float64(8.766)]), 3))
        self.assertEqual(out['groups']['Local']['ours']['LC-I'], expected)
        self.assertEqual(out['n']['add'], 20)

    def test_psnr_joins_names_and_intersects_finite_rows(self):
        tables = synthetic_tables()
        tables[('paste', 'implicit')]['one.png']['PSNR'] = float('inf')
        tables[('ours', 'implicit')] = dict(reversed(list(tables[('ours', 'implicit')].items())))
        tables[('ours', 'implicit')]['two.png']['PSNR'] = 9
        out = aggregate_tables(tables)
        self.assertEqual(out['n']['implicit_PSNR_common_finite'], 1)
        self.assertEqual(out['groups']['Implicit']['ours']['PSNR'], 9)
        self.assertEqual(out['groups']['Implicit']['direct']['PSNR'], 3)
        self.assertEqual(out['groups']['Implicit']['ours']['LC-I'], 2)

    def test_missing_names_and_nan_are_not_imputed(self):
        tables = synthetic_tables()
        del tables[('paste', 'implicit')]['one.png']
        tables[('ours', 'implicit')]['one.png']['LC-T'] = float('nan')
        out = aggregate_tables(tables)
        self.assertEqual(out['n']['implicit_PSNR_common_finite'], 1)
        self.assertEqual(out['groups']['Implicit']['ours']['LC-T'], 3)

    def test_csv_coercion_and_duplicates(self):
        text = 'Image_Name,LC-T,LC-I,PSNR,SSIM,LPIPS\na.png,bad,,inf,0.9,0.01\n'
        table = read_table(text)
        self.assertTrue(np.isnan(table['a.png']['LC-T']))
        self.assertTrue(np.isnan(table['a.png']['LC-I']))
        self.assertTrue(np.isinf(table['a.png']['PSNR']))
        with self.assertRaises(ValueError):
            read_table(text + 'a.png,1,1,1,1,1\n')

    def test_archive_reader_requires_one_regular_member_per_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.tar'
            text = b'Image_Name,LC-T,LC-I,PSNR,SSIM,LPIPS\na.png,1,2,3,4,5\n'
            with tarfile.open(path, 'w') as archive:
                for model in ARMS.values():
                    for subtask, tag in TAG.items():
                        member = tarfile.TarInfo(f'scores/{model}/{tag}/{model}/{subtask}_detailed.csv')
                        member.size = len(text)
                        archive.addfile(member, io.BytesIO(text))
            result = aggregate_tables(load_archive(path))
            self.assertEqual(result['groups']['Local']['direct']['LC-I'], 2)

    def test_incomplete_population_fails(self):
        tables = synthetic_tables()
        del tables[('ours', 'implicit')]
        with self.assertRaises(ValueError):
            aggregate_tables(tables)

    def test_empty_psnr_intersection_and_unmeasured_cells_fail(self):
        for metric in ('PSNR', 'LC-I'):
            tables = synthetic_tables()
            for row in tables[('ours', 'implicit')].values():
                row[metric] = float('nan')
            with self.assertRaises(ValueError):
                aggregate_tables(tables)

    def test_bundled_evaluator_matches_approved_portable_hash(self):
        from eval.compbench.run_eval import PORTABLE_EVALUATOR
        self.assertEqual(len(verify_evaluator(PORTABLE_EVALUATOR)), 64)

    def test_wrong_evaluator_is_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eval_all.py'
            path.write_text('raise RuntimeError("must not execute")\n')
            with self.assertRaises(ValueError):
                verify_evaluator(path)
