"""Historical selector: full support, with explicit scope and guarantee labels.

Modified for local package imports and explicit input records.
"""
from __future__ import annotations
import numpy as np
from .compose import compose_row, audit_components, exceedance_report, SUPPORT_CEILING_S
from .inputs import RowInputs

COMMIT_ARM = 'commit_v14_support'
OUT_OF_SCOPE_ARM = 'out_of_scope_global_edit'
COVERAGE_FLOOR_Q = 0.50
FORBIDDEN_ARMS = frozenset({'direct_proposal', 'direct', 'unguarded_direct'})

def assert_never_direct(arm: str, row_id: str, alpha: np.ndarray,
                        final: np.ndarray, direct: np.ndarray) -> bool:
    """Checked on the PIXELS, not on the label.

    A label alone proves nothing: a support that covers the whole frame gives
    alpha == 255 everywhere, the composite IS Direct exactly, and a readback
    guarantee check would pass VACUOUSLY over an empty set while the receipt
    still said `ships_with_guarantee: True`.  So the arm name and the
    guaranteed set being non-empty are both checked -- but a vacuous row is
    now CLASSIFIED out of scope by the composer and ships under
    OUT_OF_SCOPE_ARM with `ships_with_guarantee: False`, instead of raising and
    shipping nothing.  Reaching this raise under the
    commit arm therefore means the scope classifier and the pixels disagree,
    which is a real bug and must stop the run.
    """
    if arm in FORBIDDEN_ARMS:
        raise RuntimeError(
            f"Composition invariant violated: row {row_id} tried to ship {arm}. "
            "Unguarded Direct is not a selectable arm.")
    if arm == COMMIT_ARM and int((alpha == 0).sum()) == 0:
        raise RuntimeError(
            f"Composition invariant violated: row {row_id} carries the commit arm with "
            "an EMPTY guaranteed set; the scope classifier should have labelled "
            f"it {OUT_OF_SCOPE_ARM}.")
    # NOT a violation: if Direct itself changed nothing outside the support,
    # the guarded composite legitimately equals Direct pixel for pixel.  Both
    # This equality is valid for a perfectly localized edit:
    # raising here would refuse a perfectly localised edit.  Recorded, not
    # gated.  The guarantee is proved by the readback comparison to SOURCE.
    return bool(np.array_equal(final, direct))


def select_row(row: RowInputs) -> dict:
    """Ship the FULL necessary support.  No pruning, no anchor, no refusal.

    The only branch is SCOPE, and it is decided in the composer on the pixels.
    `base_target` is not read here at all: after the mechanism review there is
    no path by which a SAM/grounding artefact decides a committed pixel.
    """
    frame_px = int(row.diff.size)
    rec, support, labels, alpha, final = compose_row(row)
    core = alpha >= 128
    guarded = alpha == 0
    q = float(guarded.sum() / frame_px)
    in_scope = rec["scope"] == "in_scope"

    arm = COMMIT_ARM if in_scope else OUT_OF_SCOPE_ARM
    equals_direct = assert_never_direct(arm, row.rid, alpha, final, row.direct)
    audit = audit_components(row.diff, core, row.protected)
    receipt = {
        "row_id": row.rid, "split": row.split, "operation": row.operation,
        "instruction": row.instruction, "frame_px": frame_px,
        "chosen_arm": arm,
        # derived from the pixels, never
        # hard-coded.  The readback in main() re-derives it independently and
        # raises on disagreement.
        "ships_with_guarantee": bool(int(guarded.sum()) > 0),
        # `s > C` with a still-NON-EMPTY guaranteed set is a
        # real branch (a wide band edit), and there `ships_with_guarantee` is
        # legitimately True while the row is still out of scope.  The two facts
        # are different and must not be conflated: the pixel fact stays as it
        # is, and a SEPARATE field says whether the row is counted in the
        # guarantee headline.  Only in-scope rows are.
        "counted_in_guarantee_headline": bool(in_scope
                                              and int(guarded.sum()) > 0),
        "guarantee_vacuous": bool(int(guarded.sum()) == 0),
        "meets_coverage_floor_q": bool(q >= COVERAGE_FLOOR_Q),
        "coverage_floor_q": COVERAGE_FLOOR_Q,
        "coverage_floor_is_reported_not_gated": True,
        "guarantee_boundary": "alpha == 0",
        "guarantee_boundary_note":
            "NOT the complement of the binary support PNG: dilate 7 + feather 3 "
            "write outside the support, so the support PNG is the write SEED "
            "and {alpha == 0} is the guaranteed set.  The gap is reported as "
            "nonguaranteed_outside_support_px.",
        "scope": rec["scope"],
        "scope_reasons": rec["scope_reasons"],
        "manual_review_required": bool(not in_scope),
        "support_ceiling_s": SUPPORT_CEILING_S,
        "support_ceiling_statistic": "s = |support| / |frame| (PRE-REGISTERED)",
        "policy_label": ("full_support" if in_scope
                         else "out_of_scope_global_edit"),
        "pruning_applied": False,
        "undercommit_vs_full_support_px": 0,
        "full_support_px": rec["support_px"],
        "full_support_guarantee": q,
        "shipped_support_px": int(support.sum()),
        "shipped_support_fraction": rec["support_fraction"],
        "shipped_core_px": int(core.sum()),
        "shipped_core_fraction": float(core.sum() / frame_px),
        "guaranteed_px": int(guarded.sum()),
        "guaranteed_fraction": q,
        # the 0 < alpha < 128 fringe is neither guaranteed
        # nor inside `core`.  Reported explicitly rather than left implicit in
        # a guarantee/fragment partition.
        "fringe_px_0_lt_alpha_lt_128": rec["fringe_px_0_lt_alpha_lt_128"],
        "outside_support_px": rec["outside_support_px"],
        "nonguaranteed_outside_support_px": rec["nonguaranteed_outside_support_px"],
        "shipped_equals_direct": equals_direct,
        "protected_written_px": int((row.protected & core).sum()),
        "v11_committed": rec["v11_committed"],
        # provenance in the receipt itself, so a
        # receipt is self-contained without the compose diagnostic.
        "source_path": str(row.source_path),
        "source_sha256": rec["source_sha256"],
        "direct_path": str(row.direct_path),
        "direct_sha256": rec["direct_sha256"],
    }
    receipt.update({k: audit[k] for k in
                    ("general_components", "general_px", "visible_fragments",
                     "visible_fragment_px")})
    receipt["visible_detail"] = audit["visible_detail"]
    receipt["exceedance"] = exceedance_report(support, core, row.declared_box)
    receipt["drift"] = rec["drift"]
    receipt["roi_companion"] = rec["roi_companion"]
    return receipt, alpha, final, support

