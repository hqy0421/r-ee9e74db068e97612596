# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tests for prediction-ID managed rolling calibration."""

import os

import numpy as np
import pytest

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.rolling import CalibrationUnavailable
from deel.puncc.api.rolling import EventOrderError
from deel.puncc.api.rolling import PredictionIDConflict
from deel.puncc.api.rolling import RollingConformalPredictor
from deel.puncc.api.rolling import StaleTruthVersion
from deel.puncc.api.rolling import TruthWithdrawn
from deel.puncc.api.rolling import UnknownPredictionID


class FixedLinearPredictor:
    """Deterministic, fixed (untrained) predictor used to assert that the
    predictor is never retrained and that point predictions are frozen."""

    def __init__(self, slope=2.0, bias=1.0):
        self.slope = slope
        self.bias = bias
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X, y=None, **kwargs):  # pragma: no cover - never called
        self.fit_calls += 1

    def predict(self, X, **kwargs):
        self.predict_calls += 1
        X = np.asarray(X, dtype=np.float64)
        return self.slope * X.squeeze() + self.bias


def _make_rolling(
    window=None, min_samples=1, predictor=None, calibrator=None, noise=0.5
):
    predictor = predictor or FixedLinearPredictor()
    calibrator = calibrator or BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )
    return RollingConformalPredictor(
        predictor, calibrator, window=window, min_samples=min_samples
    )


def _issue_point_only(rolling_cp, prediction_id, X, alpha, **kwargs):
    """Issue a prediction when no (or not enough) calibration sample is
    expected: the point-only record is retained behind the expected
    CalibrationUnavailable exception."""
    with pytest.raises(CalibrationUnavailable):
        rolling_cp.issue_prediction(prediction_id, X, alpha, **kwargs)


def _prime(
    rolling_cp, n, *, x0=1.0, gap=1.0, noise=0.5, start_time=0.0, alpha=0.2
):
    """Issue ``n`` predictions and backfill their ground truths.

    Point prediction for x = x0 + i*gap is 2*x + 1. Ground truth is offset by
    a constant ``noise``, so every residual equals ``noise``. After priming,
    subsequent issuances have ``n`` known calibration samples.
    """
    for i in range(n):
        x = x0 + i * gap
        t = start_time + i
        _issue_point_only(
            rolling_cp,
            f"seed{i}",
            np.array([x]),
            alpha=alpha,
            issue_time=float(t),
        )
        rolling_cp.update_truth(
            f"seed{i}",
            np.array([2.0 * x + 1.0 + noise]),
            version=1,
            event_time=float(t) + 0.1,
        )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_validates_arguments():
    predictor = FixedLinearPredictor()
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )

    with pytest.raises(ValueError):
        RollingConformalPredictor(object(), calibrator)

    with pytest.raises(ValueError):
        RollingConformalPredictor(predictor, object())

    with pytest.raises(ValueError):
        RollingConformalPredictor(predictor, calibrator, window=0)

    with pytest.raises(ValueError):
        RollingConformalPredictor(predictor, calibrator, window=-1.0)

    with pytest.raises(ValueError):
        RollingConformalPredictor(predictor, calibrator, min_samples=0)


# ---------------------------------------------------------------------------
# Basic issuance flow and interval correctness
# ---------------------------------------------------------------------------


def test_first_issuance_unavailable_but_point_prediction_retained():
    rolling_cp = _make_rolling(min_samples=2)

    # No ground truth known yet: interval unavailable, but point prediction
    # is retained.
    _issue_point_only(
        rolling_cp, "p0", X=np.array([1.0]), alpha=0.1, issue_time=0.0
    )
    assert not rolling_cp.is_interval_available("p0")
    np.testing.assert_array_equal(
        rolling_cp.get_point_prediction("p0"), np.array(3.0)
    )
    with pytest.raises(CalibrationUnavailable):
        rolling_cp.get_interval("p0")


def test_backfill_enables_interval_with_manual_residual_check():
    rolling_cp = _make_rolling(min_samples=2)

    # Two point-only issuances first.
    _issue_point_only(
        rolling_cp, "p0", X=np.array([1.0]), alpha=0.4, issue_time=0.0
    )
    _issue_point_only(
        rolling_cp, "p1", X=np.array([2.0]), alpha=0.4, issue_time=1.0
    )

    # Backfill both truths (residuals 0.5 each).
    rolling_cp.update_truth("p0", np.array(3.5), version=1, event_time=1.2)
    rolling_cp.update_truth("p1", np.array(5.5), version=1, event_time=1.4)

    # A new prediction now has two known samples.
    y_pred, y_lo, y_hi = rolling_cp.issue_prediction(
        "p2", X=np.array([3.0]), alpha=0.4, issue_time=2.0
    )

    expected_residuals = np.array([0.5, 0.5])
    q_hat = np.quantile(
        np.concatenate([expected_residuals, [np.inf]]),
        0.6,
        method="inverted_cdf",
    )
    np.testing.assert_allclose(y_pred, np.array([7.0]))
    np.testing.assert_allclose(y_lo, np.array([7.0 - q_hat]))
    np.testing.assert_allclose(y_hi, np.array([7.0 + q_hat]))
    assert rolling_cp.is_interval_available("p2")


def test_issue_uses_only_ground_truths_known_at_issuance_time():
    rolling_cp = _make_rolling()

    # A first point-only prediction.
    _issue_point_only(
        rolling_cp, "a", X=np.array([0.0]), alpha=0.2, issue_time=0.0
    )
    # A ground truth arriving *after* the next issuance must not be used.
    _issue_point_only(
        rolling_cp, "b", X=np.array([1.0]), alpha=0.2, issue_time=1.0
    )
    rolling_cp.update_truth("a", np.array([2.0]), version=1, event_time=1.1)

    # c is issued right after a's truth: only a is a calibration candidate
    # (b has no known truth yet). n=1 requires alpha >= 1/2.
    y_pred_c, lo_c, hi_c = rolling_cp.issue_prediction(
        "c", X=np.array([2.0]), alpha=0.5, issue_time=1.2
    )
    # Point pred for x=0 is 1.0; truth 2.0 -> residual 1.0.
    q_expected = np.quantile(
        np.concatenate([np.array([1.0]), [np.inf]]),
        0.5,
        method="inverted_cdf",
    )
    np.testing.assert_allclose(y_pred_c, 5.0)
    np.testing.assert_allclose(lo_c, 5.0 - q_expected)
    np.testing.assert_allclose(hi_c, 5.0 + q_expected)


def test_predictor_is_fixed_and_point_predictions_frozen():
    predictor = FixedLinearPredictor()
    rolling_cp = _make_rolling(predictor=predictor)
    _prime(rolling_cp, n=1)

    # A calibrated issuance (n=1 -> alpha >= 1/2).
    rolling_cp.issue_prediction(
        "new", X=np.array([5.0]), alpha=0.5, issue_time=2.0
    )

    # Replaying an existing, calibrated ID never re-runs the predictor.
    calls_before = predictor.predict_calls
    for _ in range(3):
        y_pred, _, _ = rolling_cp.issue_prediction(
            "new", X=np.array([5.0]), alpha=0.5, issue_time=2.0
        )
    assert predictor.predict_calls == calls_before
    assert predictor.fit_calls == 0
    np.testing.assert_array_equal(y_pred, np.array([11.0]))


# ---------------------------------------------------------------------------
# Idempotency and conflict handling
# ---------------------------------------------------------------------------


def test_same_id_same_request_returns_identical_result():
    rolling_cp = _make_rolling()
    _prime(rolling_cp, n=1)

    first = rolling_cp.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )
    # Ground truth correction after b's interval was published must not alter
    # the already published interval.
    rolling_cp.update_truth("seed0", np.array([9.9]), version=2, event_time=2.5)
    second = rolling_cp.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )

    for arr_1, arr_2 in zip(first, second):
        np.testing.assert_array_equal(arr_1, arr_2)


def test_same_id_different_content_conflicts():
    rolling_cp = _make_rolling()
    _issue_point_only(
        rolling_cp, "a", X=np.array([1.0]), alpha=0.1, issue_time=0.0
    )

    # Different features
    with pytest.raises(PredictionIDConflict):
        rolling_cp.issue_prediction(
            "a", X=np.array([1.5]), alpha=0.1, issue_time=1.0
        )
    # Different alpha
    with pytest.raises(PredictionIDConflict):
        rolling_cp.issue_prediction(
            "a", X=np.array([1.0]), alpha=0.2, issue_time=1.0
        )


# ---------------------------------------------------------------------------
# Ground truth versioning: corrections, withdrawals, stale versions
# ---------------------------------------------------------------------------


def test_truth_correction_replaces_residual_for_new_issuances():
    rolling_cp = _make_rolling(noise=1.0)
    _prime(rolling_cp, n=1, noise=1.0)
    # seed0: x=1, point=3, truth v1 = 4 -> residual 1.0

    # b calibrated against version 1 (residual 1.0).
    _, lo_v1, hi_v1 = rolling_cp.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )

    # Correction: version 2 replaces version 1 (residual becomes 0.1).
    rolling_cp.update_truth("seed0", np.array([3.1]), version=2, event_time=2.5)
    assert rolling_cp.get_status("seed0")["truth_version"] == 2
    np.testing.assert_array_equal(rolling_cp.get_truth("seed0"), np.array(3.1))

    # c calibrated against the corrected truth (residual 0.1).
    _, lo_v2, hi_v2 = rolling_cp.issue_prediction(
        "c", X=np.array([2.0]), alpha=0.5, issue_time=3.0
    )

    # New interval reflects the correction; old interval stays untouched.
    assert (hi_v2 - lo_v2) < (hi_v1 - lo_v1)
    old_lo, old_hi = rolling_cp.get_interval("b")
    np.testing.assert_array_equal(old_lo, lo_v1)
    np.testing.assert_array_equal(old_hi, hi_v1)


def test_stale_versions_are_rejected_and_cannot_resurrect():
    rolling_cp = _make_rolling()
    _issue_point_only(
        rolling_cp, "a", X=np.array([1.0]), alpha=0.1, issue_time=0.0
    )

    rolling_cp.update_truth("a", np.array([3.3]), version=1, event_time=0.5)
    # Same version and older versions are stale.
    with pytest.raises(StaleTruthVersion):
        rolling_cp.update_truth("a", np.array([3.4]), version=1, event_time=0.6)
    with pytest.raises(StaleTruthVersion):
        rolling_cp.update_truth("a", np.array([3.4]), version=0, event_time=0.7)

    # Withdrawal at version 2.
    rolling_cp.withdraw_truth("a", version=2, event_time=0.8)
    assert rolling_cp.is_truth_withdrawn("a")
    with pytest.raises(TruthWithdrawn):
        rolling_cp.get_truth("a")

    # The pre-withdrawal versions cannot be resurrected.
    for stale_version in (0, 1, 2):
        with pytest.raises(StaleTruthVersion):
            rolling_cp.update_truth(
                "a",
                np.array([3.4]),
                version=stale_version,
                event_time=0.9,
            )

    # A strictly newer version restores a valid ground truth.
    rolling_cp.update_truth("a", np.array([3.45]), version=3, event_time=1.0)
    assert not rolling_cp.is_truth_withdrawn("a")
    np.testing.assert_array_equal(rolling_cp.get_truth("a"), np.array(3.45))


def test_withdrawn_truth_is_excluded_from_calibration():
    rolling_cp = _make_rolling(min_samples=1)
    _prime(rolling_cp, n=1)

    # One calibrated issuance works before withdrawal (n=1 -> alpha >= 1/2).
    rolling_cp.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )
    rolling_cp.withdraw_truth("seed0", version=2, event_time=2.5)

    # With the only truth withdrawn, no valid sample remains.
    with pytest.raises(CalibrationUnavailable):
        rolling_cp.issue_prediction(
            "c", X=np.array([3.0]), alpha=0.5, issue_time=3.0
        )
    # The point prediction of c is still retained for later backfill.
    np.testing.assert_array_equal(
        rolling_cp.get_point_prediction("c"), np.array([7.0])
    )


def test_truth_events_on_unknown_ids_are_rejected():
    rolling_cp = _make_rolling()
    with pytest.raises(UnknownPredictionID):
        rolling_cp.update_truth(
            "ghost", np.array([1.0]), version=1, event_time=0.0
        )
    with pytest.raises(UnknownPredictionID):
        rolling_cp.withdraw_truth("ghost", version=1, event_time=0.1)
    with pytest.raises(UnknownPredictionID):
        rolling_cp.get_point_prediction("ghost")


# ---------------------------------------------------------------------------
# Rolling window semantics (left-closed, right-opened)
# ---------------------------------------------------------------------------


def test_window_is_left_closed_right_opened():
    rolling_cp = _make_rolling(window=2.0, min_samples=1)

    # Build two prediction records with their truths at known issue times.
    # Whether each issuance could itself be calibrated is irrelevant to the
    # window logic, so records are created through the public flow and any
    # CalibrationUnavailable is absorbed while retaining the point record.
    for pid, t in (("seed0", 0.0), ("seed1", 1.0)):
        try:
            rolling_cp.issue_prediction(
                pid, X=np.array([1.0]), alpha=0.5, issue_time=t
            )
        except CalibrationUnavailable:
            pass
        rolling_cp.update_truth(
            pid, np.array([3.5]), version=1, event_time=t + 0.1
        )

    # At t=2 with window 2 -> window [0, 2): both seeds are candidates
    # (t=0 equals the inclusive left bound).
    rolling_cp.issue_prediction(
        "c", X=np.array([9.0]), alpha=0.4, issue_time=2.0
    )
    eligible_c = rolling_cp._eligible_calibration_ids(
        rolling_cp._predictions["c"]
    )
    assert sorted(eligible_c) == ["seed0", "seed1"]

    # At t=3 with window 2 -> window [1, 3): seed0 (t=0) excluded, seed1
    # (t=1) included. d still calibrates on the single remaining seed.
    rolling_cp.issue_prediction(
        "d", X=np.array([9.0]), alpha=0.5, issue_time=3.0
    )
    eligible_d = rolling_cp._eligible_calibration_ids(
        rolling_cp._predictions["d"]
    )
    assert eligible_d == ["seed1"]


def test_event_arriving_after_issuance_is_unknown_at_issuance():
    rolling_cp = _make_rolling(window=10.0)
    _issue_point_only(
        rolling_cp, "a", X=np.array([1.0]), alpha=0.2, issue_time=0.0
    )
    _issue_point_only(
        rolling_cp, "b", X=np.array([2.0]), alpha=0.2, issue_time=1.0
    )
    # Truth of a arrives only after b was issued: cannot calibrate b.
    rolling_cp.update_truth("a", np.array([3.5]), version=1, event_time=1.5)
    record_b = rolling_cp._predictions["b"]
    assert rolling_cp._eligible_calibration_ids(record_b) == []


# ---------------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------------


def test_events_must_be_processed_in_order():
    rolling_cp = _make_rolling()
    _issue_point_only(
        rolling_cp, "a", X=np.array([1.0]), alpha=0.1, issue_time=1.0
    )

    # An earlier event timestamp after a later one violates processing order.
    with pytest.raises(EventOrderError):
        _issue_point_only(
            rolling_cp, "b", X=np.array([2.0]), alpha=0.1, issue_time=0.5
        )

    rolling_cp = _make_rolling()
    _issue_point_only(
        rolling_cp,
        "a",
        X=np.array([1.0]),
        alpha=0.1,
        issue_time=0.0,
        seq=0,
    )
    with pytest.raises(EventOrderError):
        _issue_point_only(
            rolling_cp,
            "b",
            X=np.array([2.0]),
            alpha=0.1,
            issue_time=1.0,
            seq=0,
        )


def test_auto_sequence_numbers_are_monotonic():
    rolling_cp = _make_rolling()
    _issue_point_only(rolling_cp, "a", X=np.array([1.0]), alpha=0.1)
    assert rolling_cp.get_status("a")["issue_seq"] == 0
    rolling_cp.update_truth("a", np.array([3.2]), version=1)
    # The truth event consumes sequence number 1. The next issuance therefore
    # receives sequence number 2, even though calibration fails (alpha=0.1 is
    # inadmissible with a single sample) and the record is rolled back.
    with pytest.raises(ValueError):
        rolling_cp.issue_prediction("b", X=np.array([2.0]), alpha=0.1)
    with pytest.raises(UnknownPredictionID):
        rolling_cp.get_status("b")
    # A subsequent well-formed issuance is sequence 3.
    rolling_cp.issue_prediction("c", X=np.array([3.0]), alpha=0.5)
    assert rolling_cp.get_status("c")["issue_seq"] == 3


# ---------------------------------------------------------------------------
# No half records on failure
# ---------------------------------------------------------------------------


class FailingCalibrator(BaseCalibrator):
    """Calibrator whose calibrate call always fails unexpectedly.

    The fit override keeps the single-sample 1D normalization, so the failure
    is guaranteed to originate from `calibrate` (the half-record path under
    test), not from a shape issue.
    """

    def fit(self, *, y_true, y_pred, **kwargs):
        del kwargs
        self._residuals = np.atleast_1d(
            self.nonconf_score_func(np.asarray(y_pred), np.asarray(y_true))
        )
        self._len_calib = len(self._residuals)

    def calibrate(
        self, *, alpha, X=None, y_pred, weights=None, correction=None
    ):
        raise RuntimeError("boom")


def test_unexpected_calibration_failure_publishes_nothing():
    predictor = FixedLinearPredictor()
    failing = FailingCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )
    rolling_cp = _make_rolling(predictor=predictor, calibrator=failing)
    _prime(rolling_cp, n=1)
    with pytest.raises(RuntimeError, match="boom"):
        rolling_cp.issue_prediction(
            "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
        )
    # No half record: b is not registered at all.
    with pytest.raises(UnknownPredictionID):
        rolling_cp.get_point_prediction("b")
    # Processing clock still advanced: an older event timestamp is now rejected
    # even though the failed issuance published nothing.
    with pytest.raises(EventOrderError):
        rolling_cp.issue_prediction(
            "c", X=np.array([2.0]), alpha=0.5, issue_time=1.5
        )
    # A later, well-ordered issuance keeps the point-prediction retention
    # behavior (its calibration fails the same way, but the point remains).
    with pytest.raises(RuntimeError, match="boom"):
        rolling_cp.issue_prediction(
            "c", X=np.array([2.0]), alpha=0.5, issue_time=3.0
        )
    with pytest.raises(UnknownPredictionID):
        rolling_cp.get_point_prediction("c")


# ---------------------------------------------------------------------------
# Save / load and resume
# ---------------------------------------------------------------------------


def test_save_load_resumes_full_state(tmp_path):
    rolling_cp = _make_rolling(window=5.0, min_samples=1)
    _prime(rolling_cp, n=1)

    first = rolling_cp.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )
    path = os.path.join(tmp_path, "rolling.pkl")
    rolling_cp.save(path)

    loaded = RollingConformalPredictor.load(path)

    # Frozen results are identical on replay.
    replayed = loaded.issue_prediction(
        "b", X=np.array([2.0]), alpha=0.5, issue_time=2.0
    )
    for arr_1, arr_2 in zip(first, replayed):
        np.testing.assert_array_equal(arr_1, arr_2)

    # Versioned truths and configuration are restored.
    np.testing.assert_array_equal(loaded.get_truth("seed0"), np.array([3.5]))
    assert loaded.get_status("seed0")["truth_version"] == 1
    assert loaded.window == 5.0 and loaded.min_samples == 1

    # Processing resumes: the restored clock rejects an out-of-order event.
    with pytest.raises(EventOrderError):
        loaded.issue_prediction(
            "x", X=np.array([9.0]), alpha=0.2, issue_time=0.0, seq=0
        )
    loaded.update_truth("seed0", np.array([3.6]), version=2, event_time=2.5)
    np.testing.assert_array_equal(loaded.get_truth("seed0"), np.array(3.6))

    # Stale versions remain rejected after reload.
    with pytest.raises(StaleTruthVersion):
        loaded.update_truth("seed0", np.array([3.7]), version=1, event_time=2.6)


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    rolling_cp = _make_rolling()
    _issue_point_only(rolling_cp, "a", X=np.array([1.0]), alpha=0.1)
    path = os.path.join(tmp_path, "rolling.pkl")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("deel.puncc.api.rolling.pickle.dump", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        rolling_cp.save(path)
    assert not os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# Point-only retry path
# ---------------------------------------------------------------------------


def test_point_only_record_retries_with_original_issue_time():
    rolling_cp = _make_rolling(min_samples=2)

    # Two point-only predictions.
    _issue_point_only(
        rolling_cp, "a", X=np.array([1.0]), alpha=0.2, issue_time=0.0
    )
    _issue_point_only(
        rolling_cp, "b", X=np.array([2.0]), alpha=0.2, issue_time=1.0
    )
    # a's truth alone is not enough for min_samples=2.
    rolling_cp.update_truth("a", np.array([3.5]), version=1, event_time=1.2)
    with pytest.raises(CalibrationUnavailable):
        rolling_cp.issue_prediction(
            "b", X=np.array([2.0]), alpha=0.2, issue_time=1.0
        )
    # b's own truth arriving after its issuance cannot be used for b itself:
    # replay still only sees truths known at b's original issue time (t=1).
    rolling_cp.update_truth("b", np.array([4.5]), version=1, event_time=1.5)
    with pytest.raises(CalibrationUnavailable):
        rolling_cp.issue_prediction(
            "b", X=np.array([2.0]), alpha=0.2, issue_time=1.0
        )

    # A later prediction c can use both a and b (n=2 -> alpha >= 1/3).
    y_pred, lo, hi = rolling_cp.issue_prediction(
        "c", X=np.array([3.0]), alpha=0.4, issue_time=2.0
    )
    assert rolling_cp.is_interval_available("c")
    np.testing.assert_array_equal(y_pred, np.array(7.0))
    assert np.all(lo <= y_pred) and np.all(hi >= y_pred)


# ---------------------------------------------------------------------------
# Integration with a real sklearn predictor and BasePredictor wrapper
# ---------------------------------------------------------------------------


def test_integration_with_base_predictor_and_sklearn():
    from sklearn.linear_model import LinearRegression

    rng = np.random.RandomState(0)
    X_train = rng.randn(50, 1)
    y_train = 3.0 * X_train.squeeze() + 1.0

    model = LinearRegression().fit(X_train, y_train)
    predictor = BasePredictor(model, is_trained=True)
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )
    rolling_cp = RollingConformalPredictor(
        predictor, calibrator, window=100.0, min_samples=3
    )

    # Issue three point-only predictions, then backfill homogeneous truths.
    # n=3 requires alpha >= 1/4; alpha=0.3 is admissible.
    X_new = np.linspace(0, 1, 5).reshape(-1, 1)
    point_preds = model.predict(X_new)
    for i in range(3):
        with pytest.raises(CalibrationUnavailable):
            rolling_cp.issue_prediction(
                f"p{i}",
                X_new[i].reshape(1, -1),
                alpha=0.3,
                issue_time=float(i),
            )
    # Truths are processed after all three issuances; their timestamps must
    # stay non-decreasing relative to the last issuance (t=2).
    for i in range(3):
        rolling_cp.update_truth(
            f"p{i}",
            np.atleast_1d(point_preds[i] + 0.1),
            version=1,
            event_time=2.0 + 0.1 * (i + 1),
        )

    # Fourth prediction now has three known samples.
    y_pred, lo, hi = rolling_cp.issue_prediction(
        "p3", X_new[3].reshape(1, -1), alpha=0.3, issue_time=3.0
    )
    np.testing.assert_allclose(y_pred, np.atleast_1d(point_preds[3]))
    assert np.all(lo <= y_pred) and np.all(hi >= y_pred)
    # Width is driven by the homogeneous 0.1 residuals.
    np.testing.assert_allclose(hi - lo, np.full_like(hi, 0.2), atol=1e-12)
