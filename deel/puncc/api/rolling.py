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
"""
This module implements prediction-ID managed rolling calibration.

Online regression ground truths are frequently delayed, corrected or withdrawn.
Naively accumulating residuals computed against stale ground truths would
silently corrupt the calibration. The :class:`RollingConformalPredictor`
keeps, for every prediction ID:

* the **point prediction** computed once, at issuance time, by a fixed trained
  predictor (the predictor is never retrained);
* the **prediction interval**, published at most once and then immutable;
* the **current valid ground truth**, itself versioned, so that corrections
  replace older values and withdrawals remove them.

Residuals are never stored nor accumulated: every time a new interval is
issued, they are recomputed from the point predictions saved at issuance and
the ground truths that are *currently valid and were known at that issuance
time*. A past interval is therefore never recomputed and an old prediction is
never re-run to manufacture residuals.

Events are expected to be received in processing-time order. Every event is
tagged with a strictly increasing processing sequence number (and, optionally,
with an explicit event timestamp). Calibration samples are selected through a
left-closed, right-opened issuance-time window
``[issue_time - window, issue_time)`` restricted to ground truths already
processed (strictly earlier processing sequence number). When too few samples
are available, the interval is not published but the point prediction is
retained, so that a later ground truth can still be backfilled and used by
subsequent issuances.
"""

import hashlib
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api.backend import get_backend
from deel.puncc.api.calibration import BaseCalibrator

# Format tag of the serialized state. Bump on incompatible state changes.
_STATE_FORMAT = "rolling-cp-v1"


class RollingCalibrationError(Exception):
    """Base class for failures raised by the rolling calibration workflow."""


class PredictionIDConflict(RollingCalibrationError):
    """Raised when an existing prediction ID is re-issued with different
    request content (features or alpha).

    The same ID always returns the originally published result; an ID cannot be
    silently reused for a different request.
    """


class UnknownPredictionID(RollingCalibrationError, KeyError):
    """Raised when a ground truth event references a prediction ID that was
    never issued, or when a stored record is queried for an unknown ID.
    """


class StaleTruthVersion(RollingCalibrationError):
    """Raised when a ground truth update is older than, or equal to, the
    version already registered for the same prediction ID.

    Old ground truth versions cannot overwrite newer ones and, in particular,
    cannot be resurrected after a withdrawal or a correction.
    """


class TruthWithdrawn(RollingCalibrationError):
    """Raised when the ground truth of a prediction ID has been withdrawn and
    the caller attempts to read it.
    """


class EventOrderError(RollingCalibrationError):
    """Raised when events are not provided in processing-time order
    (non-increasing processing sequence number or event timestamp).
    """


class CalibrationUnavailable(RollingCalibrationError):
    """Raised when not enough currently valid, in-window ground truths are
    known at issuance time to calibrate an interval.

    The point prediction is nevertheless stored, without any interval, so that
    a ground truth received later can be backfilled.
    """


def _to_1d_key(prediction_id: Any) -> Any:
    """Normalize array-like prediction IDs (e.g. a numpy string) to hashable
    dictionary keys."""
    if isinstance(prediction_id, np.ndarray):
        if prediction_id.ndim == 0:
            return prediction_id.item()
        return tuple(prediction_id.tolist())
    return prediction_id


def _as_numpy(value: Any) -> np.ndarray:
    """Convert an input to a numpy array through the inferred backend."""
    return get_backend(value).to_numpy(get_backend(value).asarray(value))


def _request_fingerprint(X: Any, alpha: float) -> str:
    """Deterministic fingerprint of an issuance request.

    Two requests sharing the same prediction ID are considered identical only
    if their features and significance level are byte-for-byte equal.
    """
    X_np = _as_numpy(X)
    digest = hashlib.sha256()
    digest.update(repr((X_np.shape, X_np.dtype.str)).encode("utf-8"))
    digest.update(np.ascontiguousarray(X_np).tobytes())
    digest.update(repr(float(alpha)).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class _PredictionRecord:
    """Internal state associated with one prediction ID."""

    prediction_id: Any
    point_pred: np.ndarray
    issue_seq: int
    issue_time: float
    alpha: float
    fingerprint: str
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    @property
    def has_interval(self) -> bool:
        """Whether an immutable interval has been published for this ID."""
        return self.lower is not None and self.upper is not None


@dataclass
class _TruthRecord:
    """Versioned ground truth associated with one prediction ID."""

    value: Optional[np.ndarray]
    version: Any
    seq: int
    event_time: Optional[float]
    withdrawn: bool = False


class _RollingCalibrator(BaseCalibrator):
    """Internal calibrator used per issuance.

    The generic backend squeezes single-element arrays to 0D scalars, which
    would break :class:`BaseCalibrator`'s length-based bookkeeping on a
    one-sample rolling window. This subclass restores a 1D residual array
    after fitting, without changing any public calibration semantics.
    """

    def fit(self, *, y_true, y_pred, **kwargs):
        # Reimplemented (rather than wrapping super().fit) because the parent
        # implementation computes the calibration length before returning.
        del kwargs
        b = get_backend(y_pred)
        yp = b.asarray(y_pred)
        yt = b.asarray(y_true, like=yp)
        self._update_feature_axis(yp)
        self._residuals = np.atleast_1d(self.nonconf_score_func(yp, yt))
        self._len_calib = len(self._residuals)


def _build_issue_calibrator(template: BaseCalibrator) -> BaseCalibrator:
    """Build the calibrator used for one issuance.

    A standard :class:`BaseCalibrator` template is replaced by the internal
    single-sample-safe subclass carrying the same score / prediction-set
    functions. Any other (user-subclassed) calibrator is deep-copied as-is, so
    custom calibration behavior is fully preserved.
    """
    if type(template) is BaseCalibrator:
        return _RollingCalibrator(
            nonconf_score_func=template.nonconf_score_func,
            pred_set_func=template.pred_set_func,
            weight_func=template.weight_func,
        )
    return deepcopy(template)


class RollingConformalPredictor:
    """Prediction-ID managed rolling conformal predictor.

    The wrapped predictor is considered **fixed and already trained**: only its
    :meth:`predict` method is used, it is never fitted again. The wrapped
    calibrator is a standard :class:`BaseCalibrator` whose nonconformity score
    and prediction set functions define how intervals are built. A fresh
    (deep) copy of the calibrator is fitted on every issuance against the
    samples that are valid and known at that point in processing time.

    :param object predictor: a trained predictor exposing a ``predict(X)``
        method (e.g. a :class:`~deel.puncc.api.prediction.BasePredictor`).
    :param BaseCalibrator calibrator: calibrator defining the nonconformity
        score and prediction set construction. It is never mutated: each
        issuance operates on a deep copy.
    :param float window: length of the rolling issuance-time window, in the
        same unit as the event timestamps. Only ground truths whose issue time
        lies in the left-closed, right-opened interval
        ``[issue_time - window, issue_time)`` are calibration candidates.
        If ``None``, every ground truth known at issuance time is eligible.
    :param int min_samples: minimum number of valid, in-window calibration
        samples required to publish an interval. Defaults to 1.

    .. note::

        Ground truths are only eligible for the calibration of a prediction
        issued later if their processing sequence number is strictly smaller
        than the issuance sequence number. This encodes the "known at issuance
        time" constraint independently of timestamp ties.

    Example::

        from deel.puncc.api.rolling import RollingConformalPredictor
        from deel.puncc.api.calibration import BaseCalibrator
        from deel.puncc.api import nonconformity_scores, prediction_sets

        # Fixed trained predictor
        predictor = BasePredictor(my_trained_model, is_trained=True)

        # Standard split-conformal calibrator (absolute residual, constant
        # interval)
        calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_scores.absolute_difference,
            pred_set_func=prediction_sets.constant_interval,
        )

        rolling_cp = RollingConformalPredictor(
            predictor, calibrator, window=7.0, min_samples=10
        )

        # Issue a prediction (ground truths for older IDs may still be missing)
        y_pred, y_lo, y_hi = rolling_cp.issue_prediction(
            prediction_id="id-0", X=X_new, alpha=0.1, issue_time=100.0
        )

        # A delayed ground truth arrives, version 1
        rolling_cp.update_truth("id-0", y_true, version=1, event_time=100.5)

        # A correction replaces version 1; residuals of future issuances use
        # the new value, while already published intervals stay untouched
        rolling_cp.update_truth("id-0", y_corrected, version=2,
                                event_time=101.0)

        # A withdrawal removes the ground truth entirely
        rolling_cp.withdraw_truth("id-0", version=3, event_time=102.0)
    """

    def __init__(
        self,
        predictor: Any,
        calibrator: BaseCalibrator,
        *,
        window: Optional[float] = None,
        min_samples: int = 1,
    ):
        if not hasattr(predictor, "predict"):
            raise ValueError(
                "The provided predictor must expose a `predict` method."
            )

        if not isinstance(calibrator, BaseCalibrator):
            raise ValueError(
                "The rolling workflow requires a BaseCalibrator instance. "
                "Calibrators needing extra fit inputs (e.g. "
                "LeveragedCalibrator) are not supported, as only the point "
                "predictions and ground truths are retained per ID."
            )

        if window is not None:
            window = float(window)
            if not np.isfinite(window) or window <= 0:
                raise ValueError(
                    f"`window` must be a positive finite number or None, "
                    f"got {window}."
                )

        if not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
            raise ValueError(
                f"`min_samples` must be a positive integer, got {min_samples}."
            )

        self.predictor = predictor
        self.calibrator = calibrator
        self.window = window
        self.min_samples = int(min_samples)

        # Prediction records and versioned ground truths, keyed by prediction ID
        self._predictions: Dict[Any, _PredictionRecord] = {}
        self._truths: Dict[Any, _TruthRecord] = {}

        # Monotonic processing-time counters / timestamps
        self._last_seq: Optional[int] = None
        self._last_event_time: Optional[float] = None
        self._next_seq = 0

        # Counter of issued intervals, useful as a sanity/observability metric
        self._n_published_intervals = 0

    # ------------------------------------------------------------------
    # Processing-time ordering
    # ------------------------------------------------------------------

    def _tick(self, seq: Optional[int], event_time: Optional[float]) -> int:
        """Validate processing-time order and register the event.

        :param int seq: explicit processing sequence number. When ``None``,
            the internal monotonic counter is used.
        :param float event_time: optional event timestamp, which must be
            non-decreasing across events when provided.

        :returns: the resolved processing sequence number.
        :rtype: int
        """
        if seq is None:
            seq = self._next_seq
        elif not isinstance(seq, (int, np.integer)):
            raise ValueError(
                f"Processing sequence number must be an integer, got {seq}."
            )
        else:
            seq = int(seq)

        if self._last_seq is not None and seq <= self._last_seq:
            raise EventOrderError(
                f"Events must be received in processing-time order: sequence "
                f"{seq} is not strictly greater than the last processed "
                f"sequence {self._last_seq}."
            )

        if event_time is not None:
            event_time = float(event_time)
            if not np.isfinite(event_time):
                raise ValueError(
                    f"Event timestamps must be finite, got {event_time}."
                )
            if (
                self._last_event_time is not None
                and event_time < self._last_event_time
            ):
                raise EventOrderError(
                    f"Events must be received in processing-time order: event "
                    f"time {event_time} is earlier than the last processed "
                    f"event time {self._last_event_time}."
                )
            self._last_event_time = event_time

        self._last_seq = seq
        self._next_seq = max(self._next_seq, seq + 1)
        return seq

    # ------------------------------------------------------------------
    # Prediction issuance
    # ------------------------------------------------------------------

    def issue_prediction(
        self,
        prediction_id: Any,
        X: Any,
        alpha: float,
        *,
        issue_time: Optional[float] = None,
        seq: Optional[int] = None,
        predict_kwargs: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Issue (or replay) a point prediction and, when possible, its
        calibrated interval.

        The point prediction is computed once by the fixed predictor and saved.
        Calibration uses the ground truths currently valid, already processed
        (strictly earlier sequence number) and whose issue time falls in the
        left-closed, right-opened rolling window
        ``[issue_time - window, issue_time)``. The calibrator is fitted fresh
        on these samples; old predictions are never re-run.

        * If the ID is unknown, a new record is created. When too few samples
          are available, :class:`CalibrationUnavailable` is raised after storing
          the point prediction: no half record with an interval is published.
        * If the ID already exists and an interval was published, the same
          request returns the stored result unchanged. A different request
          content for the same ID raises :class:`PredictionIDConflict`.
        * If the ID exists but no interval could be published earlier, the
          calibration is retried against the ground truths known at the
          (retained) original issuance time.

        :param prediction_id: hashable identifier of the prediction.
        :param X: features of the new sample(s), as accepted by the
            predictor's ``predict`` method.
        :param float alpha: target maximum miscoverage.
        :param float issue_time: issuance timestamp, in the same unit as the
            other event timestamps. Defaults to the current internal clock
            value. Used only for window selection.
        :param int seq: optional explicit processing sequence number. When
            omitted, the internal monotonic counter is used.
        :param dict predict_kwargs: keyword arguments forwarded to the
            predictor's ``predict`` method.

        :returns: ``(y_pred, y_lower, y_upper)``. When calibration is
            unavailable for a new ID, the call raises
            :class:`CalibrationUnavailable`; use :meth:`get_point_prediction`
            to retrieve the retained point prediction.
        :rtype: Tuple[ndarray, Optional[ndarray], Optional[ndarray]]

        :raises PredictionIDConflict: same ID, different request content.
        :raises CalibrationUnavailable: not enough valid in-window samples.
        """
        prediction_id = _to_1d_key(prediction_id)
        predict_kwargs = predict_kwargs or {}

        if issue_time is None:
            issue_time = (
                float(self._next_seq)
                if self._last_event_time is None
                else self._last_event_time
            )
        else:
            issue_time = float(issue_time)
            if not np.isfinite(issue_time):
                raise ValueError(
                    f"Issue time must be finite, got {issue_time}."
                )

        existing = self._predictions.get(prediction_id)

        if existing is not None:
            return self._replay_existing(existing, X=X, alpha=alpha)

        # Register the event ordering *before* doing any work, so a failed
        # issuance still advances the processing clock consistently.
        resolved_seq = self._tick(seq, issue_time)
        fingerprint = _request_fingerprint(X, alpha)

        # Point prediction, computed once and saved at issuance time. Scalar
        # predictions are normalized to a 1-element 1D array so downstream
        # stacking and the calibrator's length-based checks stay well-defined.
        point_pred = np.atleast_1d(
            _as_numpy(self.predictor.predict(X, **predict_kwargs))
        )

        record = _PredictionRecord(
            prediction_id=prediction_id,
            point_pred=point_pred,
            issue_seq=resolved_seq,
            issue_time=issue_time,
            alpha=float(alpha),
            fingerprint=fingerprint,
        )
        # Provisionally store the point-only record. It is removed on
        # unexpected failures but retained on the expected
        # CalibrationUnavailable outcome.
        self._predictions[prediction_id] = record

        try:
            lower, upper = self._calibrate_record(record)
        except CalibrationUnavailable:
            # Expected outcome: keep the point prediction for later backfill.
            raise
        except Exception:
            # Any other failure: do not publish a half record.
            self._predictions.pop(prediction_id, None)
            raise

        record.lower = lower
        record.upper = upper
        self._n_published_intervals += 1
        return record.point_pred, record.lower, record.upper

    def _replay_existing(
        self,
        record: _PredictionRecord,
        *,
        X: Any,
        alpha: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Handle an issuance call targeting an already known prediction ID."""
        fingerprint = _request_fingerprint(X, alpha)
        if fingerprint != record.fingerprint:
            raise PredictionIDConflict(
                f"Prediction ID {record.prediction_id!r} was already issued "
                f"with different request content (features or alpha). "
                f"Issued intervals are immutable; use a new prediction ID."
            )

        if record.has_interval:
            # Published intervals are frozen: identical request returns the
            # original result.
            return record.point_pred, record.lower, record.upper

        # Point-only record: retry calibration, still against the ground truths
        # known at the original issuance time (no future leakage, no re-run of
        # the predictor).
        lower, upper = self._calibrate_record(record)
        record.lower = lower
        record.upper = upper
        self._n_published_intervals += 1
        return record.point_pred, record.lower, record.upper

    # ------------------------------------------------------------------
    # Ground truth management
    # ------------------------------------------------------------------

    def update_truth(
        self,
        prediction_id: Any,
        y_true: Any,
        version: Any,
        *,
        event_time: Optional[float] = None,
        seq: Optional[int] = None,
    ) -> None:
        """Register or correct a ground truth for a prediction ID.

        The ground truth is versioned: a new version must be strictly greater
        than the current one. Corrections replace the previous value, which
        therefore cannot be resurrected by a late, older event. Withdrawn
        ground truths can be restored only by a strictly greater version.

        :param prediction_id: identifier of the previously issued prediction.
        :param y_true: ground truth value aligned with the saved point
            prediction.
        :param version: monotonically increasing ground truth version
            (numbers, strings or any strictly orderable values).
        :param float event_time: optional timestamp of the event.
        :param int seq: optional explicit processing sequence number.

        :raises UnknownPredictionID: the ID was never issued.
        :raises StaleTruthVersion: the version is not newer than the current
            one, including attempts to resurrect an old value after a
            withdrawal or correction.
        """
        prediction_id = _to_1d_key(prediction_id)
        if prediction_id not in self._predictions:
            raise UnknownPredictionID(
                f"Cannot register a ground truth for unknown prediction ID "
                f"{prediction_id!r}. Issue the prediction first."
            )

        resolved_seq = self._tick(seq, event_time)
        self._apply_truth(
            prediction_id,
            value=_as_numpy(y_true),
            version=version,
            seq=resolved_seq,
            event_time=event_time,
            withdrawn=False,
        )

    def withdraw_truth(
        self,
        prediction_id: Any,
        version: Any,
        *,
        event_time: Optional[float] = None,
        seq: Optional[int] = None,
    ) -> None:
        """Withdraw (revoke) the ground truth associated with a prediction ID.

        A withdrawn ground truth stops being a calibration candidate for all
        subsequent issuances. A later, strictly greater version may still
        restore it through :meth:`update_truth`; older versions can never be
        resurrected.

        :param prediction_id: identifier of the prediction.
        :param version: version of the withdrawal event, strictly greater than
            the current ground truth version.
        :param float event_time: optional timestamp of the event.
        :param int seq: optional explicit processing sequence number.

        :raises UnknownPredictionID: the ID was never issued.
        :raises StaleTruthVersion: the withdrawal is not newer than the
            current version.
        """
        prediction_id = _to_1d_key(prediction_id)
        if prediction_id not in self._predictions:
            raise UnknownPredictionID(
                f"Cannot withdraw a ground truth for unknown prediction ID "
                f"{prediction_id!r}. Issue the prediction first."
            )

        resolved_seq = self._tick(seq, event_time)
        self._apply_truth(
            prediction_id,
            value=None,
            version=version,
            seq=resolved_seq,
            event_time=event_time,
            withdrawn=True,
        )

    def _apply_truth(
        self,
        prediction_id: Any,
        *,
        value: Optional[np.ndarray],
        version: Any,
        seq: int,
        event_time: Optional[float],
        withdrawn: bool,
    ) -> None:
        """Version-check and store a ground truth update or withdrawal."""
        current = self._truths.get(prediction_id)

        if current is not None:
            try:
                is_newer = version > current.version
            except TypeError as exc:
                raise StaleTruthVersion(
                    f"Ground truth versions for ID {prediction_id!r} are not "
                    f"comparable: new {version!r} vs current "
                    f"{current.version!r}."
                ) from exc
            if not is_newer:
                raise StaleTruthVersion(
                    f"Ground truth version {version!r} for ID "
                    f"{prediction_id!r} is not newer than the current version "
                    f"{current.version!r}. Old versions cannot overwrite "
                    f"newer ones or be resurrected after a withdrawal."
                )

        self._truths[prediction_id] = _TruthRecord(
            value=value,
            version=version,
            seq=seq,
            event_time=event_time,
            withdrawn=withdrawn,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _eligible_calibration_ids(self, record: _PredictionRecord) -> List[Any]:
        """Return the prediction IDs eligible to calibrate ``record``.

        Eligibility combines:

        * a currently valid (present, non-withdrawn) ground truth;
        * strict processing-time precedence (the ground truth event was
          processed before the issuance);
        * the left-closed, right-opened issuance-time window on the candidate
          prediction's own issue time.
        """
        eligible = []
        for other_id, other in self._predictions.items():
            if other_id == record.prediction_id:
                continue

            truth = self._truths.get(other_id)
            if truth is None or truth.withdrawn or truth.value is None:
                continue

            # "Known at issuance time": the ground truth event must have been
            # processed strictly before the issuance.
            if truth.seq >= record.issue_seq:
                continue

            # Left-closed, right-opened rolling window on issue times.
            if self.window is not None:
                in_window = (
                    record.issue_time - self.window
                    <= other.issue_time
                    < record.issue_time
                )
                if not in_window:
                    continue

            eligible.append(other_id)
        return eligible

    def _calibrate_record(
        self, record: _PredictionRecord
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a fresh calibrator on the currently valid in-window samples and
        build the interval for the saved point prediction.

        Residuals are recomputed from saved point predictions and current
        ground truths: nothing is accumulated and no old predictor call is
        made.
        """
        eligible_ids = self._eligible_calibration_ids(record)

        if len(eligible_ids) < self.min_samples:
            raise CalibrationUnavailable(
                f"Only {len(eligible_ids)} valid calibration sample(s) are "
                f"known at issuance time for ID {record.prediction_id!r}, "
                f"but min_samples={self.min_samples} is required. The point "
                f"prediction is retained; no interval is published."
            )

        y_pred_list = [
            self._predictions[other_id].point_pred for other_id in eligible_ids
        ]
        y_true_list = [
            self._truths[other_id].value for other_id in eligible_ids
        ]
        y_pred_calib = np.stack(
            [
                np.atleast_1d(np.asarray(p, dtype=np.float64)).reshape(-1)
                for p in y_pred_list
            ],
            axis=0,
        )
        y_true_calib = np.stack(
            [
                np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1)
                for t in y_true_list
            ],
            axis=0,
        )
        # Scalar targets produce a (n_samples,) calibration set, matching the
        # shape expected by BaseCalibrator's nonconformity functions.
        if y_pred_calib.shape[1] == 1:
            y_pred_calib = y_pred_calib.reshape(-1)
            y_true_calib = y_true_calib.reshape(-1)

        # A fresh calibrator keeps each issuance independent and leaves the
        # user-provided template calibrator untouched.
        calibrator = _build_issue_calibrator(self.calibrator)
        calibrator.fit(y_true=y_true_calib, y_pred=y_pred_calib)
        lower, upper = calibrator.calibrate(
            y_pred=record.point_pred, alpha=record.alpha
        )
        return _as_numpy(lower), _as_numpy(upper)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def get_point_prediction(self, prediction_id: Any) -> np.ndarray:
        """Return the point prediction saved at issuance time for an ID.

        :raises UnknownPredictionID: the ID was never issued.
        """
        prediction_id = _to_1d_key(prediction_id)
        self._require_record(prediction_id)
        return self._predictions[prediction_id].point_pred

    def get_interval(self, prediction_id: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return the immutable interval published for an ID.

        :raises UnknownPredictionID: the ID was never issued.
        :raises CalibrationUnavailable: no interval was published for the ID.
        """
        prediction_id = _to_1d_key(prediction_id)
        self._require_record(prediction_id)
        record = self._predictions[prediction_id]
        if not record.has_interval:
            raise CalibrationUnavailable(
                f"No interval has been published for ID {prediction_id!r}. "
                f"The point prediction is available via "
                f"`get_point_prediction`."
            )
        return record.lower, record.upper

    def get_truth(self, prediction_id: Any) -> np.ndarray:
        """Return the currently valid ground truth value for an ID.

        :raises UnknownPredictionID: no ground truth is registered for the ID.
        :raises TruthWithdrawn: the ground truth is currently withdrawn.
        """
        prediction_id = _to_1d_key(prediction_id)
        truth = self._truths.get(prediction_id)
        if truth is None:
            raise UnknownPredictionID(
                f"No ground truth registered for ID {prediction_id!r}."
            )
        if truth.withdrawn:
            raise TruthWithdrawn(
                f"Ground truth for ID {prediction_id!r} is withdrawn."
            )
        return truth.value

    def is_interval_available(self, prediction_id: Any) -> bool:
        """Return ``True`` if an interval was published for the ID."""
        prediction_id = _to_1d_key(prediction_id)
        record = self._predictions.get(prediction_id)
        return record is not None and record.has_interval

    def is_truth_withdrawn(self, prediction_id: Any) -> bool:
        """Return ``True`` if the current ground truth for the ID is withdrawn.
        Unknown IDs return ``False``.
        """
        prediction_id = _to_1d_key(prediction_id)
        truth = self._truths.get(prediction_id)
        return truth is not None and truth.withdrawn

    def get_status(self, prediction_id: Any) -> dict:
        """Return a snapshot of the state associated with a prediction ID.

        The snapshot contains the issuance sequence and time, the alpha level,
        whether an interval is available, the current ground truth version (if
        any) and the withdrawal flag.

        :raises UnknownPredictionID: the ID was never issued.
        """
        prediction_id = _to_1d_key(prediction_id)
        self._require_record(prediction_id)
        record = self._predictions[prediction_id]
        truth = self._truths.get(prediction_id)
        return {
            "prediction_id": prediction_id,
            "issue_seq": record.issue_seq,
            "issue_time": record.issue_time,
            "alpha": record.alpha,
            "has_interval": record.has_interval,
            "truth_version": None if truth is None else truth.version,
            "truth_withdrawn": truth is not None and truth.withdrawn,
        }

    def _require_record(self, prediction_id: Any) -> None:
        if prediction_id not in self._predictions:
            raise UnknownPredictionID(
                f"Unknown prediction ID {prediction_id!r}."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the full rolling state to ``path``.

        The fixed predictor, the template calibrator, every saved point
        prediction, published interval, versioned ground truth and the
        processing clocks are persisted. The write is atomic: a failure while
        serializing cannot leave a truncated file behind.

        :param str path: destination file path.
        """
        state = {
            "format": _STATE_FORMAT,
            "window": self.window,
            "min_samples": self.min_samples,
            "predictor": self.predictor,
            "calibrator": self.calibrator,
            "predictions": self._predictions,
            "truths": self._truths,
            "last_seq": self._last_seq,
            "last_event_time": self._last_event_time,
            "next_seq": self._next_seq,
            "n_published_intervals": self._n_published_intervals,
        }
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as output_file:
                pickle.dump(state, output_file)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def load(path: str) -> "RollingConformalPredictor":
        """Load a previously saved rolling predictor from ``path``.

        Processing can resume directly: event ordering, saved point
        predictions, published intervals and versioned ground truths are all
        restored.

        :param str path: file path produced by :meth:`save`.
        :rtype: RollingConformalPredictor
        """
        with open(path, "rb") as input_file:
            state = pickle.load(input_file)

        if not isinstance(state, dict) or state.get("format") != _STATE_FORMAT:
            raise RollingCalibrationError(
                "The provided file does not contain a rolling predictor state "
                f"in format {_STATE_FORMAT!r}."
            )

        loaded = object.__new__(RollingConformalPredictor)
        loaded.predictor = state["predictor"]
        loaded.calibrator = state["calibrator"]
        loaded.window = state["window"]
        loaded.min_samples = state["min_samples"]
        loaded._predictions = state["predictions"]
        loaded._truths = state["truths"]
        loaded._last_seq = state["last_seq"]
        loaded._last_event_time = state["last_event_time"]
        loaded._next_seq = state["next_seq"]
        loaded._n_published_intervals = state["n_published_intervals"]
        return loaded
