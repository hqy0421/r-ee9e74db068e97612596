Rolling Calibration
===================

Online regression ground truths are frequently delayed, corrected or
withdrawn. Accumulating residuals computed against stale ground truths would
silently corrupt the calibration. The
:class:`~deel.puncc.api.rolling.RollingConformalPredictor` manages calibration
**per prediction ID**, so that residuals always reflect the ground truth
currently considered valid.

The workflow is built on three rules:

1. The wrapped predictor is **fixed and already trained**. Only its
   ``predict`` method is ever called; it is never retrained, and old
   predictions are never re-run to manufacture residuals.
2. At issuance time, the actual point prediction is **saved** under the
   prediction ID. A published interval is immutable.
3. Ground truths are **versioned**. A correction must carry a strictly greater
   version and replaces the previous value; a withdrawal removes the ground
   truth from future calibrations. Old versions can never be resurrected.

Calibration samples for a new issuance are selected by a left-closed,
right-opened issuance-time window
``[issue_time - window, issue_time)`` restricted to ground truths that were
already processed (strictly earlier processing sequence number). Residuals
are recomputed on every issuance from the saved point predictions and the
currently valid ground truths; nothing is accumulated.

When too few samples are available, the interval is not published
(:class:`~deel.puncc.api.rolling.CalibrationUnavailable`) but the point
prediction is retained, so a ground truth received later can still be
backfilled and used by subsequent issuances.

.. code-block:: python

    import numpy as np

    from deel.puncc.api import nonconformity_scores, prediction_sets
    from deel.puncc.api.calibration import BaseCalibrator
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.api.rolling import (
        CalibrationUnavailable,
        RollingConformalPredictor,
    )

    # A fixed, already trained predictor (never retrained afterwards).
    predictor = BasePredictor(my_trained_model, is_trained=True)

    # Standard split-conformal calibrator.
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )

    rolling_cp = RollingConformalPredictor(
        predictor, calibrator, window=7.0, min_samples=10
    )

    # Issue a prediction. The first issuances, with no known ground truth,
    # raise CalibrationUnavailable but keep the point prediction.
    try:
        y_pred, y_lo, y_hi = rolling_cp.issue_prediction(
            "id-0", X=X_new, alpha=0.1, issue_time=100.0
        )
    except CalibrationUnavailable:
        y_pred = rolling_cp.get_point_prediction("id-0")

    # A delayed ground truth arrives, then a correction and a withdrawal.
    rolling_cp.update_truth("id-0", y_true, version=1, event_time=100.5)
    rolling_cp.update_truth(
        "id-0", y_corrected, version=2, event_time=101.0
    )
    rolling_cp.withdraw_truth("id-0", version=3, event_time=102.0)

    # Persist the full state and resume processing later.
    rolling_cp.save("rolling_state.pkl")
    rolling_cp = RollingConformalPredictor.load("rolling_state.pkl")

.. automodule:: deel.puncc.api.rolling
   :members:
   :undoc-members:
   :show-inheritance:
