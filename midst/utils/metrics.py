from sklearn import metrics
from abc import abstractmethod
from scipy.signal import find_peaks
from typing import Dict, Any, Optional, Sequence

import numpy as np


class Metric:
    """
    Base class for metrics
    """

    def __init__(self, horizon: int, trajectory_length: int):
        self.horizon = horizon
        self.trajectory_length = trajectory_length

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        pass


class PointWiseMetric(Metric):
    @abstractmethod
    def compute_point_wise_metric(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        pass

    @staticmethod
    def detect_extremes(y: np.ndarray, distance: int) -> Sequence[Sequence[Sequence[np.ndarray]]]:
        signals = [
            [
                [
                    y[:, m, t, n]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        max_peaks = [
            [
                [
                    find_peaks(
                        signals[m][t][n],
                        height=(np.mean(signals[m][t][n]) + np.std(signals[m][t][n])),
                        distance=distance,
                    )[0]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        min_peaks = [
            [
                [
                    find_peaks(
                        (1 / signals[m][t][n]),
                        height=(1 / (np.mean(signals[m][t][n]) + np.std(signals[m][t][n]))),
                        distance=distance,
                    )[0]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        peaks_inds = [
            [
                [
                    np.sort(np.unique(np.concatenate([min_peaks[m][t][n], max_peaks[m][t][n]])))
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        min_n_peaks = [
            [
                [
                    len(peaks_inds[m][t][n])
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        min_n_peaks = min(
            [
                min(
                    [
                        min(min_n_peaks[m][t])
                        for t in range(y.shape[2])
                    ]
                )
                for m in range(y.shape[1])
            ]
        )
        peaks_inds = [
            [
                [
                    peaks_inds[m][t][n][:min_n_peaks]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        return peaks_inds

    def __call__(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        inputs = self._prepare_inputs(x=x, y=y, y_pred=y_pred)
        system_aggregated_results = self.aggregate_metric_across_systems(inputs=inputs)
        results = self.compute_point_wise_metric(inputs=system_aggregated_results)

        return results


class ScikitMASE(PointWiseMetric):
    def __init__(
            self,
            trajectory_length: int,
            m: int = 1,
            states: Optional[Sequence[int]] = None,
            threshold: Optional[float] = None,
            prediction_ind: Optional[int] = None,
            apply_by_system: bool = True,
    ):
        super(ScikitMASE, self).__init__(
            horizon=m,
            trajectory_length=trajectory_length,
        )

        self._threshold = threshold
        self._states = states
        self._prediction_ind = prediction_ind
        self._apply_by_system = apply_by_system

    def _apply_threshold(
            self,
            predictions: np.ndarray,
            metric: np.ndarray,
            apply_by_system: bool = True,
    ) -> (np.ndarray, np.ndarray):
        axis = 0 if apply_by_system else 2
        if self._threshold is not None:
            threshed_metric = metric.copy()
            threshed_metric = [
                np.delete(
                    (
                        np.reshape(threshed_metric[i], -1)
                        if apply_by_system else
                        np.reshape(threshed_metric[..., i, :], -1)
                    ),
                    np.where(
                        (
                                (np.abs(np.reshape(predictions[i], (-1,)) < self._threshold)) == True
                        )
                    )[0])
                for i in range(metric.shape[axis])
            ]
            n_removed = [
                (
                        (
                            metric[i].size
                            if apply_by_system else
                            metric[..., i, :].size
                        ) -
                        threshed_metric[i].size
                )
                for i in range(metric.shape[axis])
            ]
            n_removed = np.array(n_removed)

        else:
            threshed_metric = [
                np.reshape(metric[i], (-1,))
                for i in range(metric.shape[0])
            ]
            n_removed = np.array([0] * metric.shape[0])

        return threshed_metric, n_removed

    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        # For handling Hankel-based results
        if len(x.shape) == 5:
            x = np.swapaxes(x[..., -1, :], 2, 3)

        # Take only the relevant predictions from y
        assert y.shape == y_pred.shape

        if y.shape[2] != self.horizon:
            y = y[..., -self.horizon:, :]
            y_pred = y_pred[..., -self.horizon:, :]

        if self._prediction_ind is not None:
            y = y[..., [self._prediction_ind], :]
            y_pred = y_pred[..., [self._prediction_ind], :]

        if self._states is not None:
            x = x[..., self._states]
            y = y[..., self._states]
            y_pred = y_pred[..., self._states]

        out = {
            'x': x,
            'y': y,
            'y_pred': y_pred
        }

        return out

    def compute_point_wise_metric(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        metric = inputs['metric']
        valid_metric = metric[~np.isnan(metric)]
        metric_mean = np.mean(valid_metric)
        metric_std = np.std(valid_metric)

        out = inputs.copy()
        out['metric_mean'] = metric_mean
        out['metric_std'] = metric_std

        return out

    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        x = inputs['x']
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute prediction & naive errors
        x_ind = -self.horizon if x.shape[2] >= self.horizon else -1
        naive_predictions = np.repeat(a=x[:, :, x_ind, :][:, :, None, :], repeats=y.shape[2], axis=2)
        naive_predictions_errors = np.abs((y - naive_predictions))
        predictions_errors = np.abs((y - y_pred))

        # Apply threshold if required
        naive_predictions_errors, naive_n_removed = self._apply_threshold(
            predictions=y,
            metric=naive_predictions_errors,
            apply_by_system=self._apply_by_system,
        )
        predictions_errors, n_removed = self._apply_threshold(
            predictions=y,
            metric=predictions_errors,
            apply_by_system=self._apply_by_system,
        )

        naive_predictions_errors = [
            np.mean(naive_predictions_errors[i])
            if len(naive_predictions_errors[i]) else
            np.nan
            for i in range(len(naive_predictions_errors))
        ]
        naive_predictions_errors = np.array(naive_predictions_errors)
        predictions_errors = [
            np.mean(predictions_errors[i])
            if len(predictions_errors[i]) else
            np.nan
            for i in range(len(predictions_errors))
        ]
        predictions_errors = np.array(predictions_errors)

        # Compute system-wise metric
        mase = predictions_errors / naive_predictions_errors

        # Remove the NaN values (which might exists due to thresholding
        mase = mase[~np.isnan(mase)]

        out = {
            'metric': mase,
            'n_removed': np.sum(n_removed),
            'n_samples': y_pred.size,
        }

        return out

    def __call__(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        inputs = self._prepare_inputs(x=x, y=y, y_pred=y_pred)
        system_aggregated_results = self.aggregate_metric_across_systems(inputs=inputs)
        results = self.compute_point_wise_metric(inputs=system_aggregated_results)

        return results


class ScikitPeaksMASE(ScikitMASE):
    def __init__(
            self,
            trajectory_length: int,
            m: int = 1,
            states: Optional[Sequence[int]] = None,
            threshold: Optional[float] = None,
            prediction_ind: Optional[int] = None,
            peaks_distance: int = 1,
    ):
        super(ScikitPeaksMASE, self).__init__(
            trajectory_length=trajectory_length,
            m=m,
            states=states,
            threshold=threshold,
            prediction_ind=prediction_ind,
        )

        self._peaks_distance = peaks_distance

    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        # For handling Hankel-based results
        if len(x.shape) == 5:
            x = np.swapaxes(x[..., -1, :], 2, 3)

        # Take only the relevant predictions from y
        assert y.shape == y_pred.shape

        if y.shape[2] != self.horizon:
            y = y[..., -self.horizon:, :]
            y_pred = y_pred[..., -self.horizon:, :]

        if self._prediction_ind is not None:
            y = y[..., [self._prediction_ind], :]
            y_pred = y_pred[..., [self._prediction_ind], :]

        if self._states is not None:
            x = x[..., self._states]
            y = y[..., self._states]
            y_pred = y_pred[..., self._states]

        peaks_inds = self.detect_extremes(y, distance=self._peaks_distance)

        out = {
            'x': x,
            'y': y,
            'y_pred': y_pred,
            'peaks_inds': peaks_inds,
        }

        return out

    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']
        peaks_inds = inputs['peaks_inds']

        # Compute prediction & naive errors
        y_peaks = [
            [
                [
                    y[:, m, -t, n][peaks_inds[m][t][n]]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        y_peaks = np.concatenate(
            [
                np.concatenate([p[:, None] for p in peaks[0]], axis=1)[None, :, :]
                for peaks in y_peaks
            ],
            axis=0,
        )

        naive_predictions_errors = np.concatenate(
            [
                np.abs((y_peaks[:, 2::2, :] - y_peaks[:, :-2:2, :])),
                np.abs((y_peaks[:, 3::2, :] - y_peaks[:, 1:-1:2, :])),
            ],
            axis=1,
        )
        naive_predictions_errors = np.swapaxes(naive_predictions_errors, axis1=0, axis2=1)
        naive_predictions_errors = np.mean(naive_predictions_errors, axis=(1, 2))

        predictions_errors = np.abs((y - y_pred))
        predictions_errors = [
            [
                [
                    predictions_errors[:, m, t, n][peaks_inds[m][t][n]]
                    for n in range(y.shape[3])
                ]
                for t in range(y.shape[2])
            ]
            for m in range(y.shape[1])
        ]
        predictions_errors_peaks = np.concatenate(
            [
                np.concatenate([p[:, None] for p in peaks[0]], axis=1)[None, :, :]
                for peaks in predictions_errors
            ],
            axis=0,
        )
        predictions_errors_peaks = np.swapaxes(predictions_errors_peaks, axis1=0, axis2=1)
        predictions_errors_peaks = np.mean(predictions_errors_peaks, axis=(1, 2))[2:]

        # Compute system-wise metric
        mase = predictions_errors_peaks / naive_predictions_errors

        out = {
            'metric': mase,
            'n_removed': 0,
            'n_samples': y_pred.size,
        }

        return out


class ScikitPeaksScaledMSE(ScikitPeaksMASE):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']
        peaks_inds = inputs['peaks_inds']

        # Compute system-wise metric
        smse = np.concatenate(
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [
                                (
                                        np.sqrt(
                                            np.power(
                                                (y[peaks_inds[m][t][n], m, t, n] - y_pred[
                                                    peaks_inds[m][t][n], m, t, n]),
                                                2,
                                            )
                                        )
                                        /
                                        np.abs(y[peaks_inds[m][t][n], m, t, n])
                                )[:, None] * 100
                                for n in range(y.shape[3])
                            ], axis=1
                        )[:, None, ...]
                        for t in range(y.shape[2])
                    ], axis=1
                )[:, None, ...]
                for m in range(y.shape[1])
            ], axis=1
        )
        mase = smse.mean(axis=(1, 2, 3))

        out = {
            'metric': mase,
            'n_removed': 0,
            'n_samples': y_pred.size,
        }

        return out


class ScikitMSE(ScikitMASE):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']
        n_samples = y.size

        # Apply threshold if required
        y, _ = self._apply_threshold(
            predictions=y,
            metric=y,
        )
        y_pred, n_removed = self._apply_threshold(
            predictions=y,
            metric=y_pred,
        )

        # Compute mse
        mse = [
            metrics.mean_squared_error(
                y_true=np.reshape(y[i], (-1,)),
                y_pred=np.reshape(y_pred[i], (-1,)),
            )
            for i in range(len(y))
            if len(y_pred[i]) > 0
        ]
        mse = np.array(mse)

        out = {
            'metric': mse,
            'n_removed': np.sum(n_removed),
            'n_samples': n_samples,
            'deleted_indices': [],
        }

        return out


class ScikitScaledMSE(ScikitMSE):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']
        n_samples = y.size
        norms = np.linalg.norm(y, axis=-1)[..., None]

        # Compute mse
        mse = [
            (np.power((y[i] - y_pred[i]), 2) / np.power(norms[i], 2)).mean() * 100
            for i in range(len(y))
        ]
        mse = np.array(mse)

        out = {
            'metric': mse,
            'n_removed': 0,
            'n_samples': n_samples,
            'deleted_indices': [],
        }

        return out


class ScikitScaledMAE(ScikitMSE):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']
        n_samples = y.size

        # Compute mse
        smae = [
            (np.abs((y[i] - y_pred[i])) / np.abs(y[i])).mean() * 100
            for i in range(len(y))
        ]
        smae = np.array(smae)

        out = {
            'metric': smae,
            'n_removed': 0,
            'n_samples': n_samples,
            'deleted_indices': [],
        }

        return out


class ScikitQTcMase(ScikitMASE):
    def __init__(
            self,
            qt_interval_ind: int,
            rr_interval_ind: int,
            prediction_horizon: int = 1,
            threshold: Optional[float] = None,
            trajectory_length: int = 6,
            framingham_correction: bool = True,
            point_wise: bool = True,
            apply_by_system: bool = True,
    ):
        super(ScikitQTcMase, self).__init__(
            trajectory_length=trajectory_length,
            m=prediction_horizon,
            threshold=threshold,
            apply_by_system=apply_by_system,
        )

        self._qt_interval_ind = qt_interval_ind
        self._rr_interval_ind = rr_interval_ind
        self._framingham_correction = framingham_correction
        self._point_wise = point_wise

    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        inputs = super()._prepare_inputs(
            x=x,
            y=y,
            y_pred=y_pred,
        )

        x = inputs['x']
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Unpack predictions if they are in a "single-system" format
        if y.shape[1] == 1:
            x = x.swapaxes(2, 3)
            x = x.reshape(x.shape[0], 5, -1, x.shape[3])
            x = x.swapaxes(2, 3)

            y = y.swapaxes(2, 3)
            y = y.reshape(y.shape[0], 5, -1, y.shape[3])
            y = y.swapaxes(2, 3)

            y_pred = y_pred.swapaxes(2, 3)
            y_pred = y_pred.reshape(y_pred.shape[0], 5, -1, y_pred.shape[3])
            y_pred = y_pred.swapaxes(2, 3)

        x_rr = x[:, :, -self.horizon:, self._rr_interval_ind][..., None]
        y_rr = y[:, :, -self.horizon:, self._rr_interval_ind][..., None]
        y_pred_rr = y_pred[:, :, -self.horizon:, self._rr_interval_ind][..., None]
        x_qt = x[:, :, -self.horizon:, self._qt_interval_ind][..., None]
        y_qt = y[:, :, -self.horizon:, self._qt_interval_ind][..., None]
        y_pred_qt = y_pred[:, :, -self.horizon:, self._qt_interval_ind][..., None]

        if self._framingham_correction:
            x_qtc = x_qt + (0.154 * (1 - x_rr))
            y_qtc = y_qt + (0.154 * (1 - y_rr))
            y_pred_qtc = y_pred_qt + (0.154 * (1 - y_pred_rr))

        else:
            x_qtc = x_qt / np.sqrt(x_rr)
            y_qtc = y_qt / np.sqrt(y_rr)
            y_pred_qtc = y_pred_qt / np.sqrt(np.abs(y_pred_rr))

        out = {
            'x': x_qtc,
            'y': y_qtc,
            'y_pred': y_pred_qtc
        }

        return out


class ExtractQTcData(ScikitQTcMase):
    def __init__(
            self,
            qt_interval_ind: int,
            rr_interval_ind: int,
            prediction_horizon: int = 1,
            threshold: Optional[float] = None,
            trajectory_length: int = 6,
            framingham_correction: bool = True,
            return_gt_y: bool = False,
    ):
        super(ExtractQTcData, self).__init__(
            qt_interval_ind=qt_interval_ind,
            rr_interval_ind=rr_interval_ind,
            prediction_horizon=prediction_horizon,
            threshold=threshold,
            trajectory_length=trajectory_length,
            framingham_correction=framingham_correction,
        )

        self._return_gt_y = return_gt_y

    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        if  self._return_gt_y:
            return_data = y

        else:
            return_data = y_pred

        out = {
            'metric': return_data,
            'n_removed': 0,
            'n_samples': y_pred.size,
        }

        return out


class ScikitQTcAccuracy(ScikitQTcMase):
    def _find_negative_predictions(self, predictions: np.ndarray) -> np.ndarray:
        negative_predictions = (predictions < (self._threshold if self._threshold is not None else 0)).astype(np.int32)
        return negative_predictions

    def _find_positives_predictions(self, predictions: np.ndarray) -> np.ndarray:
        positive_predictions = (predictions >= (self._threshold if self._threshold is not None else 0)).astype(np.int32)
        return positive_predictions

    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute accuracy
        binary_y_qtc = self._find_positives_predictions(y)
        binary_y_pred_qtc = self._find_positives_predictions(y_pred)

        if self._point_wise:
            accuracy = [
                metrics.accuracy_score(
                    y_true=np.reshape(binary_y_qtc[i], (-1,)),
                    y_pred=np.reshape(binary_y_pred_qtc[i], (-1,)),
                )
                for i in range(binary_y_qtc.shape[0])
            ]

        else:
            accuracy = [
                metrics.accuracy_score(
                    y_true=binary_y_qtc[i].max(axis=1)[..., 0],
                    y_pred=binary_y_pred_qtc[i].max(axis=1)[..., 0],
                )
                for i in range(binary_y_qtc.shape[0])
            ]

        accuracy = np.array(accuracy)
        out = {
            'metric': accuracy,
            'n_removed': 0,
            'n_samples': y_pred.size,
        }

        return out


class ScikitQTcPrecision(ScikitQTcAccuracy):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute precision
        binary_y_qtc = self._find_positives_predictions(y)
        binary_y_pred_qtc = self._find_positives_predictions(y_pred)

        if self._point_wise:
            precision = [
                metrics.precision_score(
                    y_true=np.reshape(binary_y_qtc[i], (-1,)),
                    y_pred=np.reshape(binary_y_pred_qtc[i], (-1,)),
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        else:
            precision = [
                metrics.precision_score(
                    y_true=binary_y_qtc[i].max(axis=1)[..., 0],
                    y_pred=binary_y_pred_qtc[i].max(axis=1)[..., 0],
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        precision = np.array(precision)
        out = {
            'metric': precision,
            'n_removed': np.zeros_like(precision),
            'n_samples': y_pred.size,
            'deleted_indices': [],
        }

        return out


class ScikitQTcRecall(ScikitQTcAccuracy):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute precision
        binary_y_qtc = self._find_positives_predictions(y)
        binary_y_pred_qtc = self._find_positives_predictions(y_pred)

        if self._point_wise:
            recall = [
                metrics.recall_score(
                    y_true=np.reshape(binary_y_qtc[i], (-1,)),
                    y_pred=np.reshape(binary_y_pred_qtc[i], (-1,)),
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        else:
            recall = [
                metrics.recall_score(
                    y_true=binary_y_qtc[i].max(axis=1)[..., 0],
                    y_pred=binary_y_pred_qtc[i].max(axis=1)[..., 0],
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        recall = np.array(recall)

        out = {
            'metric': recall,
            'n_removed': np.zeros_like(recall),
            'n_samples': y_pred.size,
            'deleted_indices': [],
        }

        return out


class ScikitQTcSpecificity(ScikitQTcAccuracy):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute precision
        binary_y_qtc = self._find_positives_predictions(y)
        binary_y_pred_qtc = self._find_positives_predictions(y_pred)

        # Scikit's confusion_matrix(y, y_pred).ravel() returns a tuple of size 4, with the following elements:
        # tn, fp, fn, tp, where specificity = tn / (tn + fp)
        if self._point_wise:
            specificity = [
                metrics.confusion_matrix(
                    y_true=np.reshape(binary_y_qtc[i], (-1,)),
                    y_pred=np.reshape(binary_y_pred_qtc[i], (-1,)),
                    normalize='all',
                ).ravel()
                for i in range(binary_y_qtc.shape[0])
            ]

        else:
            specificity = [
                metrics.confusion_matrix(
                    y_true=binary_y_qtc[i].max(axis=1)[..., 0],
                    y_pred=binary_y_pred_qtc[i].max(axis=1)[..., 0],
                    normalize='all',
                ).ravel()
                for i in range(binary_y_qtc.shape[0])
            ]

        for i, sp in enumerate(specificity):
            if len(sp) < 4:
                tn = np.sum(((binary_y_qtc == 0) * (binary_y_pred_qtc == 0))) / binary_y_qtc.size
                fp = np.sum(((binary_y_qtc == 0) * (binary_y_pred_qtc == 1))) / binary_y_qtc.size
                fn = np.sum(((binary_y_qtc == 1) * (binary_y_pred_qtc == 0))) / binary_y_qtc.size
                tp = np.sum(((binary_y_qtc == 1) * (binary_y_pred_qtc == 1))) / binary_y_qtc.size
                specificity[i] = (tn, fp, fn, tp)

        specificity = [
            (sp[0] / (sp[0] + sp[1]))
            for sp in specificity
        ]
        specificity = np.array(specificity)
        out = {
            'metric': specificity,
            'n_removed': np.zeros_like(specificity),
            'n_samples': y_pred.size,
            'deleted_indices': [],
        }
        return out


class ScikitQTcF1(ScikitQTcAccuracy):
    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Compute precision
        binary_y_qtc = self._find_positives_predictions(y)
        binary_y_pred_qtc = self._find_positives_predictions(y_pred)

        if self._point_wise:
            f1 = [
                metrics.f1_score(
                    y_true=np.reshape(binary_y_qtc[i], (-1,)),
                    y_pred=np.reshape(binary_y_pred_qtc[i], (-1,)),
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        else:
            f1 = [
                metrics.f1_score(
                    y_true=binary_y_qtc[i].max(axis=1)[..., 0],
                    y_pred=binary_y_pred_qtc[i].max(axis=1)[..., 0],
                )
                for i in range(binary_y_qtc.shape[0])
                if np.sum(binary_y_qtc[i]) > 0
            ]

        f1 = np.array(f1)
        out = {
            'metric': f1,
            'n_removed': np.zeros_like(f1),
            'n_samples': y_pred.size,
            'deleted_indices': [],
        }

        return out


class ScikitQTcPT(ScikitQTcAccuracy):
    def __init__(
            self,
            qt_interval_ind: int,
            rr_interval_ind: int,
            prediction_horizon: int = 1,
            threshold: Optional[float] = None,
            trajectory_length: int = 6,
            framingham_correction: bool = True,
            point_wise: bool = True,
    ):
        super(ScikitQTcPT, self).__init__(
            qt_interval_ind=qt_interval_ind,
            rr_interval_ind=rr_interval_ind,
            prediction_horizon=prediction_horizon,
            threshold=threshold,
            trajectory_length=trajectory_length,
            framingham_correction=framingham_correction,
            point_wise=point_wise,
        )

        self.tpr_calc = ScikitQTcRecall(
            qt_interval_ind=qt_interval_ind,
            rr_interval_ind=rr_interval_ind,
            prediction_horizon=prediction_horizon,
            threshold=threshold,
            trajectory_length=trajectory_length,
            framingham_correction=framingham_correction,
            point_wise=point_wise,
        )
        self.fpr_calc = ScikitQTcSpecificity(
            qt_interval_ind=qt_interval_ind,
            rr_interval_ind=rr_interval_ind,
            prediction_horizon=prediction_horizon,
            threshold=threshold,
            trajectory_length=trajectory_length,
            framingham_correction=framingham_correction,
            point_wise=point_wise,
        )

    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        # Do nothing, computation is done by the attributed calculators

        out = {
            'x': x,
            'y': y,
            'y_pred': y_pred
        }

        return out

    def aggregate_metric_across_systems(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs['x']
        y = inputs['y']
        y_pred = inputs['y_pred']

        # Calculate TPR
        tpr_inputs = self.tpr_calc(x=x, y=y, y_pred=y_pred)
        tpr = tpr_inputs['metric']
        tnr_inputs = self.fpr_calc(x=x, y=y, y_pred=y_pred)
        tnr = tnr_inputs['metric']

        tpr = np.mean(tpr)
        tnr = np.mean(tnr)
        pt = (
                ((1 - tnr) ** 0.5) /
                (((1 - tnr) ** 0.5) + (tpr ** 0.5))
        )
        pt = np.array([pt, ])
        out = {
            'metric': pt,
            'n_removed': np.zeros_like(pt),
            'n_samples': y_pred.size,
            'deleted_indices': [],
        }

        return out
