from abc import ABC
from scipy import spatial
from matplotlib.path import Path
from matplotlib.spines import Spine
from prettytable import PrettyTable
from midst.utils.metrics import Metric
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection
from typing import Sequence, Union, List, Tuple, Any, Dict, Optional

import os
import glob
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


class VisDynamics(ABC):
    """
    A class for managing the visualizations of the concurrent dynamics experiments
    """

    def __init__(self):
        pass

    @staticmethod
    def _plot(
            plots: Union[List, Tuple],
            colors: Tuple = ('r', 'b', 'g', 'y', 'm', 'k'),
            legend_msg: str = '',
            title: str = '',
            line_style: str = '-',
            marker: str = 'x',
            alpha: float = 1.,
            axes: plt.Axes = None,
    ) -> None:
        """
        Utility method for plotting a sequence of plots with a shared legend message

        :param plots: A sequence of iterables to be plotted.
        :param colors: Colors to plot, each plot i will be in the color i % len(colors)
        :param legend_msg: A shared name in the legend. Then entries will be 'legend_msg i' for each plot
        :param title: Plot title
        :param line_style: The plotted line style
        :param marker: The marker style
        :param alpha: Color intensity
        :param axes: Axes to use for plotting, if None generates a new one. Default is None.
        """

        if axes is None:
            fig, axes = plt.subplots(figsize=[21, 11])

        plt.sca(axes)

        legend = []
        for i, p in enumerate(plots):
            plt.plot(
                p,
                c=f"{colors[int(i % len(colors))]}",
                marker=marker,
                linestyle=line_style,
                alpha=alpha,
            )
            legend.append(f"{legend_msg} {i}")

        plt.title(title)
        plt.legend(legend)

    @staticmethod
    def _generate_trajectory(array: np.ndarray, overlap: int) -> np.ndarray:
        """
        Generates a 1D continuous trajectory out of a 2D array.

        :param array: (np.ndarray) The array from which to generate the trajectory
        :param overlap: (int) The overlap between each two consecutive rows.

        :return: (np.ndarray) The continuous 1D trajectory
        """

        concatenated_array = np.concatenate(
            (
                    [array[0, :], ] +
                    [
                        array[i + 1, overlap:]
                        for i in range(array.shape[0] - 1)
                    ]
            ),
            0)

        return concatenated_array

    @staticmethod
    def visualize_concurrent_dynamics_preds(
            preds: np.ndarray,
            gt: np.ndarray,
            k: int = None,
            n: int = None,
            overlap: int = 0,
            save_path: str = None,
    ) -> None:
        """
        A utility method for visualizing the dynamic trajectories of the given `ConcurrentDynamics` object.

        :param preds: (np.ndarray) The predictions, should be with shape (Batch, Dynamics, Time, Observable Components).
        :param gt: (np.ndarray) The ground-truth, should be with shape (Batch, Dynamics, Time, Observable Components).
        :param k: (int) Number of dynamics components to visualize, if None then plots all of them. Default is None.
        :param n: (int) Number of observable components to visualize, if None then plots all of them. Default is None.
        :param overlap: (int) The overlap between each two consecutive windows in the batch. Used to generate a
        continuous trajectory.
        :param save_path: (str) If not None, specifies where to save the figure to.
        """

        assert preds.shape == gt.shape, f"preds shape and gt shape must be the same, " \
                                        f"but they are {preds.shape} and {gt.shape}, respectively."

        # Plot ground-truth dynamical components vs. predicted one
        k = preds.shape[1] if k is None else min(k, preds.shape[1])
        n = preds.shape[3] if n is None else min(n, preds.shape[3])
        fig, axes = plt.subplots(figsize=[21, 11])

        # Concatenate entire trajectories from batches
        gt_trajectoris = [
            [
                np.expand_dims(
                    VisDynamics._generate_trajectory(array=gt[:, d, :, o],
                                                     overlap=overlap), 1
                )
                for o in range(n)
            ]
            for d in range(k)
        ]
        gt_trajectoris = [np.expand_dims(np.concatenate(arr, 1), 0) for arr in
                          gt_trajectoris]
        gt = np.concatenate(gt_trajectoris, 0)

        preds_trajectoris = [
            [
                np.expand_dims(
                    VisDynamics._generate_trajectory(array=preds[:, d, :, o],
                                                     overlap=overlap), 1
                )
                for o in range(n)
            ]
            for d in range(k)
        ]
        preds_trajectoris = [np.expand_dims(np.concatenate(arr, 1), 0) for arr in
                             preds_trajectoris]
        preds = np.concatenate(preds_trajectoris, 0)

        gt_pkl_save_path = os.path.join(save_path, "gt_plots.pkl")
        with open(gt_pkl_save_path, 'wb') as f:
            pickle.dump(obj=gt, file=f)

        preds_pkl_save_path = os.path.join(save_path, "preds_plots.pkl")
        with open(preds_pkl_save_path, 'wb') as f:
            pickle.dump(obj=preds, file=f)

        # Pick sample from batch
        VisDynamics._plot(
            plots=[preds[0, :, i] for i in range(n)],
            title='Observable Components - Dynamic Component 1',
            axes=axes,
            line_style='--',
        )
        VisDynamics._plot(
            plots=[gt[0, :, i] for i in range(n)],
            title='Observable Components - Dynamic Component 1',
            axes=axes,
        )
        VisDynamics._plot(
            plots=[[gt[0, 0, i].tolist(), ] + gt[0, :-1, i].tolist() for i in range(n)],
            title='Observable Components - Dynamic Component 1',
            axes=axes,
            line_style=':',
        )

        legend = [
            f"Observable Component {i} - Predictions" for i in range(n)
        ]
        legend += [
            f"Observable Component {i} - Ground-Truth" for i in range(n)
        ]
        legend += [
            f"Observable Component {i} - Naive Predictor" for i in range(n)
        ]
        plt.legend(legend)

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path,
                                   "ObservableComponentsDynamicComponent1.pdf"),
                dpi=300,
                orientation='landscape',
                format='pdf',
            )

        fig, axes = plt.subplots(figsize=[21, 11])
        VisDynamics._plot(
            plots=[preds[i, :, 0] for i in range(k)],
            title='Dynamic Components - Observable Component 1',
            axes=axes,
            line_style='--',
        )
        VisDynamics._plot(
            plots=[gt[i, :, 0] for i in range(k)],
            title='Dynamic Components - Observable Component 1',
            axes=axes,
        )
        VisDynamics._plot(
            plots=[([gt[i, 0, 0].tolist(), ] + gt[i, :-1, 0].tolist()) for i in
                   range(k)],
            title='Dynamic Components - Observable Component 1',
            axes=axes,
            line_style=':',
        )

        legend = [
            f"Dynamics Component {i} - Predictions" for i in range(k)
        ]
        legend += [
            f"Dynamics Component {i} - Ground-Truth" for i in range(k)
        ]
        legend += [
            f"Dynamics Component {i} - Naive Predictor" for i in range(k)
        ]
        plt.legend(legend)

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, "DynamicComponentsDynamicComponent1.pdf"),
                dpi=300,
                orientation='landscape',
                format='pdf',
            )

        plt.show()

    @staticmethod
    def visualize_cross_models_statistics(
            dir_paths: Sequence[str],
            legend: Sequence[str] = None,
    ) -> None:
        """
        A utility method for plotting the loss and accuracy of various models against each other

        :param dir_paths: (Sequence[str]) A sequence of paths, where each path points to a dir with 2 files, one with
        the word 'loss' and one with the word 'acc' in them, to be plotted.
        :param legend: (Sequence[str]) A sequence of legend names for each plot

        :return: None
        """

        assert len(dir_paths) > 0
        assert len(legend) == len(dir_paths) if legend is not None else True

        n = len(dir_paths)

        losses = []
        accuracies = []
        for i in range(n):
            loss_file = glob.glob(os.path.join(dir_paths[i], '*loss*'))[0]
            acc_file = glob.glob(os.path.join(dir_paths[i], '*acc*'))[0]

            with open(loss_file, 'rb') as f:
                loss = pickle.load(f)
                losses.append(loss)

            with open(acc_file, 'rb') as f:
                acc = pickle.load(f)
                accuracies.append(acc)

        fig, axes = plt.subplots(figsize=[21, 11])
        VisDynamics._plot(
            plots=losses,
            title='Loss',
            axes=axes,
        )
        axes.legend(legend)

        fig, axes = plt.subplots(figsize=[21, 11])
        VisDynamics._plot(
            plots=accuracies,
            title='Accuracy',
            axes=axes,
        )
        axes.legend(legend)

        plt.show()

    @staticmethod
    def visualize_system_roots(
            dynamics_path: str,
            title: str = '',
            save_path: str = None,
            display_k: int = None,
    ) -> None:
        """
        A utility method for plotting the eigenvalues map of the real data

        :param dynamics_path: Path to the real data .pkl dynamics file
        :param title: Title to attach to the plot
        :param save_path: Where to save the plot to
        :param display_k: Number of singular values to display

        :return: None
        """

        with open(dynamics_path, 'rb') as f:
            ref_dynamics = pickle.load(f)

        ref_eigs = []
        for d in ref_dynamics:
            w = np.linalg.eigvals(d)
            w_real = w.real
            w_imag = w.imag
            ref_eigs.append((w_real, w_imag))

        if display_k is None:
            n_roots = len(ref_eigs)

        else:
            n_roots = min(len(ref_eigs), display_k)

        fig, axes = plt.subplots(ncols=n_roots, figsize=[21, 11])
        if n_roots == 1:
            axes = (axes,)

        for i in range(n_roots):
            axes[i].plot(ref_eigs[i][0], ref_eigs[i][1], 'bo', fillstyle='none', )
            axes[i].set_xlabel("Real")
            axes[i].set_ylabel("Im")
            axes[i].set_title(f"Sub-System {i + 1}")
            axes[i].set_xlim([-2, 2])
            axes[i].set_ylim([-2, 2])
            plt.suptitle(f"{title}")

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, 'SingularValues.pdf'),
                orientation='landscape',
                format='pdf',
            )

    @staticmethod
    def visualize_system_vs_pred_roots(
            ref_dynamics_path: str,
            learned_dynamics_path: Sequence[str] = None,
            title: str = '',
    ) -> None:
        """
        A utility method for plotting the eigenvalues map of the real data vs. the learned dynamics

        :param ref_dynamics_path: Path to the real data .pkl dynamics file
        :param learned_dynamics_path: Path to .pkl learned dynamics files, should be a sequence with one
        file per sub-system.
        :param title: Title to attach to each plot

        :return: None
        """

        with open(ref_dynamics_path, 'rb') as f:
            d = pickle.load(f)
            ref_dynamics = d.dynamics

        learned_dynamics = []
        for p in learned_dynamics_path:
            with open(p, 'rb') as f:
                d = pickle.load(f)
                learned_dynamics.append(d[0][0])

        ref_eigs = []
        for d in ref_dynamics:
            w = np.linalg.eigvals(d)
            w_real = w.real
            w_imag = w.imag
            ref_eigs.append((w_real, w_imag))

        learned_eigs = []
        for d in learned_dynamics:
            w = np.linalg.eigvals(d)
            w_real = w.real
            w_imag = w.imag
            learned_eigs.append((w_real, w_imag))
            # learned_eigs.append((w_real, np.zeros_like(w_real)))

        n_systems = len(ref_eigs)
        fig, axes = plt.subplots(ncols=n_systems, figsize=[21, 11])
        if n_systems == 1:
            axes = (axes,)

        for i in range(n_systems):
            axes[i].plot(ref_eigs[i][0], ref_eigs[i][1], 'bo', fillstyle='none', )
            axes[i].plot(learned_eigs[i][0], learned_eigs[i][1], 'rx')
            axes[i].set_xlabel("Real")
            axes[i].set_ylabel("Im")
            axes[i].set_title(f"Sub-System {i + 1}")
            axes[i].set_xlim([-2, 2])
            axes[i].set_ylim([-2, 2])
            plt.suptitle(f"{title} - Eigenvalues")

    @staticmethod
    def visualize_system_vs_pred_svd(
            ref_dynamics_path: str,
            learned_dynamics_path: Union[Sequence[Sequence[str]], Sequence[str]] = None,
            title: str = '',
            save_path: str = None,
            cmap_name: str = 'winter'
    ) -> None:
        with open(ref_dynamics_path, 'rb') as f:
            d = pickle.load(f)
            ref_dynamics = d.dynamics

        learned_dynamics = []
        for p in learned_dynamics_path:
            if isinstance(p, tuple) or isinstance(p, list):
                sub_dynamics = []
                for sub_p in p:
                    with open(sub_p, 'rb') as f:
                        d = pickle.load(f)
                        d = d[-1]
                        sub_dynamics.append(np.expand_dims(d, 1))

                sub_dynamics = np.concatenate(sub_dynamics, 1)
                learned_dynamics.append(sub_dynamics)

            else:
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                    d = d[-1]
                    if len(d.shape) == 3:
                        d = d[0]

                    learned_dynamics.append(d)

        ref_svd = []
        for d in ref_dynamics:
            u, s, vt = np.linalg.svd(d)
            ref_svd.append((u, s, vt))

        learned_svd = []
        for d in learned_dynamics:
            u, s, vt = np.linalg.svd(d)
            learned_svd.append((u, s, vt))

        ref_eigval = []
        for d in ref_dynamics:
            e = np.linalg.eig(d)[0]
            ref_eigval.append((e.real, e.imag))

        learned_eigval = []
        for d in learned_dynamics:
            e = np.linalg.eig(d)[0]
            learned_eigval.append((e.real, e.imag))

        # Plot Singular values
        n_systems = len(ref_svd)
        fig, axes = plt.subplots(ncols=n_systems, figsize=[21, 11])
        if n_systems == 1:
            axes = (axes,)

        for i in range(n_systems):
            singular_values_dist = np.linalg.norm((ref_svd[i][1] - learned_svd[i][1]), ord=2)
            axes[i].plot(ref_svd[i][1], 'bo', fillstyle='none', )
            axes[i].plot(learned_svd[i][1], 'rx')
            axes[i].set_xlabel("Singular Value index")
            axes[i].set_ylabel("Singular Value")
            axes[i].set_title(f"SS {i + 1} - L2 Dist = {singular_values_dist:.3f}")
            plt.suptitle(f"{title} - Singular Values")

            if i == 0:
                axes[0].legend(['GT', 'Learned'])

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, 'SVD.pdf'),
                orientation='landscape',
                format='pdf',
            )

        # Plot Eigen values
        fig, axes = plt.subplots(ncols=n_systems, figsize=[21, 11])
        if n_systems == 1:
            axes = (axes,)

        cmap = plt.get_cmap(cmap_name)
        colors = np.linspace(start=0, stop=1, num=ref_eigval[0][0].shape[0])
        colors = [cmap(c) for c in colors]
        for i in range(n_systems):
            ref_sorted = np.argsort(ref_eigval[i][0])
            learned_sorted = np.argsort(learned_eigval[i][0])

            eig_values_dist = np.linalg.norm(
                np.array(
                    [
                        ref_eigval[i][0][ref_sorted[k]] - learned_eigval[i][0][learned_sorted[k]]
                        for k in range(ref_eigval[i][0].shape[0])
                    ]
                ),
                ord=2,
            )

            for k in range(ref_eigval[i][0].shape[0]):
                axes[i].plot(
                    ref_eigval[i][0][ref_sorted[k]],
                    ref_eigval[i][1][ref_sorted[k]],
                    c=colors[k],
                    marker='o',
                    linestyle='None',
                    fillstyle='none',
                )
                axes[i].plot(
                    learned_eigval[i][0][learned_sorted[k]],
                    learned_eigval[i][1][learned_sorted[k]],
                    c=colors[k],
                    marker='x',
                    linestyle='None',
                )

            axes[i].set_xlabel(r"Re($\lambda$)")
            axes[i].set_ylabel(r"Imag($\lambda$)")
            axes[i].set_title(f"{i + 1} - L2 Dist = {eig_values_dist:.3f}")
            plt.suptitle(f"{title} - Eigenvalues")

            if i == 0:
                axes[0].legend(['GT', 'Learned'])

        t = np.linspace(start=(-2 * np.pi), stop=(2 * np.pi), num=100)
        for i in range(n_systems):
            axes[i].plot(np.cos(t), np.sin(t), 'k-', linewidth=1)

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, 'EigVals.pdf'),
                orientation='landscape',
                format='pdf',
            )

        # Plot |Eigen values|
        fig, axes = plt.subplots(ncols=n_systems, figsize=[21, 11])
        if n_systems == 1:
            axes = (axes,)

        for i in range(n_systems):
            ref_sorted = np.argsort(ref_eigval[i][0])
            learned_sorted = np.argsort(learned_eigval[i][0])
            for k in range(ref_eigval[i][0].shape[0]):
                axes[i].plot(
                    ref_eigval[i][0][ref_sorted[k]],
                    0,
                    c=colors[k],
                    marker='o',
                    fillstyle='none',
                    linestyle='None',
                )
                axes[i].plot(
                    learned_eigval[i][0][learned_sorted[k]],
                    0,
                    c=colors[k],
                    marker='x',
                    linestyle='None',
                )

            axes[i].set_xlabel(r"|($\lambda$)|")
            axes[i].set_title(f"Sub-System {i + 1}")
            plt.suptitle(f"{title} - |Eigenvalues|")

            if i == 0:
                axes[0].legend(['GT', 'Learned'])

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, 'AbsEigVals.pdf'),
                orientation='landscape',
                format='pdf',
            )

        # Plot Cosine distance
        fig, axes = plt.subplots(ncols=n_systems, figsize=[21, 11])
        if n_systems == 1:
            axes = (axes,)

        for i in range(n_systems):
            u_cos_distance = [
                spatial.distance.cosine(ref_svd[i][0][:, j], learned_svd[i][0][:, j])
                for j in range(learned_svd[i][0].shape[1])
            ]
            v_cos_distance = [
                spatial.distance.cosine(ref_svd[i][2][:, j], learned_svd[i][2][:, j])
                for j in range(learned_svd[i][2].shape[1])
            ]
            axes[i].plot(u_cos_distance, 'bo', fillstyle='none')
            axes[i].plot(v_cos_distance, 'rx', fillstyle='none')
            axes[i].set_xlabel("Vector index")
            axes[i].set_ylabel("Cosine Distance")
            axes[i].set_title(f"Sub-System {i + 1}")
            plt.suptitle(f"{title} - Cosine Distance")

            if i == 0:
                axes[0].legend(['U', 'V'])

        if save_path is not None:
            plt.savefig(
                fname=os.path.join(save_path, 'CosDist.pdf'),
                orientation='landscape',
                format='pdf',
            )


class CVExpSummarizer(ABC):
    """
    A class for summarizing the results of cross-validation experiments.
    """

    def __init__(self):
        pass

    @staticmethod
    def summarize_train_loss_and_acc(exp_dir: str) -> Sequence[np.ndarray]:
        val_d = os.path.join(exp_dir, 'Val')
        acc_file = glob.glob(os.path.join(val_d, 'fit_eval_acc*'))
        acc_file = acc_file[0]
        loss_file = glob.glob(os.path.join(val_d, 'fit_eval_loss*'))
        loss_file = loss_file[0]

        with open(acc_file, 'rb') as f:
            acc = pickle.load(f)
            val_acc = np.array(acc[0])

        with open(loss_file, 'rb') as f:
            loss = pickle.load(f)
            val_loss = np.array(loss[0])

        train_d = os.path.join(exp_dir, 'Train')
        acc_file = glob.glob(os.path.join(train_d, 'fit_train_acc*'))
        acc_file = acc_file[0]
        loss_file = glob.glob(os.path.join(train_d, 'fit_train_loss*'))
        loss_file = loss_file[0]

        with open(acc_file, 'rb') as f:
            acc = pickle.load(f)
            train_acc = np.array(acc[0])

        with open(loss_file, 'rb') as f:
            loss = pickle.load(f)
            train_loss = np.array(loss[0])

        return (
            train_acc,
            train_loss,
            val_acc,
            val_loss,
        )

    @staticmethod
    def summarize_train_cv_loss_and_acc(exp_dir: str) -> Sequence[np.ndarray]:
        folds_dirs = os.listdir(exp_dir)
        folds_dirs = [
            os.path.join(exp_dir, d)
            for d in folds_dirs
            if os.path.isdir(os.path.join(exp_dir, d)) and 'Fold' in d
        ]

        test_accuracies = []
        test_losses = []
        train_accuracies = []
        train_losses = []
        for fold_dir in folds_dirs:
            test_d = os.path.join(fold_dir, 'Val')
            acc_file = glob.glob(os.path.join(test_d, 'fit_eval_acc*'))
            acc_file = acc_file[0]
            loss_file = glob.glob(os.path.join(test_d, 'fit_eval_loss*'))
            loss_file = loss_file[0]

            with open(acc_file, 'rb') as f:
                acc = pickle.load(f)
                acc = np.expand_dims(np.array(acc[0]), 0)

            with open(loss_file, 'rb') as f:
                loss = pickle.load(f)
                loss = np.expand_dims(np.array(loss[0]), 0)

            test_accuracies.append(acc)
            test_losses.append(loss)

            train_d = os.path.join(fold_dir, 'Train')
            acc_file = glob.glob(os.path.join(train_d, 'fit_train_acc*'))
            acc_file = acc_file[0]
            loss_file = glob.glob(os.path.join(train_d, 'fit_train_loss*'))
            loss_file = loss_file[0]

            with open(acc_file, 'rb') as f:
                acc = pickle.load(f)
                acc = np.expand_dims(np.array(acc[0]), 0)

            with open(loss_file, 'rb') as f:
                loss = pickle.load(f)
                loss = np.expand_dims(np.array(loss[0]), 0)

            train_accuracies.append(acc)
            train_losses.append(loss)

        train_accuracies = np.concatenate(train_accuracies, 0)
        val_accuracies = np.concatenate(test_accuracies, 0)
        train_losses = np.concatenate(train_losses, 0)
        val_losses = np.concatenate(test_losses, 0)

        train_acc_mean = np.mean(train_accuracies, axis=0)
        train_acc_std = np.std(train_accuracies, axis=0)
        val_acc_mean = np.mean(val_accuracies, axis=0)
        val_acc_std = np.std(val_accuracies, axis=0)
        train_loss_mean = np.mean(train_losses, axis=0)
        train_loss_std = np.std(train_losses, axis=0)
        val_loss_mean = np.mean(val_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)

        return (
            train_acc_mean,
            train_acc_std,
            val_acc_mean,
            val_acc_std,
            train_loss_mean,
            train_loss_std,
            val_loss_mean,
            val_loss_std,
        )

    @staticmethod
    def summarize_cv_exp(
            exp_dir: str,
    ) -> Tuple[float, ...]:
        """
        This method go over a cross-validation experiments and query the final test results from each fold. Finally,
        it aggregates them.

        :param exp_dir: (str) Path to the directory holding the results of all folds.

        :return: (Tuple[float]) A tuple of floats, containing the mean and std of the loss metric and mean and std of
        the evaluation metric on all folds
        """

        folds_dirs = os.listdir(exp_dir)
        folds_dirs = [
            os.path.join(exp_dir, d)
            for d in folds_dirs
            if os.path.isdir(os.path.join(exp_dir, d)) and 'Fold' in d
        ]

        # For single fold experiments
        if len(folds_dirs) == 0:
            folds_dirs = (exp_dir,)

        test_accuracies = []
        test_losses = []
        train_accuracies = []
        train_losses = []
        for fold_dir in folds_dirs:
            test_d = os.path.join(fold_dir, 'Test')
            acc_file = glob.glob(os.path.join(test_d, '*acc*'))
            acc_file = acc_file[0]
            loss_file = glob.glob(os.path.join(test_d, '*loss*'))
            loss_file = loss_file[0]

            with open(acc_file, 'rb') as f:
                acc = pickle.load(f)
                acc = np.mean(acc)

            with open(loss_file, 'rb') as f:
                loss = pickle.load(f)
                loss = np.mean(loss)

            test_accuracies.append(acc)
            test_losses.append(loss)

            train_d = os.path.join(fold_dir, 'Train')
            acc_file = glob.glob(os.path.join(train_d, '*acc*'))
            acc_file = acc_file[0]
            loss_file = glob.glob(os.path.join(train_d, '*loss*'))
            loss_file = loss_file[0]

            with open(acc_file, 'rb') as f:
                acc = pickle.load(f)
                acc = np.mean(acc)

            with open(loss_file, 'rb') as f:
                loss = pickle.load(f)
                loss = np.mean(loss)

            train_accuracies.append(acc)
            train_losses.append(loss)

        return (
            float(np.mean(train_losses).item()),
            float(np.std(train_losses).item()),
            float(np.mean(train_accuracies).item()),
            float(np.std(train_accuracies).item()),
            float(np.mean(test_losses).item()),
            float(np.std(test_losses).item()),
            float(np.mean(test_accuracies).item()),
            float(np.std(test_accuracies).item()),
        )

    @staticmethod
    def summarize_multiple_cv_exp(
            exp_dirs: Sequence[Union[Sequence, str]],
    ) -> Tuple[List, ...]:
        """
        This method go over multiple cross-validation experiments, it queries the final test results from each fold,
        of every experiment and aggregates them. Finally, it prints a comparison.

        :param exp_dirs: (Sequence[str]) Paths to the directories holding the results of all folds for each experiment.

        :return: (Tuple[List...]) A tuple of the following lists:
        Experiment names
        Loss means
        Loss STDs
        Accuracies means
        Accuracies STDs
        """

        exp_names = []
        train_exp_loss_means = []
        train_exp_loss_stds = []
        train_exp_acc_means = []
        train_exp_acc_stds = []
        test_exp_loss_means = []
        test_exp_loss_stds = []
        test_exp_acc_means = []
        test_exp_acc_stds = []

        # Summarize every experiment
        for exp in exp_dirs:
            if isinstance(exp, str):
                stats = CVExpSummarizer.summarize_cv_exp(exp)
                exp_names.append(exp.split(os.sep)[-1])

            else:
                stats = [CVExpSummarizer.summarize_cv_exp(e) for e in exp]
                train_loss_means = np.mean([s[0] for s in stats])
                train_loss_stds = np.mean([s[1] for s in stats])
                train_acc_means = np.mean([s[2] for s in stats])
                train_acc_stds = np.mean([s[3] for s in stats])
                test_loss_means = np.mean([s[4] for s in stats])
                test_loss_stds = np.mean([s[5] for s in stats])
                test_acc_means = np.mean([s[6] for s in stats])
                test_acc_stds = np.mean([s[7] for s in stats])
                stats = (
                    train_loss_means,
                    train_loss_stds,
                    train_acc_means,
                    train_acc_stds,
                    test_loss_means,
                    test_loss_stds,
                    test_acc_means,
                    test_acc_stds,
                )
                exp_names.append(exp[0].split(os.sep)[-1])

            train_exp_loss_means.append(stats[0])
            train_exp_loss_stds.append(stats[1])
            train_exp_acc_means.append(stats[2])
            train_exp_acc_stds.append(stats[3])
            test_exp_loss_means.append(stats[4])
            test_exp_loss_stds.append(stats[5])
            test_exp_acc_means.append(stats[6])
            test_exp_acc_stds.append(stats[7])

        # Print a comparison of all results
        for i in range(len(exp_dirs)):
            print(f"{'-' * 25}")
            print(f"{exp_names[i]}:")
            print(
                f"Train Loss: {train_exp_loss_means[i]} \u00B1 {train_exp_loss_stds[i]},"
                f"Train Acc: {train_exp_acc_means[i]} \u00B1 {train_exp_acc_stds[i]}"
            )
            print(
                f"Test Loss: {test_exp_loss_means[i]} \u00B1 {test_exp_loss_stds[i]},"
                f"Test Acc: {test_exp_acc_means[i]} \u00B1 {test_exp_acc_stds[i]}"
            )

        outputs = (
            exp_names,
            train_exp_loss_means,
            train_exp_loss_stds,
            train_exp_acc_means,
            train_exp_acc_stds,
            test_exp_loss_means,
            test_exp_loss_stds,
            test_exp_acc_means,
            test_exp_acc_stds,
        )

        return outputs

    @staticmethod
    def _compute_metric(
            exp_dir: str,
            metric: Metric,
            stage: str = 'Test',
    ) -> Dict[str, Any]:

        files_dir = os.path.join(exp_dir, stage)
        inputs_file = glob.glob(os.path.join(files_dir, '*x_gt*'))
        inputs_file = inputs_file[0]
        gt_file = glob.glob(os.path.join(files_dir, '*y_gt*'))
        gt_file = gt_file[0]
        pred_file = glob.glob(os.path.join(files_dir, '*y_pred*'))
        pred_file = pred_file[0]

        with open(inputs_file, 'rb') as f:
            inputs = pickle.load(f)

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)

        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)

        inputs = np.concatenate(inputs, 0)
        gt = np.concatenate(gt, 0)
        pred = np.concatenate(pred, 0)
        metric_out = metric(x=inputs, y=gt, y_pred=pred)

        if 'match' not in metric_out:
            metric_out['match'] = 'NA'

        return metric_out

    @staticmethod
    def _compute_leave_1_out_metric(
            exp_dirs: Sequence[str],
            metric: Metric,
            stage: str = 'Test',
    ) -> Dict[str, Any]:

        aggregated_inputs = []
        aggregated_y = []
        aggregated_y_pred = []
        for exp_dir in exp_dirs:
            files_dir = os.path.join(exp_dir, stage)
            inputs_file = glob.glob(os.path.join(files_dir, '*x_gt*'))
            inputs_file = inputs_file[0]
            gt_file = glob.glob(os.path.join(files_dir, '*y_gt*'))
            gt_file = gt_file[0]
            pred_file = glob.glob(os.path.join(files_dir, '*y_pred*'))
            pred_file = pred_file[0]

            with open(inputs_file, 'rb') as f:
                inputs = pickle.load(f)

            with open(gt_file, 'rb') as f:
                gt = pickle.load(f)

            with open(pred_file, 'rb') as f:
                pred = pickle.load(f)

            inputs = np.concatenate(inputs, 0)
            gt = np.concatenate(gt, 0)
            pred = np.concatenate(pred, 0)

            aggregated_inputs.append(inputs)
            aggregated_y.append(gt)
            aggregated_y_pred.append(pred)

        aggregated_inputs = np.concatenate(aggregated_inputs, axis=0)
        aggregated_y = np.concatenate(aggregated_y, axis=0)
        aggregated_y_pred = np.concatenate(aggregated_y_pred, axis=0)
        metric_out = metric(x=aggregated_inputs, y=aggregated_y, y_pred=aggregated_y_pred)

        if 'match' not in metric_out:
            metric_out['match'] = 'NA'

        return metric_out

    @staticmethod
    def mean_match(matches: Sequence[Union[str, float]]) -> Union[float, str]:
        if any([isinstance(m, str) for m in matches]):
            matches = 'NA'

        else:
            matches = float(np.mean(matches).item())

        return matches

    @staticmethod
    def report_metric(
            exp_dirs: Sequence[Union[Sequence[Sequence[str]], Sequence[str], str]],
            metric: Metric,
            stage: str = 'Test',
            horizon_per_exp: Sequence[int] = None,
            trajectory_length_per_exp: Sequence[int] = None,
    ) -> Tuple[Sequence[float], ...]:
        """
        This method go over a list of single experiments, compute a given metric
        and its corresponding statistics in a given axis.

        :param exp_dirs: (Sequence[Union[Sequence[Sequence[str]], Sequence[str], str]]) Path to the experiment directory.
        :param metric: (Callable) The metric to compute. The callable should accept a list of numpy arrays, with
        each element being a batch sample from the respective variable (gt or prediction), and return the calculated
        metric over both lists, one against the other.
        :param stage: (str) Stage on which to compute the metric, one of 'Train', 'Val', or 'Test'
        :param horizon_per_exp: (Sequence[int]) A sequence of integers corresponding to the forecast horizon of 
        each experiment
        :param trajectory_length_per_exp: (Sequence[int]) A sequence of integers corresponding to the
         trajectory length of each experiment

        :return: (Tuple[float]) A tuple of floats, containing the mean and std of the metric on all folds
        """

        exp_names = []
        exp_means = []
        exp_stds = []
        exp_n_samples = []
        exp_n_removed = []
        exp_match = []
        exp_metrics = []
        exp_agg_metrics = []
        for i, exp in enumerate(exp_dirs):
            if horizon_per_exp is not None:
                metric.horizon = horizon_per_exp[i]

            if trajectory_length_per_exp is not None:
                metric.trajectory_length = trajectory_length_per_exp[i]

            if isinstance(exp, str):
                exp_names.append(exp.split(os.sep)[-1])
                stats = CVExpSummarizer._compute_metric(
                    exp_dir=exp,
                    metric=metric,
                    stage=stage,
                )
                stats['agg_metric'] = []

            elif isinstance(exp, Sequence) and isinstance(exp[0], str):
                exp_names.append(exp[0].split(os.sep)[-1])
                stats = [
                    CVExpSummarizer._compute_metric(
                        exp_dir=e,
                        metric=metric,
                        stage=stage,
                    )
                    for e in exp
                ]

                means = np.mean([s['metric_mean'] for s in stats])
                stds = np.std([s['metric_mean'] for s in stats])
                samples = np.mean([s['n_samples'] for s in stats])
                removed = np.mean([s['n_removed'] for s in stats])
                matches = CVExpSummarizer.mean_match([s['match'] for s in stats])
                metric_ = [s['metric'] for s in stats]
                min_n_samples = min([len(m) for m in metric_])
                agg_metric = np.mean(np.concatenate([m[None, :min_n_samples] for m in metric_], axis=0), axis=0)

                stats = {
                    'metric': [ss for s in metric_ for ss in s],
                    'metric_mean': means,
                    'metric_std': stds,
                    'n_samples': samples,
                    'n_removed': removed,
                    'match': matches,
                    'agg_metric': agg_metric,
                }

            elif isinstance(exp, Sequence) and isinstance(exp[0], Sequence) and isinstance(exp[0][0], str):
                exp_names.append(exp[0][0].split(os.sep)[-1])
                stats = [
                    [
                        CVExpSummarizer._compute_metric(
                            exp_dir=e,
                            metric=metric,
                            stage=stage,
                        )
                        for e in ee
                    ]
                    for ee in exp
                ]

                means = np.mean([ss['metric_mean'] for s in stats for ss in s])
                stds = np.std([ss['metric_mean'] for s in stats for ss in s])
                samples = np.mean([ss['n_samples'] for s in stats for ss in s])
                removed = np.mean([ss['n_removed'] for s in stats for ss in s])
                matches = CVExpSummarizer.mean_match([ss['match'] for s in stats for ss in s])
                metric_ = [ss['metric'] for s in stats for ss in s]
                agg_metric = [
                    np.mean(np.concatenate([ss['metric'][None, ...] for ss in s], axis=0), axis=0)[None, ...]
                    for s in stats
                ]
                agg_metric = np.mean(np.concatenate(agg_metric, axis=0), axis=0)
                stats = {
                    'metric': [ss for s in metric_ for ss in s],
                    'metric_mean': means,
                    'metric_std': stds,
                    'n_samples': samples,
                    'n_removed': removed,
                    'match': matches,
                    'agg_metric': agg_metric,
                }

            else:
                exp_names.append(exp[0][0].split(os.sep)[-1])
                stats_per_system = [
                    [
                        CVExpSummarizer._compute_metric(
                            exp_dir=e,
                            metric=metric,
                            stage=stage,
                        )
                        for e in e_per_seed
                    ]
                    for e_per_seed in exp
                ]

                stats = [
                    (
                        np.mean([s['metric_mean'] for s in system_stats]),
                        np.mean([s['metric_std'] for s in system_stats]),
                        np.mean([s['n_samples'] for s in system_stats]),
                        np.mean([s['n_removed'] for s in system_stats]),
                        CVExpSummarizer.mean_match([s['match'] for s in system_stats]),
                    )
                    for system_stats in stats_per_system
                ]

                means = np.mean([s[0] for s in stats])
                stds = np.std([s[1] for s in stats])
                samples = np.mean([s[2] for s in stats])
                removed = np.mean([s[3] for s in stats])
                matches = CVExpSummarizer.mean_match([s[4] for s in stats])

                stats = {
                    'metric': [],
                    'metric_mean': means,
                    'metric_std': stds,
                    'n_samples': samples,
                    'n_removed': removed,
                    'match': matches,
                    'agg_metric': [],
                }

            exp_means.append(stats['metric_mean'])
            exp_stds.append(stats['metric_std'])
            exp_n_samples.append(stats['n_samples'])
            exp_n_removed.append(stats['n_removed'])
            exp_metrics.append(stats['metric'])
            exp_match.append(stats['match'])
            exp_agg_metrics.append(stats['agg_metric'])

        # Print a comparison of all results
        for i in range(len(exp_dirs)):
            print(f"{'-' * 25}")
            print(f"{exp_names[i]}:")

        return exp_names, exp_agg_metrics, exp_metrics, exp_means, exp_stds, exp_n_samples, exp_n_removed, exp_match

    @staticmethod
    def report_leave_1_out_metric(
            exp_dirs: Sequence[Sequence[str]],
            metric: Metric,
            stage: str = 'Test',
    ) -> Tuple[Sequence[str], ...]:
        """
        This method go over a list of single experiments, compute a given metric
        and its corresponding statistics in a given axis.

        :param exp_dirs: (Sequence[Union[Sequence[str], str]]) Path to the experiment directory.
        :param metric: (Callable) The metric to compute. The callable should accept a list of numpy arrays, with
        each element being a batch sample from the respective variable (gt or prediction), and return the calculated
        metric over both lists, one against the other.
        :param stage: (str) Stage on which to compute the metric, one of 'Train', 'Val', or 'Test'

        :return: (Tuple[float]) A tuple of floats, containing the mean and std of the metric on all folds
        """

        exp_names = []
        exp_means = []
        exp_stds = []
        exp_n_samples = []
        exp_n_removed = []
        exp_match = []
        exp_matric = []
        for i, exp in enumerate(exp_dirs):
            exp_names.append(exp[0].split(os.sep)[-2])
            stats = CVExpSummarizer._compute_leave_1_out_metric(
                exp_dirs=exp,
                metric=metric,
                stage=stage,
            )

            exp_means.append(stats['metric_mean'])
            exp_stds.append(stats['metric_std'])
            exp_n_samples.append(stats['n_samples'])
            exp_n_removed.append(stats['n_removed'])
            exp_match.append(stats['match'])
            exp_matric.append(stats['metric'])

        # Print a comparison of all results
        for i in range(len(exp_dirs)):
            print(f"{'-' * 25}")
            print(f"{exp_names[i]}:")

        return exp_names, exp_matric, exp_means, exp_stds, exp_n_samples, exp_n_removed, exp_match

    @staticmethod
    def exp_pred_gt(exp_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method extract the test-time ground-truth and predictions.

        :param exp_dir: (str) Path to the directory holding the results of all folds.

        :return: (Tuple[np.ndarray]) A tuple of two np.ndarray, containing the ground truth and predictions,
        in that order.
        """

        test_d = os.path.join(exp_dir, 'Test')
        gt_file = glob.glob(os.path.join(test_d, 'y_gt_eval*.pkl'))[0]
        pred_file = glob.glob(os.path.join(test_d, 'y_pred_eval*.pkl'))[0]

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)
            gt = np.concatenate(gt, 0)

        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)
            pred = np.concatenate(pred, 0)

        return gt, pred

    @staticmethod
    def compute_multi_systems_metrics(
            exp_dirs: Sequence[str],
            metrics: Sequence[Metric]
    ) -> Tuple[Tuple[Tuple[float, float]]]:
        """
        This method go over multiple cross-validation experiments, it queries the final test predictions
        and ground truth from each fold of every experiment and aggregates them. If there are multiple
        concurrent systems, it averages over their predictions. Finally, it prints a comparison of
        the metrics of each experiment.

        :param exp_dirs: (Sequence[str]) Paths to the directories holding the results of all folds for each experiment.
        :param metrics: (Sequence[Callable]) A sequence of metrics to compute.
        The callable should accept a list of numpy arrays, with
        each element being a batch sample from the respective variable (gt or prediction),
        and return the calculated metric over both lists, one against the other.

        :return: (Tuple[Tuple[Tuple[int, int]]]) A tuple of tuples, where each inner tuple contains another
         tuple with the following two values:
         metric mean
         metric STD

         For a specific metric
        """

        experiments_metrics = []
        for exp in exp_dirs:
            folds_dirs = os.listdir(exp)
            folds_dirs = [
                os.path.join(exp, d)
                for d in folds_dirs
                if os.path.isdir(os.path.join(exp, d)) and 'Fold' in d
            ]

            folds_metrics = []
            for fold_dir in folds_dirs:
                gt, pred = CVExpSummarizer.exp_pred_gt(fold_dir)
                gt = gt.mean(1)
                pred = pred.mean(1)

                fold_metric = [metric(gt, pred) for metric in metrics]
                folds_metrics.append(fold_metric)

            aggregated_metrics = tuple(
                (
                    float(np.mean(
                        [fold_metric[m] for fold_metric in folds_metrics]).item()),
                    float(np.std(
                        [fold_metric[m] for fold_metric in folds_metrics]).item()),
                )
                for m in range(len(metrics))
            )
            experiments_metrics.append(aggregated_metrics)

        experiments_metrics = tuple(experiments_metrics)

        return experiments_metrics

    @staticmethod
    def radar_factory(
            num_vars: int,
            frame: str = 'circle',
    ):
        """
        Create a radar chart with `num_vars` axes.

        This function creates a RadarAxes projection and registers it.

        Parameters
        ----------
        num_vars : int
            Number of variables for radar chart.
        frame : {'circle', 'polygon'}
            Shape of frame surrounding axes.

        """

        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        class RadarAxes(PolarAxes):
            name = 'radar'
            # use 1 line segment to connect specified points
            RESOLUTION = 1

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                          radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                  spine_type='circle',
                                  path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)

        return theta


class AttractorsVisualizer(ABC):
    @staticmethod
    def plot_attractor(
            systems_states: Sequence[np.ndarray],
            title: str = '',
            show: bool = False,
            linewidth: float = 1.,
            cmap_name: str = None,
            color: str = 'b',
            s: int = 20,
            alpha: float = 0.5,
            axes: plt.Axes = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            zlabel: Optional[str] = None,
    ):
        if axes is None:
            fig, axes = plt.subplots(
                figsize=[21, 9],
                ncols=len(systems_states),
                nrows=1,
                subplot_kw={'projection': '3d'}
            )

        if len(systems_states) == 1:
            axes = [axes, ]

        for i, states in enumerate(systems_states):
            axes[i].plot(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                c=color,
                linewidth=linewidth,
                alpha=alpha if cmap_name is None else 0.0
            )

            if xlabel is not None:
                axes[i].set_xlabel(xlabel)

            if ylabel is not None:
                axes[i].set_ylabel(ylabel)

            if zlabel is not None:
                axes[i].set_zlabel(zlabel)

            if cmap_name is not None:
                n = states.shape[0]
                cmap = plt.get_cmap(cmap_name)
                for j in range(0, n - s, s):
                    axes[i].plot(
                        states[j:(j + s + 1), 0],
                        states[j:(j + s + 1), 1],
                        states[j:(j + s + 1), 2],
                        linewidth=linewidth,
                        color=cmap(j / n),
                        alpha=alpha,
                    )

        plt.suptitle(title)

        if show:
            plt.show()

    @staticmethod
    def plot_predictions(
            gt_states: Sequence[np.ndarray],
            pred_states: Sequence[np.ndarray],
            title: str = '',
            show: bool = False,
            linewidth: float = 1.,
            save_path: str = None,
    ):
        fig, axes = plt.subplots(
            figsize=[21, 9],
            ncols=len(gt_states),
            nrows=1,
            subplot_kw={'projection': '3d'}
        )

        # Plot ground-truth
        AttractorsVisualizer.plot_attractor(
            systems_states=gt_states,
            title='',
            show=False,
            linewidth=linewidth,
            cmap_name=None,
            color='b',
            axes=axes,
        )

        # Plot predictions
        AttractorsVisualizer.plot_attractor(
            systems_states=pred_states,
            title='',
            show=False,
            linewidth=linewidth,
            cmap_name=None,
            color='r',
            axes=axes,
        )

        plt.suptitle(title)
        axes[0].legend(["GT", "Predictions"])

        if save_path is not None:
            plt.savefig(
                fname=save_path,
                dpi=300,
                orientation='landscape',
                format='pdf',
            )

        if show:
            plt.show()


def count_parameters(model_path: str):
    model = torch.load(model_path, map_location=torch.device('cpu'))['model']

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.items():
        n_params = parameter.numel()
        table.add_row([name, n_params])
        total_params += n_params

    print(table)
    print(f"Total Trainable Params in {model_path}: {total_params}")

    return total_params
