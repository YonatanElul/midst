from abc import ABC
from itertools import product
from torch import from_numpy, Tensor, float32
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Union, Sequence, Optional, cast
from midst.data.synthetic_data import Attractors
from midst.utils.defaults import (
    GT_TENSOR_INPUTS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
)

import os
import glob
import h5py
import torch
import pickle
import shutil
import numpy as np
import netCDF4 as nc


class StrangeAttractorsDataset(Dataset):
    def __init__(
            self,
            filepath: str = None,
            temporal_horizon: int = None,
            overlap: int = None,
            time: int = 1,
            dt: float = 1.,
            attractors: Attractors = None,
            attractors_path: str = None,
            prediction_horizon: int = 1,
            noise: Optional[dict] = None,
            system_ind: Optional[Union[int, Sequence[int]]] = None,
            n_systems: Optional[int] = None,
            single_system: bool = False,
    ):
        super().__init__()

        assert not (single_system and system_ind is not None)

        self._filepath = filepath
        self._attractors_path = attractors_path
        self._prediction_horizon = prediction_horizon
        self._time = time
        self._dt = dt
        self._noise = noise
        self._system_ind = system_ind
        self._n_systems = n_systems
        self._single_system = single_system

        total_trajectory_length = int(time / dt)

        if attractors_path is not None:
            with open(attractors_path, 'rb') as f:
                attractors = pickle.load(f)

        if os.path.isfile(filepath):
            with h5py.File(filepath, mode='r') as hdf5_file:
                attractors_trajectory = hdf5_file['/data']['attractors_trajectory'][:]
                total_trajectory_length = attractors_trajectory.shape[1] - 1
                self._attractors_trajectory = attractors_trajectory

        self._temporal_horizon = temporal_horizon
        self._overlap = overlap
        self._total_trajectory_length = total_trajectory_length
        self._attractors = attractors
        self._length = (
                               (total_trajectory_length - temporal_horizon - prediction_horizon) //
                               (temporal_horizon - overlap)
                       ) - 1

        # If the data already exists then we don't need to re-generate it
        if not os.path.isfile(filepath):
            attractors_trajectory = self.generate_trajectory(
                total_trajectory_length=total_trajectory_length,
                attractors=attractors,
                dt=dt,
            )
            with h5py.File(filepath, 'w') as file:
                dataset = file.create_group('/data')
                dataset.create_dataset(
                    'attractors_trajectory',
                    shape=attractors_trajectory.shape,
                    dtype=attractors_trajectory.dtype,
                    data=attractors_trajectory,
                )

            self._attractors_trajectory = attractors_trajectory

    @staticmethod
    def generate_trajectory(
            total_trajectory_length: int,
            attractors: Attractors,
            dt: float,
    ) -> np.ndarray:
        attractors.compute_trajectory(n=total_trajectory_length, dt=dt)
        attractors_trajectory = np.concatenate(
            [
                np.expand_dims(att, 1) for att in attractors.trajectories
            ], 1
        )

        return attractors_trajectory

    def __len__(self) -> int:
        return self._length

    def __getitem__(
            self,
            index: int,
    ) -> Dict[str, Tensor]:
        assert index <= self._length, \
            f"Cannot access index {index} in a trajectory of length {self._length}"

        x_sequence = self._attractors_trajectory[
                     :, index:(index + self._temporal_horizon + self._prediction_horizon + 1)
                     ]
        y_sequence = [
            x_sequence[:, t:(t + self._temporal_horizon)]
            for t in range(1, self._prediction_horizon + 1)
        ]
        y_sequence = np.concatenate(y_sequence, 1)
        x_sequence = x_sequence[:, :self._temporal_horizon]

        if self._noise is not None:
            x_sequence += np.random.normal(loc=self._noise['loc'], scale=self._noise['scale'], size=x_sequence.shape)
            y_sequence += np.random.normal(loc=self._noise['loc'], scale=self._noise['scale'], size=y_sequence.shape)

        if self._n_systems is not None:
            x_sequence = x_sequence[:self._n_systems, ...]
            y_sequence = y_sequence[:self._n_systems, ...]

        if self._system_ind is not None:
            x_sequence = x_sequence[self._system_ind, :, :]
            y_sequence = y_sequence[self._system_ind, :, :]

            if isinstance(self._system_ind, int):
                x_sequence = x_sequence[None, ...]
                y_sequence = y_sequence[None, ...]

        if self._single_system:
            x_sequence = np.swapaxes(x_sequence, axis1=0, axis2=1)
            x_sequence = np.reshape(x_sequence, (1, x_sequence.shape[0], -1))
            y_sequence = np.swapaxes(y_sequence, axis1=0, axis2=1)
            y_sequence = np.reshape(y_sequence, (1, y_sequence.shape[0], -1))

        sample = {
            GT_TENSOR_INPUTS_KEY: from_numpy(x_sequence).type(float32),
            GT_TENSOR_PREDICITONS_KEY: from_numpy(y_sequence).type(float32),
        }

        return sample


class ECGRDVQDataOrganizer(ABC):
    """
    A utility class for preparing the hdf5 recordings of the ecgrdvq dataset as patient-ready samples.

    The WAVES attribute contains the indices of following events:
    P-Wave start, computed as: P-Wave peak - ((Q - P-Wave peak) / 2)
    P-Wave peak
    Q
    R-Peak
    S
    T-Wave peak
    T-wave end, computed as: P-Wave peak - ((T-Wave peak - S) / 2)
    """

    ARRAY_KEYS = (
        'rdr_index',
        'sex',
        'age',
        'height',
        'weight',
        'time_post_treatment',
        'sysbp',
        'diabp',
        'treatment_key',
        'avg_rr',
        'avg_pr',
        'avg_qt',
        'avg_qrs',
        'dosage',
        'erd_30',
        'lrd_30',
        't_amp',
    )
    ATTRIBUTES_KEYS = (
        'db_name',
        'race',
    )
    SEX = {
        'M': 0,
        'F': 1,
    }
    TREATMENTS = {
        'Placebo': 0,
        'Ranolazine': 1,
        'Verapamil HCL': 2,
        'Quinidine Sulph': 3,
        'Dofetilide': 4,
    }

    def __init__(
            self,
            dir_path: str,
            cache_path: str,
            arrays_keys: Union[list, tuple],
            attributes_keys: Union[list, tuple],
            min_allowed_beats: int = 5,
            csv_only: bool = False,
    ):
        """
        :param dir_path: (str) Path to the directory holding the hdf5 formatted recordings.
        :param cache_path: (str) Path to the directory where the processed Dataset should be cached.
        :param arrays_keys: (Union[list, tuple]) keys of array-like attributes to write.
        :param attributes_keys: (Union[list, tuple]) keys of one-per-record attributes to write.
        :param min_allowed_beats: (int) The threshold for throwing records with fewer annotated beats then the minimum
        allowed
        :param csv_only: (bool) Whether data was generated strictly from the accompanying csv file
        """

        super().__init__()

        assert all([k in self.ARRAY_KEYS for k in arrays_keys])
        assert all([k in self.ATTRIBUTES_KEYS for k in attributes_keys])

        self._dir_path = dir_path
        self._cache_path = cache_path
        self._csv_only = csv_only

        os.makedirs(cache_path, exist_ok=True)

        self._patients_start = 1001
        self._patients_end = 1022
        self._n_repetitions = 3
        self._n_treatments = 5
        self._treatments = (
            "Placebo",
            "Dofetilide",
            "Quinidine Sulph",
            "Ranolazine",
            "Verapamil HCL"
        )
        self._post_sampling_times = (
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            12.0,
            14.0,
            24.0,
        )
        self._pre_sampling_times = (-0.5,)
        self._arrays_keys = arrays_keys
        self._attributes_keys = attributes_keys
        self._min_allowed_beats = min_allowed_beats

        self._peak_to_ind = {
            'P_Peak': 1,
            'R_Peak': 3,
            'T_Peak': 5,
        }
        self._lead_for_peaks = 2

        pre = [
            [
                glob.glob(
                    os.path.join(dir_path, f'{p}', 'pre_treatment', f'{t}', '*.h5')
                )
                for p in range(self._patients_start, self._patients_end + 1)
            ]
            for t in self._treatments
        ]
        post = [
            [
                glob.glob(
                    os.path.join(
                        dir_path, f'{str(p)}', 'post_treatment', f'{t}', '*.h5'
                    )
                )
                for p in range(self._patients_start, self._patients_end + 1)
            ]
            for t in self._treatments
        ]
        valid_inds_pre = self._valid_patients(files=pre)
        valid_inds_post = self._valid_patients(files=post)
        valid_patients = cast(
            List,
            np.arange((self._patients_end - self._patients_start + 1)).tolist()
        )
        valid_patients = [
            i for i in valid_patients
            if (i in valid_inds_pre and i in valid_inds_post)
        ]
        pre = [
            [
                treatment_files[i] for i in valid_patients
            ]
            for treatment_files in pre
        ]
        post = [
            [
                treatment_files[i] for i in valid_patients
            ]
            for treatment_files in post
        ]
        pre_per_patient = [
            [
                treatment[p]
                for treatment in pre
            ]
            for p in range(len(valid_patients))
        ]
        post_per_patient = [
            [
                treatment[p]
                for treatment in post
            ]
            for p in range(len(valid_patients))
        ]

        patients = []
        for p in range(len(post_per_patient)):
            treatment_per_patient = []
            for t in range(len(self._treatments)):
                patient = pre_per_patient[p][t] + post_per_patient[p][t]
                treatment_per_patient.append(patient)

            patients.append(treatment_per_patient)

        self._build_patient_record(
            files=patients,
        )

    def _valid_patients(self, files: list) -> list:
        """
        Utility method for filtering out patients with missing recordings.

        :param files: (list) List of files of all patients

        :return: (list) List of indices of all valid patients, i.e. that have all treatment files
        """

        valid_patients = np.arange(
            (self._patients_end - self._patients_start + 1)
        ).tolist()

        for treatment_list in files:
            n_files = [
                len(patient_list)
                for patient_list in treatment_list
            ]
            max_f = max(n_files)
            valid_lists = [
                i
                for i, patient_list in enumerate(treatment_list)
                if len(patient_list) == max_f
            ]
            valid_patients = [
                i
                for i in valid_patients
                if i in valid_lists
            ]

        return valid_patients

    def _build_patient_record(self, files: list) -> None:
        """
        A utility method for merging together the records of each patient together for specific time points and
        organizing the various treatments as various channels / dynamics.

        :param files: (dict) Files of all patients in the set.
        """

        for p_num, patient_files in enumerate(files):
            print(f"Preparing {p_num + 1} / {len(files)}")

            # Separate_Linear records based on treatments and time-points
            files_per_time_point = [
                [
                    [
                        f
                        for f in treatment_files
                        if (f"_{str(t)}" in f)
                    ]
                    for t in (self._pre_sampling_times + self._post_sampling_times)
                    if len([f for f in treatment_files if (f"_{str(t)}" in f)])
                ]
                for treatment_files in patient_files
            ]
            n_time_points = len(files_per_time_point[0])
            current_time_combos = []
            for time_point in range(n_time_points):
                current_time_treatments = [
                    treatment[time_point]
                    for treatment in files_per_time_point
                ]
                combo = product(*current_time_treatments)
                current_time_combos.append(combo)

            # Build all possible samples available for that patient
            patients_array_data = []
            for time_combo in current_time_combos:
                for combo in time_combo:
                    for i, file in enumerate(combo):
                        with h5py.File(file, 'r') as h5file:
                            dataset = h5file['record']
                            if i == 0:
                                attributes_fields = {
                                    key: dataset.attrs[key]
                                    for key in self._attributes_keys
                                }
                                array_fields = {
                                    key: np.expand_dims(dataset[key], 0)
                                    for key in self._arrays_keys if key != 'waves'
                                }

                                if not self._csv_only:
                                    array_fields['waves'] = [dataset['waves'][:], ]
                                    for peak in self._peak_to_ind:
                                        # Correct annotation mistakes for the end of the T wave
                                        locs = dataset['waves'][self._peak_to_ind[peak], :].astype(np.int)
                                        locs[locs < 0] = 0
                                        locs[locs >= dataset['x'].shape[0]] = dataset['x'].shape[0] - 1
                                        locs = np.unique(locs)

                                        # Extract the voltage peaks
                                        array_fields[peak] = [
                                            np.array(dataset['x'][locs, self._lead_for_peaks]),
                                        ]

                            else:
                                for key, value in array_fields.items():
                                    if key not in (['waves', ] + list(self._peak_to_ind.keys())):
                                        array_fields[key] = np.concatenate(
                                            [value, np.expand_dims(dataset[key], 0)],
                                            axis=0,
                                        )

                                    elif key == 'waves' and not self._csv_only:
                                        array_fields[key].append(dataset['waves'][:])

                                    elif not self._csv_only:
                                        peak = dataset['x'][:, self._lead_for_peaks]
                                        peak = peak[dataset['waves'][self._peak_to_ind['R_Peak']].astype(np.int)]
                                        array_fields[key].append(peak)

                    if not self._csv_only:
                        # Concatenate waves base on the signal with the minimal amount of annotations
                        waves = array_fields['waves']
                        min_beats = min([w.shape[1] for w in waves])

                        if min_beats < self._min_allowed_beats:
                            continue

                        waves = np.concatenate(
                            [
                                np.expand_dims(w[:, :min_beats], 0)
                                for w in waves
                            ],
                            axis=0
                        )
                        array_fields['waves'] = waves

                        for peak in self._peak_to_ind:
                            peaks = np.concatenate(
                                [p[None, :min_beats] for p in array_fields[peak]],
                                axis=0
                            )
                            array_fields[peak] = peaks

                    patients_array_data.append(array_fields)

            if not self._csv_only:
                min_beats_for_patient = min(
                    [
                        p['waves'].shape[-1]
                        for p in patients_array_data
                    ]
                )

                for p in range(len(patients_array_data)):
                    patients_array_data[p]['waves'] = patients_array_data[p]['waves'][
                                                      ...,
                                                      :min_beats_for_patient,
                                                      ].astype(np.int)

                    for peak in self._peak_to_ind:
                        patients_array_data[p][peak] = patients_array_data[p][peak][
                                                       ...,
                                                       :min_beats_for_patient,
                                                       ]

            # Concatenate all of the sample into joint arrays
            patients_array_data = {
                key: np.concatenate(
                    [
                        np.expand_dims(p[key], 0)
                        for p in patients_array_data
                    ],
                    0
                )
                for key in patients_array_data[0]
            }

            # Set the path in which to save the prepared sample
            patient_id = file.split(os.sep)[
                [
                    i
                    for i, s in enumerate(file.split(os.sep))
                    if '10' in s
                ][0]
            ]
            os.makedirs(os.path.join(self._cache_path, patient_id), exist_ok=True)
            patient_data_file = os.path.join(
                self._cache_path, patient_id, f'{patient_id}.h5'
            )

            # Write the sample
            with h5py.File(patient_data_file, 'w') as file:
                dataset = file.create_group('/record')

                for key, val in patients_array_data.items():
                    dataset.create_dataset(
                        key,
                        shape=val.shape,
                        dtype=val.dtype,
                        data=val,
                    )

                for key, val in attributes_fields.items():
                    dataset.attrs[key] = val


class ECGRDVQDataset(Dataset):
    """
    Virtually combines multiple HDF5 files into one.
    This class can be wrapped in a PyTorch DataLoader which can then shuffle
    and lazy-load sample from all the files included in this virtual dataset.
    """

    TREATMENT_TIMES_SAMPLES_INDICES = {
        -0.5: (0, 242),
        0.5: (243, 485),
        1: (486, 728),
        1.5: (729, 971),
        2: (972, 1214),
        2.5: (1215, 1457),
        3: (1458, 1700),
        3.5: (1701, 1943),
        4: (1944, 2186),
        5: (2187, 2429),
        6: (2430, 2672),
        7: (2673, 2915),
        8: (2916, 3158),
        12: (3159, 3401),
        14: (3402, 3644),
        24: (3645, 3887),
    }

    TREATMENTS = {
        'Placebo': 0,
        'Ranolazine': 1,
        'Verapamil HCL': 2,
        'Quinidine Sulph': 3,
        'Dofetilide': 4,
    }

    def __init__(
            self,
            trajectory_length: int,
            prediction_horizon: int = 1,
            dir_path: str = None,
            randomized_orderings: bool = False,
            simple_prediction: bool = False,
            system_ind: Optional[int] = None,
            single_dynamics: bool = False,
    ):
        """
        :param trajectory_length: Number of time points to include in the trajectory of a single sample
        :param prediction_horizon: Number of time points to predict into the future
        :param dir_path: Path to filter with .h5 files.
        :param randomized_orderings: Whether or not to randomize the ordering selection between time-points
        (results in a combinatorial explosion of data points to evaluate)
        :param simple_prediction: For Koopman based models, we need to provide the GT predictions per
        each time-step (False), but for 'classical', such as ResNet, we just need to output the final predictions
        from the latest time-point (True).
        """

        super().__init__()

        assert trajectory_length < len(self.TREATMENT_TIMES_SAMPLES_INDICES)

        self.trajectory_length = trajectory_length
        self.prediction_horizon = prediction_horizon
        self.randomized_orderings = randomized_orderings
        self.simple_prediction = simple_prediction
        self.frequency = 1000
        self._m_systems = 5  # 1 system per treatment
        self._r_recordings_per_time_point = 3  # number of measurement done at each time point before / after treatment
        self.samples_per_time_after_treatment = self._r_recordings_per_time_point ** self._m_systems
        self.times_after_treatment = sorted(tuple(self.TREATMENT_TIMES_SAMPLES_INDICES.keys()))
        self._system_ind = system_ind
        self._single_dynamics = single_dynamics

        file_pattern = f'{dir_path}{os.path.sep}**{os.path.sep}*.h5'
        file_paths = sorted(glob.glob(file_pattern, recursive=True))
        self.file_paths = file_paths

        self.n_samples_per_file = (
                self.samples_per_time_after_treatment *
                (len(self.times_after_treatment) - trajectory_length - prediction_horizon + 1)
        )
        n_samples = (self.n_samples_per_file * len(file_paths)) if randomized_orderings else len(file_paths)
        self._length = n_samples

    def __enter__(self):

        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, index) -> Dict:
        """
        Implement the builtin getitem method

        :param index: (int) Index of the item to be fetched

        :return: (Dict) Dictionary with the sample at index 'index'
        """

        file_index = index // self.n_samples_per_file
        row_index = index % self.samples_per_time_after_treatment
        if self.randomized_orderings:
            x_rows_indices = [
                (i * self.samples_per_time_after_treatment) + np.random.choice(
                    a=np.arange(self.samples_per_time_after_treatment),
                    replace=False,
                    size=(1,),
                )
                for i in range(self.trajectory_length)
            ]
            x_rows_indices = np.array(x_rows_indices)[:, 0]

        else:
            x_rows_indices = np.arange(
                start=row_index,
                stop=(row_index + (self.trajectory_length * self.samples_per_time_after_treatment)),
                step=self.samples_per_time_after_treatment,
            )

        y_rows_indices = np.concatenate(
            [
                x_rows_indices + (self.samples_per_time_after_treatment * h)
                for h in range(1, self.prediction_horizon + 1)
            ]
        )

        argsort_y_rows_indices = np.argsort(y_rows_indices)
        y_rows_indices = np.sort(y_rows_indices)
        original_y_rows_indices = np.argsort(argsort_y_rows_indices)

        with h5py.File(self.file_paths[file_index], mode='r') as h5file:
            dataset = h5file['record']

            # Constant per patient
            sex = dataset['sex'][row_index] - 0.5
            age = dataset['age'][row_index] / 100
            height = dataset['height'][row_index] / 200
            weight = dataset['weight'][row_index] / 100

            # Depends on time after treatment
            x_time_post_treatment = dataset['time_post_treatment'][x_rows_indices] / 24
            x_avg_pr = dataset['avg_pr'][x_rows_indices] / self.frequency
            x_avg_qrs = dataset['avg_qrs'][x_rows_indices] / self.frequency
            x_avg_qt = dataset['avg_qt'][x_rows_indices] / self.frequency
            x_avg_rr = dataset['avg_rr'][x_rows_indices] / self.frequency
            x_lrd_30 = dataset['lrd_30'][x_rows_indices] / 100
            x_erd_30 = dataset['erd_30'][x_rows_indices] / 100
            x_t_amp = dataset['t_amp'][x_rows_indices] / 1000
            x_dosage = dataset['dosage'][x_rows_indices] / 1000

            # Replicate the constants per patient per time point
            x_sex = np.repeat(sex[None, ...], x_time_post_treatment.shape[0], 0)
            x_age = np.repeat(age[None, ...], x_time_post_treatment.shape[0], 0)
            x_height = np.repeat(height[None, ...], x_time_post_treatment.shape[0], 0)
            x_weight = np.repeat(weight[None, ...], x_time_post_treatment.shape[0], 0)

            # Gather the GT values to predict
            y_time_post_treatment = (dataset['time_post_treatment'][:][y_rows_indices] / 24)[original_y_rows_indices]
            y_avg_pr = (dataset['avg_pr'][:][y_rows_indices] / self.frequency)[original_y_rows_indices]
            y_avg_qrs = (dataset['avg_qrs'][:][y_rows_indices] / self.frequency)[original_y_rows_indices]
            y_avg_qt = (dataset['avg_qt'][:][y_rows_indices] / self.frequency)[original_y_rows_indices]
            y_avg_rr = (dataset['avg_rr'][:][y_rows_indices] / self.frequency)[original_y_rows_indices]
            y_lrd_30 = (dataset['lrd_30'][:][y_rows_indices] / 100)[original_y_rows_indices]
            y_erd_30 = (dataset['erd_30'][:][y_rows_indices] / 100)[original_y_rows_indices]
            y_t_amp = (dataset['t_amp'][:][y_rows_indices] / 1000)[original_y_rows_indices]
            y_dosage = (dataset['dosage'][:][y_rows_indices] / 1000)[original_y_rows_indices]

            # Replicate the constants per patient per time point
            y_sex = np.repeat(sex[None, ...], y_time_post_treatment.shape[0], 0)
            y_age = np.repeat(age[None, ...], y_time_post_treatment.shape[0], 0)
            y_height = np.repeat(height[None, ...], y_time_post_treatment.shape[0], 0)
            y_weight = np.repeat(weight[None, ...], y_time_post_treatment.shape[0], 0)

        x = np.concatenate(
            [
                x_time_post_treatment,
                x_avg_pr,
                x_avg_qrs,
                x_avg_qt,
                x_avg_rr,
                x_lrd_30,
                x_erd_30,
                x_t_amp,
                x_dosage,
                x_sex,
                x_age,
                x_height,
                x_weight,
            ],
            axis=-1,
        )
        y = np.concatenate(
            [
                y_time_post_treatment,
                y_avg_pr,
                y_avg_qrs,
                y_avg_qt,
                y_avg_rr,
                y_lrd_30,
                y_erd_30,
                y_t_amp,
                y_dosage,
                y_sex,
                y_age,
                y_height,
                y_weight,
            ],
            axis=-1,
        )

        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)
        x = from_numpy(x).type(torch.float32)
        y = from_numpy(y).type(torch.float32)

        if self.simple_prediction:
            y = y[:, -self.prediction_horizon:, :]

        if self._system_ind is not None:
            x = x[self._system_ind, :, :][None, :, :]
            y = y[self._system_ind, :, :][None, :, :]

        if self._single_dynamics:
            x = torch.swapaxes(x, axis0=0, axis1=1)
            x = x.reshape((1, x.shape[0], -1))
            y = torch.swapaxes(y, axis0=0, axis1=1)
            y = y.reshape((1, y.shape[0], -1))

        sample = {
            GT_TENSOR_INPUTS_KEY: x,
            GT_TENSOR_PREDICITONS_KEY: y,
        }

        return sample


class ECGRDVQLeaveOneOutDataLoadersGenerator:
    """
    Virtually combines multiple HDF5 files into one.
    This class can be wrapped in a PyTorch DataLoader which can then shuffle
    and lazy-load sample from all the files included in this virtual dataset.
    """

    def __init__(
            self,
            dir_paths: Sequence[str],
            datasets_save_dir: str,
            trajectory_length: int,
            prediction_horizon: int = 1,
            n_val_patients: int = 2,
            system_ind: Optional[int] = None,
            batch_size: int = 16,
            num_workers: int = 4,
            pin_memory: bool = True,
            drop_last: bool = False,
            simple_predictions: bool = False,
            single_dynamics: bool = False,
    ):
        self._dir_paths = dir_paths
        self._datasets_save_dir = datasets_save_dir
        self._trajectory_length = trajectory_length
        self._prediction_horizon = prediction_horizon
        self._n_val_patients = n_val_patients
        self._system_ind = system_ind
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._single_dynamics = single_dynamics

        # List all available patients
        files_paths = []
        for d_path in dir_paths:
            file_pattern = f'{d_path}{os.path.sep}**{os.path.sep}*.h5'
            file_paths = sorted(glob.glob(file_pattern, recursive=True))
            files_paths.extend(file_paths)

        # Generate all leave-1-out train/val/test splits
        train_sets = []
        test_sets = []
        validation_sets = []
        for i, patient in enumerate(files_paths):
            patients = files_paths.copy()
            test_sets.append((patient,))

            patients.pop(i)
            inds = np.arange(len(patients))
            val_inds = np.random.choice(a=inds, size=n_val_patients, replace=False)
            train_inds = np.setdiff1d(inds, val_inds)
            validation_sets.append(tuple(patients[p] for p in val_inds))
            train_sets.append(tuple(patients[p] for p in train_inds))

        # Write all leave-1-out train/val/test splits
        self._n_sets = len(train_sets)
        self._ds_paths_per_fold = []
        for i in range(self._n_sets):
            fold_dir = os.path.join(datasets_save_dir, f'Fold_{i}')
            os.makedirs(fold_dir, exist_ok=True)
            fold_dir_train = os.path.join(datasets_save_dir, fold_dir, 'Train')
            os.makedirs(fold_dir_train, exist_ok=True)
            fold_dir_val = os.path.join(datasets_save_dir, fold_dir, 'Val')
            os.makedirs(fold_dir_val, exist_ok=True)
            fold_dir_test = os.path.join(datasets_save_dir, fold_dir, 'Test')
            os.makedirs(fold_dir_test, exist_ok=True)

            self._ds_paths_per_fold.append(
                (fold_dir_train, fold_dir_val, fold_dir_test)
            )

            for p in train_sets[i]:
                patient_file = p.split(os.sep)[-1]
                new_file_path = os.path.join(fold_dir_train, patient_file)
                if not os.path.isfile(new_file_path):
                    shutil.copyfile(p, new_file_path)

            for p in validation_sets[i]:
                patient_file = p.split(os.sep)[-1]
                new_file_path = os.path.join(fold_dir_val, patient_file)
                if not os.path.isfile(new_file_path):
                    shutil.copyfile(p, new_file_path)

            for p in test_sets[i]:
                patient_file = p.split(os.sep)[-1]
                new_file_path = os.path.join(fold_dir_test, patient_file)
                if not os.path.isfile(new_file_path):
                    shutil.copyfile(p, new_file_path)

        # Generate all leave-1-out train/val/test datasets objects
        self._datasets_per_fold = []
        for i in range(self._n_sets):
            train_ds = ECGRDVQDataset(
                trajectory_length=trajectory_length,
                prediction_horizon=prediction_horizon,
                dir_path=self._ds_paths_per_fold[i][0],
                randomized_orderings=True,
                simple_prediction=simple_predictions,
                single_dynamics=single_dynamics,
            )
            val_ds = ECGRDVQDataset(
                trajectory_length=trajectory_length,
                prediction_horizon=prediction_horizon,
                dir_path=self._ds_paths_per_fold[i][1],
                randomized_orderings=True,
                simple_prediction=simple_predictions,
                single_dynamics=single_dynamics,
            )
            test_ds = ECGRDVQDataset(
                trajectory_length=trajectory_length,
                prediction_horizon=prediction_horizon,
                dir_path=self._ds_paths_per_fold[i][2],
                randomized_orderings=False,
                simple_prediction=simple_predictions,
                single_dynamics=single_dynamics,
            )
            self._datasets_per_fold.append(
                (train_ds, val_ds, test_ds)
            )

    def __len__(self) -> int:
        return self._n_sets

    def __getitem__(self, fold_index: int) -> (DataLoader, DataLoader, DataLoader):
        train_ds, val_ds, test_ds = self._datasets_per_fold[fold_index]

        train_dl = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

        dls = (train_dl, val_dl, test_dl)
        return dls


class CompositeDataset(Dataset):
    def __init__(
            self,
            dataset_1: Dataset,
            dataset_2: Dataset,
    ):
        super(CompositeDataset, self).__init__()

        assert len(dataset_1) == len(dataset_2)

        sample_1 = dataset_1[0]
        sample_2 = dataset_1[0]

        assert sorted(list(sample_1.keys())) == sorted(list(sample_2.keys()))

        self.ds1 = dataset_1
        self.ds2 = dataset_2
        self._length = len(dataset_1)

    def __len__(self):
        return self._length

    def __getitem__(self, index) -> Dict:
        sample_1 = self.ds1[index]
        sample_2 = self.ds2[index]

        sample = {
            key: torch.cat(
                [sample_1[key], sample_2[key]], dim=0,
            )
            for key in sample_1.keys()
        }

        return sample


class SSTDataset(Dataset):
    def __init__(
            self,
            mode: str,
            datasets_paths: Sequence[str],
            surface_mask_path: str,
            temporal_horizon: int,
            prediction_horizon: int = 1,
            val_ratio: float = 0.1,
            test_ratio: float = 0.2,
            top_lat: Union[int, Sequence[int]] = (40,),
            bottom_lat: Union[int, Sequence[int]] = (60,),
            left_long: Union[int, Sequence[int]] = (180,),
            right_long: Union[int, Sequence[int]] = (230,),
            block_len: int = 4,
            simple_prediction: bool = False,
            single_dynamics: bool = False,
    ):

        self._datasets_paths = datasets_paths
        self._temporal_horizon = temporal_horizon
        self._prediction_horizon = prediction_horizon
        self._top_lats = top_lat if isinstance(top_lat, Sequence) else (top_lat,)
        self._bottom_lat = bottom_lat if isinstance(bottom_lat, Sequence) else (bottom_lat,)
        self._left_long = left_long if isinstance(left_long, Sequence) else (left_long,)
        self._right_long = right_long if isinstance(right_long, Sequence) else (right_long,)
        self._block_len = block_len
        self._mode = mode
        self._simple_prediction = simple_prediction
        self._single_dynamics = single_dynamics

        assert len(self._top_lats) == len(self._bottom_lat)
        assert len(self._top_lats) == len(self._left_long)
        assert len(self._top_lats) == len(self._right_long)
        assert mode in ("Train", "Test", "Val")

        mask = nc.Dataset(surface_mask_path)

        blocks = []
        lengths = []
        for k in range(len(self._top_lats)):
            mask_array = np.array(
                mask.variables['mask'][
                :, self._top_lats[k]:self._bottom_lat[k], self._left_long[k]:self._right_long[k]
                ]
            )
            mask_blocks = self._extract_blocks(mask_array, block_len)
            for i, dsp in enumerate(datasets_paths):
                ds = nc.Dataset(dsp)

                if i == 0:
                    times = nc.num2date(ds.variables['time'], ds.variables['time'].units)
                    self.times = [t.strftime("%Y-%m-%d %H:%M").split(' ')[0] for t in times]

                sst = (
                    ds.variables['sst'][
                    :, self._top_lats[k]:self._bottom_lat[k], self._left_long[k]:self._right_long[k]
                    ]
                )
                sst_blocks = self._extract_blocks(sst, block_len)
                valid_sst_blocks = [
                    np.reshape(b, (b.shape[0], 1, -1))
                    for i, b in enumerate(sst_blocks)
                    if mask_blocks[i].sum() == mask_blocks[i].size
                ]
                blocks.append(valid_sst_blocks)
                lengths.append(valid_sst_blocks[0].shape[0])

        blocks = [
            bb
            for b in blocks
            for bb in b
        ]
        self.blocks = np.concatenate(blocks, axis=1)
        assert all([l == lengths[0] for l in lengths])
        total_len = lengths[0]

        self._train_end = int((1 - val_ratio - test_ratio) * total_len)
        self._val_end = self._train_end + int(total_len * val_ratio)

        self.train_times = self.times[:self._train_end]
        self.val_times = self.times[self._train_end:self._val_end]
        self.test_times = self.times[self._val_end:]

        if mode == 'Train':
            self._offset = 0
            self._end = self._train_end
            length = self._end - temporal_horizon - prediction_horizon

        elif mode == 'Val':
            self._offset = self._train_end
            self._end = self._val_end
            length = (self._end - self._offset) - temporal_horizon - prediction_horizon

        else:
            self._offset = self._val_end
            self._end = None
            length = (total_len - self._offset) - temporal_horizon - prediction_horizon

        self._length = length

    @staticmethod
    def _extract_blocks(data: np.ndarray, block_len: int) -> Sequence[np.ndarray]:
        blocks = [
            data[:, (i * block_len):((i + 1) * block_len), (j * block_len):((j + 1) * block_len)]
            for i in range(data.shape[1] // block_len)
            for j in range(data.shape[2] // block_len)
        ]

        return blocks

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, index) -> Dict:
        x = self.blocks[(index + self._offset):(index + self._offset + self._temporal_horizon)]
        x = np.swapaxes(x, 1, 0)

        y = [
            self.blocks[
            (
                    index + self._offset + h + 1
            ):(
                    index + self._offset + self._temporal_horizon + h + 1
            )
            ]
            for h in range(self._prediction_horizon)
        ]
        y = np.concatenate(y, axis=0)
        y = np.swapaxes(y, 1, 0)

        x = from_numpy(x).type(torch.float32)
        y = from_numpy(y).type(torch.float32)

        if self._simple_prediction:
            y = y[:, (self._temporal_horizon - 1)::self._temporal_horizon, :]

        if self._single_dynamics:
            x = torch.swapaxes(x, axis0=0, axis1=1)
            x = x.reshape((1, x.shape[0], -1))
            y = torch.swapaxes(y, axis0=0, axis1=1)
            y = y.reshape((1, y.shape[0], -1))

        sample = {
            GT_TENSOR_INPUTS_KEY: x,
            GT_TENSOR_PREDICITONS_KEY: y,
        }

        return sample
