from abc import ABC

import os
import h5py
import numpy as np


class PhysioRecorder(ABC):
    """
    A utility class for converting raw PhysioNet recordings into HDF5 files which
    can be easily used as a PyTorch Dataset.
    """

    def __init__(self, db_name: str, save_dir: str):
        """
        Constructor of the PhysioRecorder.

        :param db_name: (str) The name of the database to load. This is crucial since
        different databases are saved under different formats and structures.
        :param save_dir: (str) Path to the directory in which to store the processed
        database. If None, defaults to .../DynamicalSystems/data/db_name
        """

        super(PhysioRecorder, self).__init__()

        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir
        self._db_name = db_name

    def convert_single_record(
            self,
            x: np.ndarray,
            raw_file_name: str,
            y: np.ndarray = None,
            record_id: int = None,
            raw_qrs_inds: np.ndarray = None,
            additional_arrays: dict = None,
            additional_attributes: dict = None,
            save_dir: str = None,
    ) -> None:
        """
        This method converts a single record, specified as a numpy array, together with
        its' given record id (if given), raw file name, original database
        and raw qrs indices, in order to allow full backwards
        tracking of the data's origin.

        :param x: (np.ndarray) The record to convert
        :param raw_file_name: (str) The name of the raw file
        :param y: (np.ndarray) The labels, i.e. rhythm annotations, if not None.
        :param record_id: (int) A 'record id', if not None.
        :param raw_qrs_inds: (np.ndarray) The original qrs indices corresponding
        to the recorded data, i.e. rhythm annotations, if not None.
        :param additional_arrays: (dict) Any additional arrays to record
        :param additional_attributes: (dict) Any additional attributes to record
        :param save_dir: (str) Directory in which to save the record, if None uses the directory given at the
        constructor.

        :return: None
        """

        if save_dir is None:
            save_dir = self._save_dir

        filename = os.path.join(save_dir, raw_file_name)
        filename = filename if filename.endswith('.h5') else filename + '.h5'

        # Generate corresponding record id for all samples, if relevant.
        if record_id is not None:
            record_ids = np.array([record_id])

        else:
            record_ids = np.array([])

        # If no labels are given, specify them as an empty array
        if y is None:
            y = np.array([])

        # If no raw qrs indices are given, specify them as an empty array
        if raw_qrs_inds is None:
            raw_qrs_inds = np.array([])

        # Build the hdf5 recording
        with h5py.File(filename, 'w') as file:
            dataset = file.create_group('/record')

            dataset.create_dataset('x', shape=x.shape, dtype=x.dtype, data=x)
            dataset.create_dataset('y', shape=y.shape, dtype=y.dtype, data=y)
            dataset.create_dataset(
                'id',
                shape=record_ids.shape,
                dtype=record_ids.dtype,
                data=record_ids,
            )
            dataset.create_dataset(
                'qrs',
                shape=raw_qrs_inds.shape,
                dtype=raw_qrs_inds.dtype,
                data=raw_qrs_inds,
            )

            if additional_arrays is not None:
                for key, val in additional_arrays.items():
                    dataset.create_dataset(
                        key,
                        shape=val.shape,
                        dtype=val.dtype,
                        data=val,
                    )

            dataset.attrs["db_name"] = self._db_name
            dataset.attrs["record"] = raw_file_name

            if additional_attributes is not None:
                for key, val in additional_attributes.items():
                    dataset.attrs[key] = val

    def __call__(
            self,
            x: np.ndarray,
            raw_file_name: str,
            y: np.ndarray = None,
            record_id: int = None,
            raw_qrs_inds: np.ndarray = None,
            additional_arrays: dict = None,
            additional_attributes: dict = None,
            save_dir: str = None,
    ) -> None:
        return self.convert_single_record(
            x=x,
            raw_file_name=raw_file_name,
            y=y,
            record_id=record_id,
            raw_qrs_inds=raw_qrs_inds,
            additional_arrays=additional_arrays,
            additional_attributes=additional_attributes,
            save_dir=save_dir,
        )
